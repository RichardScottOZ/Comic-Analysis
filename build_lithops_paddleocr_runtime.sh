#!/bin/bash
set -e

# Function to verify Docker daemon accessibility
check_docker() {
  if ! docker info >/dev/null 2>&1; then
    echo "ERROR: Cannot connect to Docker daemon."
    echo "Make sure Docker Desktop is running and WSL integration is enabled, or run this script with sudo."
    exit 1
  fi
}

# Verify Docker before proceeding
check_docker

echo "=== Building Lithops PaddleOCR Runtime ==="
echo ""

# Create temp directory
TEMP_DIR=$(mktemp -d)
echo "Step 1: Downloading official Lithops Dockerfile..."
curl -s https://raw.githubusercontent.com/lithops-cloud/lithops/master/runtime/aws_lambda/Dockerfile > "$TEMP_DIR/Dockerfile"

echo "Step 2: Adding PaddleOCR dependencies to Dockerfile..."
echo ""

# Insert PaddleOCR-specific dependencies after the lithops handler setup
cat > "$TEMP_DIR/Dockerfile" << 'DOCKERFILE_END'
FROM python:3.10-slim-trixie

RUN apt-get update \
    && apt-get install -y \
      g++ \
      make \
      cmake \
      unzip \
    && rm -rf /var/lib/apt/lists/* \
    && apt-cache search linux-headers-generic

ARG FUNCTION_DIR="/function"
RUN mkdir -p ${FUNCTION_DIR}

RUN pip install --upgrade --ignore-installed pip wheel six setuptools \
    && pip install --upgrade --no-cache-dir --ignore-installed \
        awslambdaric \
        boto3 \
        redis \
        httplib2 \
        requests \
        numpy \
        scipy \
        pandas \
        pika \
        kafka-python \
        cloudpickle \
        ps-mem \
        tblib \
        psutil

WORKDIR ${FUNCTION_DIR}
COPY lithops_lambda.zip ${FUNCTION_DIR}

RUN unzip lithops_lambda.zip \
    && rm lithops_lambda.zip \
    && mkdir handler \
    && touch handler/__init__.py \
    && mv entry_point.py handler/

# Install system dependencies for PaddleOCR
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Install PaddleOCR and dependencies in correct order
RUN pip install --no-cache-dir \
    opencv-python-headless==4.8.1.78 \
    opencv-contrib-python-headless==4.8.1.78

# CRITICAL: Install CPU-only PaddlePaddle (GPU version will segfault in Lambda)
# Use 2.6.2 stable - last version confirmed compatible with paddleocr 2.7.0.3
RUN pip install --no-cache-dir paddlepaddle==2.6.2

# Then install PaddleOCR and its specific dependencies
RUN pip install --no-cache-dir paddlepaddle==2.6.0 -f https://paddlepaddle.org.cn/whl/cpu.html


# Pre-download PaddleOCR models
RUN python -c "from paddleocr import PaddleOCR; ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False, show_log=False)" || true

ENTRYPOINT [ "/usr/local/bin/python", "-m", "awslambdaric" ]
CMD [ "handler.entry_point.lambda_handler" ]
DOCKERFILE_END

echo "Modified Dockerfile ready."
echo "Cleaning up old images to force fresh build..."
echo "This will take ~15-20 minutes"
echo ""

# Delete old ECR image
aws ecr batch-delete-image \
    --repository-name lithops_v362_vztq/paddleocr-runtime \
    --image-ids imageTag=latest \
    --region us-east-1 2>/dev/null || echo "No existing ECR image to delete"

# Remove local Docker image
docker rmi paddleocr-runtime:latest 2>/dev/null || echo "No local image to remove"

# Clear build cache
docker builder prune -af

# Delete any existing Lambda functions
aws lambda list-functions --region us-east-1 --query 'Functions[?contains(FunctionName, `paddleocr`)].FunctionName' --output text | \
    xargs -I {} aws lambda delete-function --function-name {} --region us-east-1 2>/dev/null || echo "No Lambda functions to delete"

echo ""
echo "Building with Lithops (this handles the lithops_lambda.zip automatically)..."
echo ""

# Build with Lithops (run without sudo - lithops in user env)
lithops runtime build -b aws_lambda -f "$TEMP_DIR/Dockerfile" paddleocr-runtime

echo ""
echo "Step 3: Deploying runtime to AWS Lambda..."
echo "Deleting any existing Lambda functions to force fresh image pull..."
aws lambda list-functions --region us-east-1 --query 'Functions[?contains(FunctionName, `paddleocr`)].FunctionName' --output text | \
    xargs -I {} aws lambda delete-function --function-name {} --region us-east-1 2>/dev/null || echo "No Lambda functions to delete"

lithops runtime deploy paddleocr-runtime -b aws_lambda --memory 10240 --timeout 600

echo ""
echo "Step 4: Creating separate Lithops config for PaddleOCR..."
echo ""

# Create PaddleOCR-specific config
cat > ~/.lithops/config_paddleocr << 'EOF'
lithops:
  backend: aws_lambda
  storage: aws_s3
  check_version: false

aws:
  region: us-east-1

aws_lambda:
  execution_role: arn:aws:iam::958539196701:role/lithops-execution-role
  runtime: paddleocr-runtime
  runtime_memory: 10240
  runtime_timeout: 600

aws_s3:
  storage_bucket: calibrecomics-extracted
  region: us-east-1
EOF

echo "=== Setup Complete ==="
echo ""
echo "PaddleOCR config location: ~/.lithops/config_paddleocr"
echo ""
echo "Runtime deployed: paddleocr-runtime (PaddleOCR only)"
echo ""
echo "You can now run PaddleOCR:"
echo "  cd src/version1 && LITHOPS_CONFIG_FILE=~/.lithops/config_paddleocr python batch_ocr_paddleocr_lithops.py --manifest ../../manifests/test_manifest100.csv --output-bucket calibrecomics-extracted --output-prefix ocr_results_paddleocr/neon_test --workers 50"
echo ""

# Ensure Docker config directory exists with correct permissions
mkdir -p "$HOME/.docker"
chmod 700 "$HOME/.docker"
# Create an empty config file if missing and restrict permissions
if [ ! -f "$HOME/.docker/config.json" ]; then
  touch "$HOME/.docker/config.json"
fi
chmod 600 "$HOME/.docker/config.json"

# If the user is not in the docker group, suggest adding them (no sudo in script)
if ! groups $(whoami) | grep -qw docker; then
  echo "WARNING: User $(whoami) is not in the 'docker' group. Docker commands may require sudo."
  echo "You can add the user to the group with: sudo usermod -aG docker $(whoami) && newgrp docker"
fi
