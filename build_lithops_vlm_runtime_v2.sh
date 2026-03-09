#!/bin/bash
set -e

# Function to verify Docker daemon accessibility
check_docker() {
  if ! docker info >/dev/null 2>&1; then
    echo "ERROR: Cannot connect to Docker daemon."
    exit 1
  fi
}

check_docker

echo "=== Building Lithops Comic VLM Runtime v2 ==="

# 1. SURGICAL CLEANUP: Delete only this specific broken runtime
echo "Deleting existing 'comic-vlm-v2-runtime' (if any)..."
# Ignore errors if it doesn't exist
lithops runtime delete -b aws_lambda comic-vlm-v2-runtime 2>/dev/null || echo "No existing runtime to delete."

echo ""
TEMP_DIR=$(mktemp -d)
echo "Step 1: Downloading official Lithops Dockerfile..."
curl -s https://raw.githubusercontent.com/lithops-cloud/lithops/master/runtime/aws_lambda/Dockerfile > "$TEMP_DIR/Dockerfile"

echo "Step 2: Customizing Dockerfile..."

# Added all essential Lithops dependencies (pika, redis, tblib, etc.)
cat > "$TEMP_DIR/Dockerfile" << 'DOCKERFILE_END'
FROM python:3.10-slim-trixie

RUN apt-get update \
    && apt-get install -y \
      g++ \
      make \
      cmake \
      unzip \
    && rm -rf /var/lib/apt/lists/*

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
        pandas \
        pillow \
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

ENTRYPOINT [ "/usr/local/bin/python", "-m", "awslambdaric" ]
CMD [ "handler.entry_point.lambda_handler" ]
DOCKERFILE_END

echo "Dockerfile ready."

echo "Step 3: Building and Deploying with Lithops..."
# This handles ECR build/push and Lambda registration
lithops runtime build -b aws_lambda -f "$TEMP_DIR/Dockerfile" comic-vlm-v2-runtime
lithops runtime deploy comic-vlm-v2-runtime -b aws_lambda --memory 1024 --timeout 600

echo ""
echo "Step 4: Updating Lithops config..."
cat > ~/.lithops/config_comic_vlm_v2 << 'EOF'
lithops:
  backend: aws_lambda
  storage: aws_s3
  check_version: false

aws:
  region: us-east-1

aws_lambda:
  execution_role: arn:aws:iam::958539196701:role/lithops-execution-role
  runtime: comic-vlm-v2-runtime
  runtime_memory: 1024
  runtime_timeout: 600

aws_s3:
  storage_bucket: calibrecomics-extracted
  region: us-east-1
EOF

echo "=== Setup Complete ==="
echo "Config: ~/.lithops/config_comic_vlm_v2"
echo "Runtime: comic-vlm-v2-runtime"
