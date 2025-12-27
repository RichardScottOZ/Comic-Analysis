#!/bin/bash
# Build and deploy Lithops OCR runtime the correct way
set -e

echo "=== Building Lithops OCR Runtime (The Right Way) ==="
echo ""

# Create temp directory
TEMP_DIR=$(mktemp -d)
cd "$TEMP_DIR"

echo "Step 1: Downloading official Lithops Dockerfile..."
wget -q https://raw.githubusercontent.com/lithops-cloud/lithops/master/runtime/aws_lambda/Dockerfile

echo "Step 2: Adding Tesseract, EasyOCR AND PaddleOCR dependencies to Dockerfile..."
# Add Tesseract, EasyOCR, and PaddleOCR with required system libraries
sed -i '/# Put your dependencies here/a \
# OCR dependencies - Tesseract, EasyOCR, and PaddleOCR\
RUN apt-get update && apt-get install -y \
    tesseract-ocr tesseract-ocr-eng \
    libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*\
RUN pip install --no-cache-dir pytesseract Pillow\
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu\
RUN pip install --no-cache-dir easyocr opencv-python-headless\
RUN pip install --no-cache-dir paddlepaddle==2.6.2 paddleocr==2.7.0.3\
# Pre-download models\
RUN python -c "import easyocr; easyocr.Reader(['"'"'en'"'"'], gpu=False, download_enabled=True)" || true\
RUN python -c "from paddleocr import PaddleOCR; PaddleOCR(use_angle_cls=True, lang='"'"'en'"'"', use_gpu=False)" || true
' Dockerfile

echo ""
echo "Modified Dockerfile ready."
echo "Cleaning up old images to force fresh build..."
echo "This will take ~15-20 minutes"
echo ""

# Clean up to force fresh build
aws ecr batch-delete-image \
  --repository-name lithops_v362_vztq/tesseract-ocr-runtime \
  --image-ids imageTag=latest \
  --region us-east-1 2>/dev/null || echo "No existing ECR image"

sudo docker rmi tesseract-ocr-runtime:latest 2>/dev/null || echo "No local image to remove"

# Delete Lambda functions
aws lambda list-functions --region us-east-1 --query 'Functions[?starts_with(FunctionName, `lithops-worker-vztq-362`)].FunctionName' --output text | \
  xargs -r -I {} aws lambda delete-function --function-name {} --region us-east-1 2>/dev/null || echo "No Lambda functions to delete"

echo ""
echo "Building with Lithops (this handles the lithops_lambda.zip automatically)..."
echo ""

cd /mnt/c/users/Richard/OneDrive/GIT/Comic-Analysis
sudo chmod 666 /var/run/docker.sock

# Let Lithops handle everything - it creates lithops_lambda.zip and builds
sudo -E /home/richard/miniconda3/envs/comicanalysis/bin/lithops runtime build -b aws_lambda -f "$TEMP_DIR/Dockerfile" tesseract-ocr-runtime

echo ""
echo "Step 3: Deploying runtime to AWS Lambda..."
lithops runtime deploy tesseract-ocr-runtime -b aws_lambda

echo ""
echo "Step 4: Updating Lithops config to use new runtime..."

# Get execution role
ROLE_ARN=$(aws iam get-role --role-name lithops-execution-role --query 'Role.Arn' --output text)

cat > ~/.lithops/config << EOF
lithops:
  backend: aws_lambda
  storage: aws_s3
  check_version: false  # Ignore Python version mismatch (3.11 local vs 3.10 Lambda)

aws:
  region: us-east-1

aws_lambda:
  execution_role: ${ROLE_ARN}
  runtime: tesseract-ocr-runtime
  runtime_memory: 10240
  runtime_timeout: 600

aws_s3:
  storage_bucket: calibrecomics-extracted
  region: us-east-1
EOF

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Config location: ~/.lithops/config"
cat ~/.lithops/config
echo ""
echo "Runtime deployed: tesseract-ocr-runtime (now includes Tesseract, EasyOCR AND PaddleOCR)"
echo ""
echo "You can now run:"
echo "  - Tesseract: ./run_ocr_test.sh"
echo "  - EasyOCR: cd src/version1 && python batch_ocr_easyocr_lithops.py --manifest ../../manifests/test_manifest100.csv --output-bucket calibrecomics-extracted --output-prefix ocr_results_easyocr/neon_test --workers 50"
echo "  - PaddleOCR: cd src/version1 && python batch_ocr_paddleocr_lithops.py --manifest ../../manifests/test_manifest100.csv --output-bucket calibrecomics-extracted --output-prefix ocr_results_paddleocr/neon_test --workers 50"
echo ""

# Cleanup
rm -rf "$TEMP_DIR"
