# CoSMo PSS Lithops Docker Runtimes

This directory contains two production-ready Dockerfiles for running CoSMo PSS with Lithops.

## Dockerfiles

### 1. Dockerfile.lithops.cpu (Recommended for Most Use Cases)

**Best for:**
- AWS Lambda
- Azure Functions  
- Google Cloud Functions
- IBM Cloud Functions
- Cost-sensitive deployments

**Features:**
- CPU-only PyTorch (smaller image size: ~2GB)
- Python 3.10 slim base
- All required ML dependencies
- Optimized for serverless FaaS platforms

**Build:**
```bash
docker build -f Dockerfile.lithops.cpu -t cosmo-pss-runtime-cpu .
```

**Expected Performance:**
- ~100ms per page on Lambda (10GB memory)
- 1.2M pages in 2-4 hours with 500-1000 workers
- Cost: ~$2,400 for 1.2M pages

### 2. Dockerfile.lithops.gpu (High Performance)

**Best for:**
- AWS Batch with GPU instances (g4dn.xlarge, g4dn.2xlarge)
- Azure Container Instances with GPU
- GCP Cloud Run with GPU
- Performance-critical deployments

**Features:**
- NVIDIA CUDA 11.8 + cuDNN 8
- GPU-enabled PyTorch
- Ubuntu 22.04 base
- All required ML dependencies

**Build:**
```bash
docker build -f Dockerfile.lithops.gpu -t cosmo-pss-runtime-gpu .
```

**Requirements:**
- NVIDIA Docker runtime
- GPU-enabled compute instances

**Expected Performance:**
- ~30ms per page on g4dn.xlarge (1x NVIDIA T4)
- 1.2M pages in 1-2 hours with 100-200 workers
- Cost: ~$520 for 1.2M pages

## Quick Start

### AWS Lambda (CPU)

```bash
# Build and push
docker build -f Dockerfile.lithops.cpu -t cosmo-pss-runtime-cpu .
aws ecr create-repository --repository-name cosmo-pss-runtime-cpu
docker tag cosmo-pss-runtime-cpu:latest 123456789012.dkr.ecr.us-east-1.amazonaws.com/cosmo-pss-runtime-cpu:latest
docker push 123456789012.dkr.ecr.us-east-1.amazonaws.com/cosmo-pss-runtime-cpu:latest

# Configure lithops
lithops runtime create cosmo-pss-runtime-cpu --backend aws_lambda
```

### AWS Batch (GPU)

```bash
# Build and push
docker build -f Dockerfile.lithops.gpu -t cosmo-pss-runtime-gpu .
aws ecr create-repository --repository-name cosmo-pss-runtime-gpu
docker tag cosmo-pss-runtime-gpu:latest 123456789012.dkr.ecr.us-east-1.amazonaws.com/cosmo-pss-runtime-gpu:latest
docker push 123456789012.dkr.ecr.us-east-1.amazonaws.com/cosmo-pss-runtime-gpu:latest

# Configure lithops
lithops runtime create cosmo-pss-runtime-gpu --backend aws_batch
```

## Image Sizes

| Dockerfile | Base Image | Final Size | Notes |
|------------|------------|------------|-------|
| CPU | python:3.10-slim | ~2.0 GB | Optimized for Lambda |
| GPU | nvidia/cuda:11.8 | ~6.5 GB | Includes CUDA toolkit |

## Verification

### Test CPU Runtime
```bash
docker run --rm cosmo-pss-runtime-cpu python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

Expected output:
```
PyTorch: 2.1.0+cpu
CUDA available: False
```

### Test GPU Runtime
```bash
docker run --rm --gpus all cosmo-pss-runtime-gpu python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"
```

Expected output:
```
PyTorch: 2.1.0+cu118
CUDA available: True
CUDA version: 11.8
```

## Cost Comparison (1.2 Million Pages)

| Runtime | Platform | Workers | Time | Cost |
|---------|----------|---------|------|------|
| **CPU** | AWS Lambda | 500-1000 | 2-4 hours | **~$2,400** ✓ Most economical |
| **GPU** | AWS Batch g4dn.xlarge | 100-200 | 1-2 hours | **~$520** ✓ Fastest |
| **CPU** | Azure Functions | 500-1000 | 2-4 hours | **~$2,200** |
| **GPU** | Azure Container GPU | 100-200 | 1-2 hours | **~$480** |

## Recommendations

1. **Start with CPU**: Most cost-effective for initial testing and small-to-medium datasets
2. **Use GPU for**: 
   - Time-critical processing (need results in < 2 hours)
   - Very large datasets (> 5M pages) where time savings justify cost
   - When GPU instances are already available
3. **Production**: Use CPU for regular batch processing, GPU for urgent re-processing

## Troubleshooting

### CPU Docker Build Fails
```
Error: Unable to find image 'python:3.10-slim'
```
**Solution**: Pull base image first: `docker pull python:3.10-slim`

### GPU Docker Build Fails
```
Error: Unable to find image 'nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04'
```
**Solution**: Pull base image first: `docker pull nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04`

### GPU Runtime Shows CUDA Unavailable
```
CUDA available: False
```
**Solution**: Ensure NVIDIA Docker runtime is installed and use `--gpus all` flag

## Further Reading

- [Lithops Documentation](https://lithops-cloud.github.io/docs/)
- [AWS Lambda Container Images](https://docs.aws.amazon.com/lambda/latest/dg/images-create.html)
- [AWS Batch GPU Configuration](https://docs.aws.amazon.com/batch/latest/userguide/gpu-jobs.html)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
