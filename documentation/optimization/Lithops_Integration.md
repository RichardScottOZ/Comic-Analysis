# CoSMo PSS with Lithops - Serverless Distributed Computing

## Overview

This guide explains how to run the CoSMo Page Stream Segmentation (PSS) embedding precomputation pipeline using [Lithops](https://github.com/lithops-cloud/lithops), a Python framework for serverless and distributed computing.

## Why Lithops?

Lithops enables:
- **Massive Parallelization**: Process hundreds/thousands of books simultaneously
- **Cost Efficiency**: Pay only for compute time used (serverless)
- **Cloud Native**: Works with AWS Lambda, Azure Functions, Google Cloud Functions, IBM Cloud Functions
- **Scalability**: Automatically scales from 1 to 1000+ workers
- **Storage Integration**: Direct integration with S3, Azure Blob, GCS, IBM COS

### Performance Comparison

**Dataset Assumptions:**
- 1000 books = ~30,000 pages average (30 pages per book typical for comic books)
- Processing time: ~100ms per page with SigLIP so400m on CPU, ~30ms on GPU

| Method | Books | Pages | Time | Cost* | Notes |
|--------|-------|-------|------|-------|-------|
| Local Sequential (GPU) | 1000 | 30,000 | ~15 hours | $0 | Single GPU machine |
| Local Sequential (CPU) | 1000 | 30,000 | ~50 hours | $0 | Single CPU machine |
| Local Parallel (4 GPUs) | 1000 | 30,000 | ~4 hours | $0 | Multi-GPU machine |
| Lithops CPU (100 workers) | 1000 | 30,000 | ~30 minutes | ~$75 | AWS Lambda + S3 |
| Lithops CPU (500 workers) | 1000 | 30,000 | ~6 minutes | ~$75 | AWS Lambda + S3 |
| Lithops GPU (50 workers) | 1000 | 30,000 | ~10 minutes | ~$150 | AWS Batch g4dn.xlarge |

*Costs are estimates for AWS us-east-1 and vary by cloud provider and region.

### Large-Scale Cost Analysis (1.2 Million Pages)

For a dataset of **1.2 million pages** (~40,000 books at 30 pages/book):

| Method | Time | AWS Cost | Azure Cost | GCP Cost | Notes |
|--------|------|----------|------------|----------|-------|
| **CPU-Based (Recommended)** |
| Lithops Lambda (500 workers) | ~4 hours | **~$3,000** | ~$2,800 | ~$3,200 | Most cost-effective |
| Lithops Lambda (1000 workers) | ~2 hours | **~$3,000** | ~$2,800 | ~$3,200 | Faster, same cost |
| **GPU-Based (High Performance)** |
| Lithops Batch GPU (100 workers) | ~2 hours | **~$6,000** | ~$5,500 | ~$6,500 | g4dn.xlarge instances |
| Lithops Batch GPU (200 workers) | ~1 hour | **~$6,000** | ~$5,500 | ~$6,500 | Faster processing |

**Cost Breakdown for 1.2M Pages (AWS Lambda CPU, 500 workers):**
- Lambda Compute: 10GB memory × 1.2M × 1 second = 12M GB-seconds × $0.0000166667 = **~$2,000**
- Lambda Requests: 40,000 invocations × $0.20/1M = **~$8**
- S3 Storage: 600GB embeddings × $0.023/GB/month = **~$14/month**
- S3 PUT Requests: 80,000 × $0.005/1000 = **~$400**
- Data Transfer: 30GB × $0.09/GB = **~$270**
- **Total One-Time Cost: ~$2,678** (rounded to ~$3,000 with overhead)

**Recommendation for 1.2M pages:** Use **CPU-based Lambda with 500-1000 workers** for optimal cost/performance ratio.

## Architecture

```
┌─────────────────┐
│   Annotations   │
│  (comics.json)  │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────┐
│     Lithops Orchestrator            │
│  (lithops_precompute.py)            │
└────────┬────────────────────────────┘
         │
         ├──► Worker 1 ──► Process Book 1 ──► S3/Azure/GCS
         ├──► Worker 2 ──► Process Book 2 ──► S3/Azure/GCS
         ├──► Worker 3 ──► Process Book 3 ──► S3/Azure/GCS
         ├──► Worker N ──► Process Book N ──► S3/Azure/GCS
         │
         ▼
┌─────────────────────────────────────┐
│   Cloud Storage (Embeddings)        │
│   - visual/book_id.pt               │
│   - text/book_id.pt                 │
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│   Classification Phase              │
│   (classify_precomputed.py)         │
└─────────────────────────────────────┘
```

## Setup

### 1. Install Lithops

```bash
pip install lithops[aws]  # For AWS
# or
pip install lithops[azure]  # For Azure
# or
pip install lithops[gcp]  # For GCP
```

### 2. Configure Cloud Credentials

#### AWS
```bash
# Option 1: Environment variables
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_DEFAULT_REGION=us-east-1

# Option 2: AWS CLI
aws configure

# Create S3 bucket for embeddings
aws s3 mb s3://cosmo-pss-embeddings
```

#### Azure
```bash
# Set environment variables
export AZURE_STORAGE_ACCOUNT=your_account
export AZURE_STORAGE_KEY=your_key

# Create blob container
az storage container create --name cosmo-pss-embeddings
```

#### GCP
```bash
# Authenticate
gcloud auth login
gcloud config set project your-project-id

# Create GCS bucket
gsutil mb gs://cosmo-pss-embeddings
```

### 3. Build Custom Runtime

The CoSMo PSS pipeline requires a custom runtime with ML dependencies. **Two Dockerfiles are provided:**

#### Option A: CPU Runtime (Recommended for Serverless)
Best for AWS Lambda, Azure Functions, Google Cloud Functions. Smaller image size, lower cost.

```bash
# Build CPU-only Docker image
docker build -f Dockerfile.lithops.cpu -t cosmo-pss-runtime-cpu .

# Push to AWS ECR
aws ecr create-repository --repository-name cosmo-pss-runtime-cpu
docker tag cosmo-pss-runtime-cpu:latest <account>.dkr.ecr.<region>.amazonaws.com/cosmo-pss-runtime-cpu:latest
docker push <account>.dkr.ecr.<region>.amazonaws.com/cosmo-pss-runtime-cpu:latest

# Register runtime with Lithops
lithops runtime create cosmo-pss-runtime-cpu --backend aws_lambda
```

#### Option B: GPU Runtime (High Performance)
Best for AWS Batch, Azure Container Instances with GPU. Faster processing, higher cost.

```bash
# Build GPU Docker image
docker build -f Dockerfile.lithops.gpu -t cosmo-pss-runtime-gpu .

# Push to AWS ECR
aws ecr create-repository --repository-name cosmo-pss-runtime-gpu
docker tag cosmo-pss-runtime-gpu:latest <account>.dkr.ecr.<region>.amazonaws.com/cosmo-pss-runtime-gpu:latest
docker push <account>.dkr.ecr.<region>.amazonaws.com/cosmo-pss-runtime-gpu:latest

# Register runtime with Lithops for AWS Batch
lithops runtime create cosmo-pss-runtime-gpu --backend aws_batch
```

#### Azure Container Registry (CPU or GPU)
```bash
az acr create --resource-group myResourceGroup --name cosmopssregistry --sku Basic
az acr login --name cosmopssregistry

# For CPU
docker tag cosmo-pss-runtime-cpu:latest cosmopssregistry.azurecr.io/cosmo-pss-runtime-cpu:latest
docker push cosmopssregistry.azurecr.io/cosmo-pss-runtime-cpu:latest

# For GPU
docker tag cosmo-pss-runtime-gpu:latest cosmopssregistry.azurecr.io/cosmo-pss-runtime-gpu:latest
docker push cosmopssregistry.azurecr.io/cosmo-pss-runtime-gpu:latest
```

#### Google Container Registry (CPU or GPU)
```bash
# Configure Docker for GCR
gcloud auth configure-docker

# For CPU
docker tag cosmo-pss-runtime-cpu:latest gcr.io/<project-id>/cosmo-pss-runtime-cpu:latest
docker push gcr.io/<project-id>/cosmo-pss-runtime-cpu:latest

# For GPU
docker tag cosmo-pss-runtime-gpu:latest gcr.io/<project-id>/cosmo-pss-runtime-gpu:latest
docker push gcr.io/<project-id>/cosmo-pss-runtime-gpu:latest
```

### 4. Configure Lithops

Edit `lithops.yaml` and update with your settings:

```yaml
lithops:
  backend: aws_lambda
  storage: aws_s3

aws:
  region: us-east-1
  lambda:
    runtime: cosmo-pss-runtime
    runtime_memory: 10240
    runtime_timeout: 900
  s3:
    storage_bucket: cosmo-pss-embeddings
    region: us-east-1

env_vars:
  PSS_FP16: "1"
  PSS_VIS_MODEL: "google/siglip-so400m-patch14-384"
  PSS_TEXT_MODEL: "Qwen/Qwen3-Embedding-0.6B"
  PSS_PRECOMP_BATCH: "16"
```

## Usage

### Basic Usage

```python
from cosmo.lithops_precompute import run_lithops_precompute

# Run precomputation
results = run_lithops_precompute(
    books_root='/path/to/books',
    annotations_path='/path/to/comics_train.json',
    output_bucket='cosmo-pss-embeddings',
    backend='aws_lambda',
    workers=100
)

# Check results
for result in results:
    print(f"{result['book_id']}: {result['status']} - {result['pages']} pages")
```

### Command Line

```bash
# AWS Lambda
python -m cosmo.lithops_precompute \
  --books-root /data/books \
  --annotations /data/comics_train.json \
  --output-bucket cosmo-pss-embeddings \
  --backend aws_lambda \
  --workers 100

# Azure Functions
python -m cosmo.lithops_precompute \
  --books-root /data/books \
  --annotations /data/comics_train.json \
  --output-bucket cosmo-pss-embeddings \
  --backend azure_functions \
  --workers 100

# Google Cloud Functions
python -m cosmo.lithops_precompute \
  --books-root /data/books \
  --annotations /data/comics_train.json \
  --output-bucket cosmo-pss-embeddings \
  --backend gcp_functions \
  --workers 100
```

### With Environment Variables

```bash
# Set environment variables
export PSS_FP16=1
export PSS_VIS_MODEL=google/siglip-so400m-patch14-384
export PSS_TEXT_MODEL=Qwen/Qwen3-Embedding-0.6B
export PSS_PRECOMP_BATCH=16

# Run
python -m cosmo.lithops_precompute \
  --books-root s3://my-books-bucket/comics/ \
  --annotations s3://my-data-bucket/comics_train.json \
  --output-bucket cosmo-pss-embeddings \
  --backend aws_lambda \
  --workers 200
```

## Advanced Usage

### GPU-Accelerated Processing (AWS Batch)

For larger models or faster processing, use AWS Batch with GPU instances:

1. Update `lithops.yaml`:
```yaml
lithops:
  backend: aws_batch
  
aws_batch:
  compute_environment: cosmo-pss-gpu-compute
  job_queue: cosmo-pss-gpu-queue
  runtime_memory: 32768
  runtime_timeout: 3600
  vcpus: 4
  instance_type: g4dn.xlarge  # 1x NVIDIA T4 GPU
```

2. Run with AWS Batch:
```python
results = run_lithops_precompute(
    books_root='/data/books',
    annotations_path='/data/comics_train.json',
    output_bucket='cosmo-pss-embeddings',
    backend='aws_batch',
    workers=50  # Fewer workers for GPU instances
)
```

### Hybrid: Serverless + Storage

Read books from S3, process in Lambda, write embeddings back to S3:

```python
results = run_lithops_precompute(
    books_root='s3://comic-books-dataset/books/',
    annotations_path='s3://comic-books-dataset/annotations/comics_train.json',
    output_bucket='cosmo-pss-embeddings',
    backend='aws_lambda',
    workers=500
)
```

### Monitoring Progress

```python
import lithops

executor = lithops.FunctionExecutor(backend='aws_lambda')

# Submit tasks
futures = executor.map(process_book_lithops, book_tasks)

# Monitor progress
while not executor.wait(futures, timeout=10):
    done = sum(1 for f in futures if f.done)
    total = len(futures)
    print(f"Progress: {done}/{total} ({100*done/total:.1f}%)")

# Get results
results = executor.get_result(futures)
```

## Cost Optimization

### Tips for Reducing Costs

1. **Right-size Memory**: Start with 4GB for CPU, increase only if needed
2. **Batch Size**: Larger batches reduce function invocations (use PSS_PRECOMP_BATCH=16-32)
3. **Region Selection**: Choose regions with lower costs (us-east-1 typically cheapest)
4. **Reserved Capacity**: For large-scale repeated runs, consider AWS Savings Plans
5. **Storage Class**: Use S3 Standard for active processing, move to Glacier for archival
6. **Use CPU for Cost**: CPU runtime is 40% cheaper than GPU for this workload

### Detailed Cost Breakdown

#### Small Scale: 1,000 Books (~30,000 pages)

**AWS Lambda CPU (10GB memory, 500 workers):**

| Component | Calculation | Cost |
|-----------|-------------|------|
| Lambda Compute | 10GB × 30,000 pages × 1 sec = 300,000 GB-sec × $0.0000166667 | **$5.00** |
| Lambda Requests | 1,000 books × $0.20/1M | **$0.00** |
| S3 Storage | 15GB × $0.023/GB/month | **$0.35/month** |
| S3 PUT Requests | 60,000 × $0.005/1000 | **$0.30** |
| Data Transfer | 1GB × $0.09/GB | **$0.09** |
| **Total One-Time** | | **$5.74** |
| **Monthly Storage** | | **$0.35** |

#### Medium Scale: 10,000 Books (~300,000 pages)

**AWS Lambda CPU (10GB memory, 500 workers):**

| Component | Calculation | Cost |
|-----------|-------------|------|
| Lambda Compute | 10GB × 300,000 pages × 1 sec = 3M GB-sec × $0.0000166667 | **$50.00** |
| Lambda Requests | 10,000 books × $0.20/1M | **$0.00** |
| S3 Storage | 150GB × $0.023/GB/month | **$3.45/month** |
| S3 PUT Requests | 600,000 × $0.005/1000 | **$3.00** |
| Data Transfer | 10GB × $0.09/GB | **$0.90** |
| **Total One-Time** | | **$53.90** |
| **Monthly Storage** | | **$3.45** |

#### Large Scale: 40,000 Books (~1.2 Million pages) - DETAILED

**AWS Lambda CPU (10GB memory, 500-1000 workers):**

| Component | Detailed Calculation | Cost |
|-----------|----------------------|------|
| **Lambda Compute** | |
| - GB-seconds | 10GB × 1,200,000 pages × 1.0 sec/page = 12,000,000 GB-sec | |
| - Rate | $0.0000166667 per GB-second | |
| - Subtotal | 12M × $0.0000166667 | **$2,000.00** |
| **Lambda Requests** | |
| - Invocations | 40,000 books | |
| - Rate | $0.20 per 1M requests | |
| - Subtotal | 40,000 × ($0.20/1M) | **$0.01** |
| **S3 Storage (First Month)** | |
| - Total size | 600GB (0.5MB per page × 1.2M pages) | |
| - Rate | $0.023 per GB/month | |
| - Subtotal | 600 × $0.023 | **$13.80** |
| **S3 PUT Requests** | |
| - Total requests | 80,000 (2 per book: visual + text) | |
| - Rate | $0.005 per 1,000 PUT requests | |
| - Subtotal | 80,000 × ($0.005/1000) | **$400.00** |
| **S3 GET Requests** (for classification) | |
| - Total requests | 80,000 | |
| - Rate | $0.0004 per 1,000 GET requests | |
| - Subtotal | 80,000 × ($0.0004/1000) | **$0.03** |
| **Data Transfer Out** | |
| - To internet | 30GB | |
| - Rate | $0.09 per GB | |
| - Subtotal | 30 × $0.09 | **$2.70** |
| **CloudWatch Logs** | |
| - Log data | 5GB | |
| - Rate | $0.50 per GB | |
| - Subtotal | 5 × $0.50 | **$2.50** |
| **TOTAL ONE-TIME COST** | | **$2,419.04** |
| **Monthly Storage Cost** | | **$13.80** |

**AWS Batch GPU (g4dn.xlarge, 100 workers):**

| Component | Calculation | Cost |
|-----------|-------------|------|
| EC2 Compute (g4dn.xlarge) | 100 workers × 2 hours × $0.526/hour | **$105.20** |
| Additional Batch overhead | ~10% | **$10.52** |
| S3 Storage | 600GB × $0.023/GB/month | **$13.80/month** |
| S3 PUT Requests | 80,000 × $0.005/1000 | **$400.00** |
| Data Transfer | 30GB × $0.09/GB | **$2.70** |
| **Total One-Time** | | **$518.42** |
| **Monthly Storage** | | **$13.80** |

**Cost Comparison Summary (1.2M pages):**

| Method | Processing Time | One-Time Cost | Best For |
|--------|----------------|---------------|----------|
| **Lambda CPU (Recommended)** | 2-4 hours | **~$2,400** | Cost-sensitive, large-scale |
| **Batch GPU** | 1-2 hours | **~$520** | Performance-critical |
| **Local GPU (4x)** | ~60 hours | **$0** | When cloud not available |

### Cost Optimization Recommendations

1. **For < 100K pages**: Use Lambda CPU, most cost-effective
2. **For 100K-1M pages**: Use Lambda CPU with 500-1000 workers
3. **For > 1M pages**: Consider AWS Batch GPU if time-critical, otherwise Lambda CPU
4. **Storage Lifecycle**: Move embeddings to S3 Glacier Deep Archive ($0.00099/GB/month) after 30 days if not frequently accessed

### Regional Cost Variations (Lambda CPU, 1.2M pages)

| Region | Compute Cost | Total Cost | vs us-east-1 |
|--------|-------------|-----------|--------------|
| us-east-1 | $2,000 | $2,420 | baseline |
| us-west-2 | $2,000 | $2,420 | +0% |
| eu-west-1 | $2,100 | $2,520 | +4% |
| ap-southeast-1 | $2,200 | $2,620 | +8% |

## Troubleshooting

### Common Issues

#### 1. Runtime Memory Exceeded
```
Error: Runtime memory exceeded
```
**Solution**: Increase `runtime_memory` in `lithops.yaml` or reduce `PSS_PRECOMP_BATCH`.

#### 2. Timeout
```
Error: Task timeout after 900s
```
**Solution**: Increase `runtime_timeout` or use AWS Batch for longer tasks.

#### 3. Model Download Fails
```
Error: Failed to download model
```
**Solution**: Pre-cache models in Docker image or increase ephemeral storage.

#### 4. Cold Start Latency
```
Warning: First batch of tasks slow
```
**Solution**: Use Lambda provisioned concurrency or warm up workers.

### Debug Mode

Enable debug logging:
```yaml
lithops:
  log_level: DEBUG
```

Run with verbose output:
```bash
python -m cosmo.lithops_precompute --books-root /data/books --annotations /data/comics_train.json --output-bucket cosmo-pss-embeddings --backend aws_lambda --workers 10 -v
```

## Best Practices

1. **Test Locally First**: Use `localhost` backend for testing
2. **Start Small**: Begin with 10-20 workers, then scale up
3. **Monitor Costs**: Set up billing alerts in your cloud console
4. **Version Control**: Tag Docker images and track runtime versions
5. **Error Handling**: Implement retry logic for transient failures
6. **Data Validation**: Verify embeddings after processing

## Integration with Classification Phase

After precomputation, use embeddings with classification:

```python
from cosmo.classify_precomputed import main as classify

# Download embeddings from S3 to local cache
import boto3
s3 = boto3.client('s3')

bucket = 'cosmo-pss-embeddings'
for book_id in book_ids:
    s3.download_file(bucket, f'visual/{book_id}.pt', f'/cache/visual/{book_id}.pt')
    s3.download_file(bucket, f'text/{book_id}.pt', f'/cache/text/{book_id}.pt')

# Run classification
classify()
```

Or run classification in Lithops too for fully distributed pipeline.

## Comparison with Local Execution

| Feature | Local | Lithops |
|---------|-------|---------|
| Setup Complexity | Low | Medium |
| Scalability | Limited by hardware | Virtually unlimited |
| Cost (small scale) | $0 (owned hardware) | Low ($5-50) |
| Cost (large scale) | High (need infrastructure) | Moderate |
| Maintenance | High | Low |
| Parallelization | Limited | Excellent |
| Fault Tolerance | Manual | Automatic |

## Next Steps

1. **Extend to Classification**: Implement distributed classification with Lithops
2. **Pipeline Orchestration**: Use Step Functions / Durable Functions for multi-stage workflows
3. **Monitoring**: Integrate with CloudWatch / Application Insights
4. **Auto-scaling**: Implement adaptive worker count based on queue depth
5. **Cost Tracking**: Add per-book cost attribution

## Resources

- [Lithops Documentation](https://lithops-cloud.github.io/docs/)
- [AWS Lambda Pricing](https://aws.amazon.com/lambda/pricing/)
- [Azure Functions Pricing](https://azure.microsoft.com/en-us/pricing/details/functions/)
- [GCP Functions Pricing](https://cloud.google.com/functions/pricing)
- [CoSMo PSS Optimizations](./CoSMo_PSS_Optimizations.md)

## Support

For issues with Lithops integration, please file an issue in the repository with:
- Cloud provider and backend used
- Error messages and logs
- Configuration (sanitized)
- Number of books and approximate dataset size
