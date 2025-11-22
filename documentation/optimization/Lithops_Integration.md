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

| Method | Books | Time | Cost* | Notes |
|--------|-------|------|-------|-------|
| Local Sequential | 1000 | ~28 hours | $0 | Single GPU machine |
| Local Parallel (4 GPUs) | 1000 | ~7 hours | $0 | Multi-GPU machine |
| Lithops (100 workers) | 1000 | ~17 minutes | ~$50 | AWS Lambda + S3 |
| Lithops (500 workers) | 1000 | ~3 minutes | ~$50 | AWS Lambda + S3 |

*Costs are estimates and vary by cloud provider and region.

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

The CoSMo PSS pipeline requires a custom runtime with ML dependencies.

```bash
# Build Docker image
docker build -f Dockerfile.lithops -t cosmo-pss-runtime .

# Push to registry (example for AWS ECR)
aws ecr create-repository --repository-name cosmo-pss-runtime
docker tag cosmo-pss-runtime:latest <account>.dkr.ecr.<region>.amazonaws.com/cosmo-pss-runtime:latest
docker push <account>.dkr.ecr.<region>.amazonaws.com/cosmo-pss-runtime:latest

# Register runtime with Lithops
lithops runtime create cosmo-pss-runtime --backend aws_lambda
```

For Azure Container Registry:
```bash
az acr create --resource-group myResourceGroup --name cosmopssregistry --sku Basic
az acr login --name cosmopssregistry
docker tag cosmo-pss-runtime:latest cosmopssregistry.azurecr.io/cosmo-pss-runtime:latest
docker push cosmopssregistry.azurecr.io/cosmo-pss-runtime:latest
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

1. **Right-size Memory**: Start with 4GB, increase only if needed
2. **Batch Size**: Larger batches reduce function invocations
3. **Region Selection**: Choose regions with lower costs
4. **Reserved Capacity**: For large-scale repeated runs
5. **Storage Class**: Use appropriate S3 storage class for embeddings

### Cost Breakdown Example (AWS Lambda)

For processing 1000 books with ~100 pages each:

| Component | Usage | Cost |
|-----------|-------|------|
| Lambda Compute | 100,000 GB-seconds | ~$1.67 |
| Lambda Requests | 1,000 invocations | ~$0.00 |
| S3 Storage | 50 GB | ~$1.15/month |
| S3 PUT Requests | 200,000 | ~$1.00 |
| Data Transfer | 5 GB | ~$0.45 |
| **Total** | | **~$4.27 + storage** |

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
