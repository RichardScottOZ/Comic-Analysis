# AWS Lambda Infrastructure Cost Analysis for 1.22M Pages
**Date:** 2025-12-31
**Context:** Batched VLM Analysis with Memory Escalation

This document analyzes the *infrastructure* cost (AWS Lambda charges) for processing the 1.22 million page dataset, separate from the Model API costs.

## 1. Lambda Pricing Model (us-east-1)
*   **Architecture:** x86_64
*   **Price per GB-second:** $0.0000166667
*   **Request Cost:** $0.20 per 1M requests (negligible)

### Cost Per 100ms Unit:
*   **128 MB:** $0.0000002083
*   **192 MB:** $0.0000003125
*   **256 MB:** $0.0000004167
*   **512 MB:** $0.0000008333
*   **1024 MB:** $0.0000016667

## 2. Execution Profile Assumptions
Based on benchmarks:
*   **Avg Duration:** 4 seconds per page (1.7s model inference + 2.3s overhead/network).
*   **Escalation Rate:**
    *   **Tier 1 (128MB):** 90% success rate (cheap).
    *   **Tier 2 (192MB):** 7% success rate.
    *   **Tier 3 (256MB+):** 3% success rate.

## 3. Cost Calculation (1.22 Million Pages)

### Step 1: 90% Processed at 128MB
*   **Count:** 1,098,000 pages
*   **Duration:** 4 seconds
*   **Cost/Page:** 4s × (128/1024 GB) × $0.0000166667 = **$0.0000083**
*   **Subtotal:** $9.11

### Step 2: 7% Retry at 192MB
*   **Count:** 85,400 pages
*   **Duration:** 4 seconds
*   **Cost/Page:** 4s × (192/1024 GB) × $0.0000166667 = **$0.0000125**
*   **Subtotal:** $1.07

### Step 3: 3% Retry at 512MB (Conservative Estimate)
*   **Count:** 36,600 pages
*   **Duration:** 4 seconds
*   **Cost/Page:** 4s × (512/1024 GB) × $0.0000166667 = **$0.0000333**
*   **Subtotal:** $1.22

### Step 4: Request Charges
*   **Total Invokes:** ~1.35 Million (including retries)
*   **Cost:** 1.35 × $0.20 = **$0.27**

## 4. Total Projected Infrastructure Cost

| Component | Cost |
|:---|---:|
| Compute (Tier 1 - 128MB) | $9.11 |
| Compute (Tier 2 - 192MB) | $1.07 |
| Compute (Tier 3 - 512MB) | $1.22 |
| Requests | $0.27 |
| **Compute Subtotal** | **$11.67** |

## 4. S3 Storage & Request Costs (us-east-1)
*   **Data Transfer:** $0.00 (Same region)
*   **GET Requests (Reading 1.22M Images):** 1.22M × $0.0004/1k = **$0.49**
*   **PUT Requests (Writing 1.22M JSONs):** 1.22M × $0.005/1k = **$6.10**
*   **LIST Requests (Skip Check logic):** ~1.3k requests = **$0.01**

## 5. Grand Total Infrastructure Projection

| Component | Cost |
|:---|---:|
| Lambda Compute | $11.67 |
| S3 Requests | $6.60 |
| **GRAND TOTAL** | **$18.27** |

## Conclusion
The infrastructure cost to process the entire 1.22M dataset on AWS Lambda is approximately **$18.27**. 
- S3 PUT requests are the second largest expense after compute.
- Total infra is **~4.5%** of the cheapest model API cost (~$382 for Nova Lite).

