# First Cosmo Run
- No `textstory` class  
- `first-page` is an artifact of how the model and dataset are set up  

![first cosmo run](image.png)

### Final Model Performance
**Average Book MNDD:** `6.9605`

**Document-level Metrics**
| Metric | Value |
|:--------|:-------:|
| Precision | 0.5057 |
| Recall | 0.4685 |
| F1 | 0.4652 |
| Segmentation Quality (SQ) | 0.9535 |
| Panoptic Quality (PQ) | 0.4426 |

**Class-level Performance**
| Class | Precision | Recall | F1-score | Support |
|:-------|:----------:|:-------:|:----------:|--------:|
| advertisement | 0.48 | 0.91 | 0.63 | 134 |
| cover | 0.66 | 0.69 | 0.67 | 89 |
| story | 0.89 | 0.85 | 0.87 | 1896 |
| textstory | 0.00 | 0.00 | 0.00 | 0 |
| first-page | 0.17 | 0.21 | 0.19 | 204 |
| credits | 0.79 | 0.57 | 0.67 | 101 |
| art | 0.48 | 0.07 | 0.13 | 162 |
| text | 0.03 | 0.75 | 0.07 | 4 |
| back_cover | 0.23 | 0.20 | 0.21 | 76 |

**Overall Summary**
| Metric | Value |
|:--------|:-------:|
| Accuracy | 0.72 |
| Macro Avg (P/R/F1) | 0.42 / 0.47 / 0.38 |
| Weighted Avg (P/R/F1) | 0.76 / 0.72 / 0.73 |
| Total Samples | 2666 |

---

# Second Cosmo Run
![second cosmo run](normCM_BookBERT.png)

Combined **advertisement**, **back cover**, and **art**

### Final Model Performance
**Average Book MNDD:** `4.5526`

**Document-level Metrics**
| Metric | Value |
|:--------|:-------:|
| Precision | 0.6033 |
| Recall | 0.5987 |
| F1 | 0.5767 |
| Segmentation Quality (SQ) | 0.9275 |
| Panoptic Quality (PQ) | 0.5425 |

**Class-level Performance**
| Class | Precision | Recall | F1-score | Support |
|:-------|:----------:|:-------:|:----------:|--------:|
| advertisement | 0.63 | 0.66 | 0.65 | 372 |
| cover | 0.72 | 0.66 | 0.69 | 89 |
| story | 0.95 | 0.91 | 0.93 | 2100 |
| textstory | 0.00 | 0.00 | 0.00 | 0 |
| first-page | 0.00 | 0.00 | 0.00 | 0 |
| credits | 0.70 | 0.64 | 0.67 | 101 |
| art | 0.00 | 0.00 | 0.00 | 0 |
| text | 0.03 | 0.75 | 0.06 | 4 |
| back_cover | 0.00 | 0.00 | 0.00 | 0 |

**Overall Summary**
| Metric | Value |
|:--------|:-------:|
| Accuracy | 0.85 |
| Macro Avg (P/R/F1) | 0.34 / 0.40 / 0.33 |
| Weighted Avg (P/R/F1) | 0.89 / 0.85 / 0.87 |
| Total Samples | 2666 |


Early stopping triggered after 7 epochs.
Epochs:  19%|▏| 77/400 [9:53:48<41:30:56, 462.71s/it, train_loss=0.8634, train_f1=0.5870, val_loss=0.7401, val_f1=0.621
Training finished.