# CODEX 5.3 Loss Function Review

## Scope Reviewed
I reviewed the Stage 3 training/loss implementation in:

- `/home/runner/work/Comic-Analysis/Comic-Analysis/src/version2/train_stage3.py`
- `/home/runner/work/Comic-Analysis/Comic-Analysis/src/version2/stage3_panel_features_framework.py`
- `/home/runner/work/Comic-Analysis/Comic-Analysis/src/version2/stage3_dataset.py`
- `/home/runner/work/Comic-Analysis/Comic-Analysis/src/version2/README_STAGE3.md`
- `/home/runner/work/Comic-Analysis/Comic-Analysis/src/version2/test_stage3_collapse.py`

Note: I did **not** find `train_stage3_model.py` in this repository. The active Stage 3 training file appears to be `train_stage3.py`.

---

## Current Loss Design (as implemented)
The training objective is a weighted sum:

- **Contrastive loss** (`contrastive_loss`) on panel embeddings, with positives defined as panels from the same page.
- **Reconstruction loss** (`reconstruction_loss`) that predicts one masked panel embedding from the mean of the remaining panel embeddings.
- **Modality alignment loss** (`modality_alignment_loss`) using image-to-text cross-entropy retrieval on panels where both image and text are present.

Combined in training as:

`total_loss = 1.0 * contrastive_loss + 0.5 * reconstruction_loss + 0.3 * alignment_loss`

This matches the Stage 3 README description at a high level.

---

## What Looks Good
- The objectives are conceptually aligned with Stage 3 goals (intra-page structure + multimodal alignment).
- Modality masks are used to avoid aligning missing modalities.
- Contrastive implementation follows a supervised-contrastive style with same-page positives and self-pair exclusion.
- Safety checks avoid crashes when too few valid panels exist.

---

## Key Concerns / Risks

### 1) Validation loss is likely noisy because objective module is not set to eval mode
In `validate()`, `model.eval()` is called, but `objectives.eval()` is not. Since `Stage3TrainingObjectives` contains `Dropout` in `reconstruction_head`, validation loss likely includes stochastic dropout behavior.

**Impact:** noisy/unstable validation metrics and potentially unreliable best-checkpoint selection.

**What I would change (in code, if editing were allowed):**
- Set `objectives.eval()` in validation.
- Set `objectives.train()` only during training.

---

### 2) Reconstruction target is not detached (possible representation drift/collapse pressure)
`reconstruction_loss` computes MSE between predicted context-based embedding and target embedding from the same current forward pass, with gradients flowing through both sides.

**Impact:** the model can reduce loss by moving both context and target embeddings together rather than learning stronger predictive structure, which can encourage over-smoothing.

**What I would change:**
- Detach the target embedding in reconstruction loss (predict a stable target).
- Optionally detach context too in an ablation to test stability vs learning signal.

---

### 3) Alignment loss is one-directional only (vision -> text)
Current alignment is `CrossEntropy(logits=image·text^T, labels=diag)`, i.e., image retrieval of text only.

**Impact:** weaker bidirectional alignment; text->image retrieval quality can lag.

**What I would change:**
- Use symmetric contrastive alignment: average of vision->text and text->vision CE losses.

---

### 4) Loss weighting is fixed and not externally configurable
Weights are hardcoded in both `train_epoch()` and `validate()`.

**Impact:** difficult tuning and reproducibility across datasets.

**What I would change:**
- Expose `contrastive_weight`, `reconstruction_weight`, and `alignment_weight` via args/config.

---

### 5) Potential training signal sparsity on pages with very few valid panels
Multiple branches return zero loss when there are <2 valid entries.

**Impact:** effective batch objective can become weak depending on dataset panel distribution and text sparsity.

**What I would change:**
- Log per-component “active sample counts”.
- Consider sampling strategy to ensure enough multi-panel and text-present samples per batch.

---

### 6) Temperature and objective hyperparameters are not fully wired through CLI
`Stage3TrainingObjectives` accepts `temperature`, but the script instantiation does not expose CLI/config control for it.

**Impact:** reduced experiment control and reproducibility.

**What I would change:**
- Add CLI/config plumbed temperature for all contrastive/alignment logits.

---

## Secondary Observations (Not strict bugs, but worth attention)
- Contrastive complexity scales quadratically with number of valid panels in batch (`O(V^2)` similarity matrix).
- Gradient clipping is applied only to `model.parameters()` while optimizer includes `objectives.parameters()` too.
- `test_stage3_collapse.py` is useful and should remain part of regular regression checks after loss changes.

---

## Suggested Prioritized Fix Order
1. **Validation mode fix** (`objectives.eval()` in validation).  
2. **Reconstruction target detachment** to stabilize representation learning.  
3. **Symmetric alignment loss** (vision<->text).  
4. **Configurable loss weights + temperature plumbing**.  
5. **Add per-loss diagnostics** (active pairs/samples, each component magnitude).

---

## Example “would-change” pseudocode (illustrative only)

```text
# validation
model.eval()
objectives.eval()

# reconstruction target stabilization
target = valid_panels[mask_idx].detach()

# symmetric alignment
loss_image_to_text = CE(sim(vision, text), diag_labels)
loss_text_to_image = CE(sim(vision, text).T, diag_labels)
alignment = 0.5 * (loss_image_to_text + loss_text_to_image)

# weighted total from config
total_loss = contrastive_weight * contrastive + reconstruction_weight * reconstruction + alignment_weight * alignment
```

---

## Bottom Line
The current Stage 3 loss framework is directionally strong, but there are a few high-impact issues in objective handling and optimization details (especially validation mode and reconstruction target gradients) that could materially affect stability and model quality. Addressing those first should improve confidence in training dynamics without changing the broader architecture.
