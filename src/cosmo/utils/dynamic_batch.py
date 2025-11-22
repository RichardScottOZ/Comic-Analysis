import torch

def try_batch(backbone_forward, batch_imgs, start_bs):
    bs = start_bs
    while bs > 0:
        try:
            return backbone_forward(batch_imgs[:bs]), bs
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                torch.cuda.empty_cache()
                bs = bs // 2
            else:
                raise
    raise RuntimeError("Could not fit any samples into GPU memory.")
