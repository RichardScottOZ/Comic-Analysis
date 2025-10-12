"""
Train CLOSURE-Lite Simple Framework
This version skips problematic sequence processing to preserve panel diversity
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import time
from pathlib import Path
import wandb
from tqdm import tqdm

# Import our modules
from closure_lite_simple_framework import ClosureLiteSimple
from closure_lite_dataset import ComicsPageDataset, collate_pages

def convert_windows_to_wsl_path(path: str) -> str:
    """Convert Windows path to WSL path"""
    if path.startswith('E:/') or path.startswith('E:\\'):
        normalized_path = path[2:].replace('\\', '/')
        return f"/mnt/e{normalized_path}"
    elif path.startswith('C:/') or path.startswith('C:\\'):
        normalized_path = path[2:].replace('\\', '/')
        return f"/mnt/c{normalized_path}"
    elif path.startswith('D:/') or path.startswith('D:\\'):
        normalized_path = path[2:].replace('\\', '/')
        return f"/mnt/d{normalized_path}"
    return path

def load_json_list(json_list_file: str) -> list:
    """Load list of JSON file paths from text file"""
    print(f"Loading JSON list from {json_list_file}")
    
    # Convert json_list_file to WSL path if needed
    json_list_file = convert_windows_to_wsl_path(json_list_file)
    # Normalize separators for WSL/Unix environments (relative paths with backslashes)
    json_list_file = json_list_file.replace('\\', '/')
    # Resolve relative path against repo root if it doesn't exist from CWD
    resolved_list_file = json_list_file
    if not os.path.isabs(resolved_list_file) and not os.path.exists(resolved_list_file):
        try:
            # repo root is three levels up from this script: openrouter/detections/benchmarks -> repo
            repo_root = Path(__file__).resolve().parents[3]
            candidate = (repo_root / resolved_list_file).as_posix()
            if os.path.exists(candidate):
                resolved_list_file = candidate
        except Exception:
            pass
    print(f"Using JSON list file: {resolved_list_file}")
    
    # If a directory is provided, collect all JSONs inside it
    if os.path.isdir(resolved_list_file):
        from pathlib import Path as _Path
        jsons = sorted(str(p) for p in _Path(resolved_list_file).glob('*.json'))
        print(f"Found {len(jsons)} JSON files under directory {resolved_list_file}")
        if not jsons:
            raise ValueError(f"No JSON files found in directory: {resolved_list_file}")
        return jsons

    json_paths = []
    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    
    for encoding in encodings:
        try:
            with open(resolved_list_file, 'r', encoding=encoding) as f:
                for line in f:
                    path = line.strip()
                    if path:
                        # Convert Windows paths to WSL paths if needed
                        wsl_path = convert_windows_to_wsl_path(path)
                        if os.path.exists(wsl_path):
                            json_paths.append(wsl_path)
                        else:
                            # Try original path too
                            if os.path.exists(path):
                                json_paths.append(path)
            print(f"Successfully loaded {len(json_paths)} JSON paths using {encoding} encoding")
            break
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"Error with {encoding} encoding: {e}")
            continue
    
    if not json_paths:
        raise ValueError(f"Could not load JSON list from {resolved_list_file}")
    
    return json_paths

def _load_precomputed_map(path: str | None) -> dict:
    if not path:
        return {}
    try:
        # Allow JSON or CSV with columns json_path,image_path
        p = convert_windows_to_wsl_path(path)
        p = p.replace('\\', '/')
        if not os.path.exists(p):
            return {}
        if p.lower().endswith('.json'):
            import json as _json
            with open(p, 'r', encoding='utf-8') as f:
                m = _json.load(f)
                if isinstance(m, dict):
                    return m
                elif isinstance(m, list):
                    # list of {json_path:..., image_path:...}
                    out = {}
                    for row in m:
                        if isinstance(row, dict) and 'json_path' in row and 'image_path' in row:
                            out[row['json_path']] = row['image_path']
                    return out
        else:
            # CSV fallback
            import csv
            out = {}
            with open(p, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for r in reader:
                    jp = r.get('json_path') or r.get('json') or r.get('jsonfile')
                    ip = r.get('image_path') or r.get('image') or r.get('img')
                    if jp and ip:
                        out[jp] = ip
            return out
    except Exception as e:
        print(f"Warning: failed to load precomputed map from {path}: {e}")
        return {}

def create_dataloader_from_list(json_list_file: str, image_root: str, batch_size: int = 4, 
                               num_workers: int = 4, max_samples: int = None, precomputed_map_file: str | None = None) -> DataLoader:
    """Create dataloader from a list of JSON file paths"""
    
    # Load the list of JSON paths
    json_paths = load_json_list(json_list_file)
    
    if max_samples:
        json_paths = json_paths[:max_samples]
        print(f"Limited to {max_samples} samples")
    
    # Pre-filter JSONs that are missing an image path to avoid worker KeyError
    def _prefilter_missing_image_path(paths):
        bad = []  # truly unusable (load errors, unexpected structure)
        good = []
        inferred_ok = []  # JSONs without an image key that we'll allow (dataset will infer)
        for p in paths:
            try:
                with open(p, 'r', encoding='utf-8') as f:
                    data = f.read()
                # lazy small check first to avoid json load if clearly missing all accepted keys
                fast_keys = ('page_image_path', 'image_path', 'image', 'IMAGE_PATH')
                fast_has_any = any(k in data for k in fast_keys)
                import json as _json
                j = _json.loads(data)
                if isinstance(j, dict):
                    page = j
                elif isinstance(j, list) and len(j) > 0 and isinstance(j[0], dict):
                    page = j[0]
                else:
                    bad.append((p, 'unexpected_json_structure'))
                    continue
                v = page.get('page_image_path') or page.get('image_path') or page.get('image') or page.get('IMAGE_PATH')
                if isinstance(v, str) and len(v.strip()) > 0:
                    good.append(p)
                else:
                    # If no explicit image key, keep for dataset-level inference
                    inferred_ok.append(p)
            except Exception as e:
                bad.append((p, f'load_error:{repr(e)}'))
        # Logging
        if bad:
            log_path_bad = f"{json_list_file}.bad_unusable_jsons.txt"
            try:
                with open(log_path_bad, 'w', encoding='utf-8') as lf:
                    for p, reason in bad:
                        lf.write(f"{p}\t{reason}\n")
                print(f"Excluded {len(bad)} unusable JSONs; details: {log_path_bad}")
            except Exception as e:
                print(f"Warning: unable to write bad-sample log: {e}")
        if inferred_ok:
            log_path_info = f"{json_list_file}.info_no_image_key_inferred.txt"
            try:
                with open(log_path_info, 'w', encoding='utf-8') as lf:
                    for p in inferred_ok:
                        lf.write(f"{p}\tno_image_key_inferred\n")
                print(f"Allowing {len(inferred_ok)} JSONs without explicit image path (will infer at load). Details: {log_path_info}")
            except Exception as e:
                print(f"Warning: unable to write info log: {e}")
        kept = good + inferred_ok
        if kept:
            print(f"Prefilter kept {len(kept)} JSONs out of {len(paths)}")
        else:
            print("Prefilter found no usable JSONs. Check your list file.")
        return kept

    json_paths = _prefilter_missing_image_path(json_paths)
    if not json_paths:
        log_path_bad = f"{json_list_file}.bad_unusable_jsons.txt"
        raise ValueError(
            "Prefilter found no usable JSONs. "
            f"Check your list file and see details in: {log_path_bad}"
        )
    
    # Convert image_root to WSL path if needed
    image_root = convert_windows_to_wsl_path(image_root)
    
    print(f"Creating dataset with {len(json_paths)} JSON files")
    print(f"Image root: {image_root}")
    pcm = _load_precomputed_map(precomputed_map_file)
    if pcm:
        print(f"Loaded precomputed json->image map with {len(pcm)} entries: {precomputed_map_file}")
    
    # Create dataset
    dataset = ComicsPageDataset(json_paths, image_root, precomputed_map=pcm)
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_pages,
        pin_memory=True
    )
    
    print(f"Created dataloader with {len(dataset)} samples")
    return dataloader

def train_epoch(model, dataloader, optimizer, scheduler, scaler, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    total_mpm = 0.0
    total_pop = 0.0
    total_rpp = 0.0
    processed_batches = 0
    
    # Running averages for progress bar
    running_loss = 0.0
    running_mpm = 0.0
    running_pop = 0.0
    running_rpp = 0.0
    
    data_iter = iter(dataloader)
    batch_idx = 0
    pbar = tqdm(total=len(dataloader), desc=f"Epoch {epoch}")
    while True:
        try:
            batch = next(data_iter)
        except StopIteration:
            break
        except Exception as e:
            # This catches DataLoader worker errors (e.g., KeyError in dataset)
            print(f"Data loading error at batch {batch_idx}: {repr(e)}")
            batch_idx += 1
            pbar.update(1)
            continue
        try:
            # Move to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass with mixed precision
            with torch.cuda.amp.autocast():
                loss, components = model(batch)
                
                # Check for NaN
                if torch.isnan(loss):
                    print(f"NaN loss detected at batch {batch_idx}! Skipping this batch.")
                    batch_idx += 1
                    pbar.update(1)
                    continue
            
            # Backward pass
            scaler.scale(loss).backward()
            
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            # Update running averages
            running_loss = 0.9 * running_loss + 0.1 * loss.item()
            running_mpm = 0.9 * running_mpm + 0.1 * components['L_mpm']
            running_pop = 0.9 * running_pop + 0.1 * components['L_pop']
            running_rpp = 0.9 * running_rpp + 0.1 * components['L_rpp']
            
            # Accumulate totals
            total_loss += loss.item()
            total_mpm += components['L_mpm']
            total_pop += components['L_pop']
            total_rpp += components['L_rpp']
            processed_batches += 1
            
        except Exception as e:
            print(f"Error at batch {batch_idx}: {e}")
        finally:
            # Update progress bar regardless
            pbar.set_postfix({
                'Loss': f"{running_loss:.4f}",
                'MPM': f"{running_mpm:.4f}",
                'POP': f"{running_pop:.4f}",
                'RPP': f"{running_rpp:.4f}"
            })
            # Save intermediate checkpoint every 10K batches
            if batch_idx > 0 and batch_idx % 10000 == 0 and processed_batches > 0:
                checkpoint = {
                    'epoch': epoch,
                    'batch_idx': batch_idx,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'loss': running_loss
                }
                torch.save(checkpoint, f'./closure_lite_simple_output/intermediate_checkpoint_epoch_{epoch}_batch_{batch_idx}.pth')
                print(f"Saved intermediate checkpoint at batch {batch_idx}")
            batch_idx += 1
            pbar.update(1)
    pbar.close()
    
    # Calculate averages
    num_batches = max(processed_batches, 1)
    avg_loss = total_loss / num_batches
    avg_mpm = total_mpm / num_batches
    avg_pop = total_pop / num_batches
    avg_rpp = total_rpp / num_batches
    
    return avg_loss, (avg_mpm, avg_pop, avg_rpp)

def main():
    parser = argparse.ArgumentParser(description='Train CLOSURE-Lite Simple Framework')
    parser.add_argument('--json_list_file', required=True, 
                       help='Text file containing list of JSON file paths')
    parser.add_argument('--image_root', required=True, 
                       help='Root directory for comic images')
    parser.add_argument('--output_dir', default='./closure_lite_simple_output', 
                       help='Output directory for checkpoints')
    parser.add_argument('--batch_size', type=int, default=4, 
                       help='Training batch size')
    parser.add_argument('--epochs', type=int, default=3, 
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, 
                       help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=4, 
                       help='Number of data loading workers')
    parser.add_argument('--max_samples', type=int, default=None, 
                       help='Maximum number of samples to use')
    parser.add_argument('--precomputed_map', type=str, default=None,
                       help='Optional JSON or CSV with json_path,image_path to skip expensive inference')
    parser.add_argument('--wandb_project', default='closure-lite-simple', 
                       help='Wandb project name')
    parser.add_argument('--resume', type=str, default=None, 
                       help='Path to checkpoint to resume from')
    parser.add_argument('--num_heads', type=int, default=4,
                       help='Number of attention heads')
    parser.add_argument('--temperature', type=float, default=0.1,
                       help='Attention temperature (lower for softer attention)')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize wandb
    wandb.init(
        project=args.wandb_project,
        config={
            'batch_size': args.batch_size,
            'epochs': args.epochs,
            'lr': args.lr,
            'max_samples': args.max_samples,
            'json_list_file': args.json_list_file,
            'num_heads': args.num_heads,
            'temperature': args.temperature,
            'simple_framework': True,
            'no_sequence_processing': True
        }
    )
    
    # Create dataloader
    print("Creating dataloader...")
    dataloader = create_dataloader_from_list(
        args.json_list_file,
        args.image_root,
        args.batch_size,
        args.num_workers,
        args.max_samples,
        args.precomputed_map
    )
    
    # Create model with simple framework
    print("Creating model with simple framework (no sequence processing)...")
    model = ClosureLiteSimple(
        d=384, 
        num_heads=args.num_heads, 
        temperature=args.temperature
    ).to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Trainable percentage: {100 * trainable_params / total_params:.1f}%")
    
    # Create optimizer with different learning rates for different components
    encoder_params = []
    other_params = []
    
    for name, param in model.named_parameters():
        if 'atom.vision.vit' in name or 'atom.text.lm' in name:
            encoder_params.append(param)
        else:
            other_params.append(param)
    
    optimizer = torch.optim.AdamW([
        {'params': encoder_params, 'lr': args.lr * 0.1},  # 10x lower LR for encoders
        {'params': other_params, 'lr': args.lr}
    ], weight_decay=0.01)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )
    
    # Mixed precision training
    scaler = torch.cuda.amp.GradScaler()
    
    # Resume from checkpoint if specified
    start_epoch = 1
    best_loss = float('inf')
    
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint.get('loss', float('inf'))
        print(f"Resumed from epoch {start_epoch-1}, best loss: {best_loss:.4f}")
    
    print("Starting training with simple framework...")
    
    # Training loop
    for epoch in range(start_epoch, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        
        # Train for one epoch
        avg_loss, (avg_mpm, avg_pop, avg_rpp) = train_epoch(
            model, dataloader, optimizer, scheduler, scaler, device, epoch
        )
        
        print(f"Epoch {epoch} - Avg Loss: {avg_loss:.4f} MPM: {avg_mpm:.4f} POP: {avg_pop:.4f} RPP: {avg_rpp:.4f}")
        
        # Log to wandb
        wandb.log({
            'epoch': epoch,
            'loss': avg_loss,
            'L_mpm': avg_mpm,
            'L_pop': avg_pop,
            'L_rpp': avg_rpp,
            'lr': scheduler.get_last_lr()[0]
        })
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'loss': avg_loss
        }
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(checkpoint, os.path.join(args.output_dir, 'best_checkpoint.pth'))
            print(f"New best model saved! Loss: {avg_loss:.4f}")
        
        # Save latest checkpoint
        torch.save(checkpoint, os.path.join(args.output_dir, 'latest_checkpoint.pth'))
        
        # Save epoch checkpoint
        torch.save(checkpoint, os.path.join(args.output_dir, f'epoch_{epoch}_checkpoint.pth'))
    
    print("Training complete!")
    wandb.finish()

if __name__ == "__main__":
    main()
