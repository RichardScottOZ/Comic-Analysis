"""
Lithops-compatible precompute embeddings for CoSMo PSS pipeline.

This module provides a distributed, serverless implementation of embedding
precomputation using the Lithops framework for cloud-native parallel processing.
"""
import os
import json
import torch
from pathlib import Path
from PIL import Image
from transformers import AutoModel, AutoProcessor, SiglipImageProcessor
from sentence_transformers import SentenceTransformer


def initialize_models():
    """Initialize models once per worker (called by Lithops)."""
    MODEL_ID = os.environ.get("PSS_VIS_MODEL", "google/siglip-so400m-patch14-384")
    TEXT_MODEL_ID = os.environ.get("PSS_TEXT_MODEL", "Qwen/Qwen3-Embedding-0.6B")
    USE_FP16 = os.environ.get("PSS_FP16", "1") == "1"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Initializing models on {DEVICE}")
    backbone = AutoModel.from_pretrained(MODEL_ID).eval().to(DEVICE)
    processor = (SiglipImageProcessor.from_pretrained(MODEL_ID)
                 if "siglip" in MODEL_ID else AutoProcessor.from_pretrained(MODEL_ID))
    text_model = SentenceTransformer(TEXT_MODEL_ID, device=str(DEVICE))
    
    return {
        'backbone': backbone,
        'processor': processor,
        'text_model': text_model,
        'device': DEVICE,
        'use_fp16': USE_FP16
    }


def load_ocr(ocr_path: str) -> str:
    """Load OCR data from JSON file."""
    try:
        with open(ocr_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return json.dumps(data.get("OCRResult", {}))
    except Exception:
        return ""


def process_book_lithops(book_data, storage):
    """
    Process a single book and save embeddings to cloud storage.
    
    This function is designed to be executed by Lithops workers.
    
    Args:
        book_data: Dict containing book metadata and paths
        storage: Lithops storage backend instance
    
    Returns:
        Dict with processing status and metadata
    """
    # Extract book info
    book_id = book_data['hash_code']
    books_root = Path(book_data['books_root'])
    book_dir = books_root / book_id
    
    # Initialize models (cached per worker)
    models = initialize_models()
    backbone = models['backbone']
    processor = models['processor']
    text_model = models['text_model']
    device = models['device']
    use_fp16 = models['use_fp16']
    
    # Check if book directory exists
    if not book_dir.is_dir():
        return {'book_id': book_id, 'status': 'missing_directory', 'pages': 0}
    
    # Get image files
    img_files = sorted([p for p in book_dir.iterdir() 
                       if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
    ocr_files = [p.with_suffix(".json") for p in img_files]
    
    if not img_files:
        return {'book_id': book_id, 'status': 'no_images', 'pages': 0}
    
    # Process in batches
    visual_batches = []
    text_batches = []
    batch_size = int(os.environ.get("PSS_PRECOMP_BATCH", "32"))
    
    for i in range(0, len(img_files), batch_size):
        batch_imgs = img_files[i:i+batch_size]
        
        # Visual features
        images = [Image.open(p).convert("RGB") for p in batch_imgs]
        inputs = processor(images=images, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.inference_mode(), torch.cuda.amp.autocast(
            enabled=use_fp16, 
            dtype=torch.float16 if use_fp16 else torch.float32
        ):
            out = backbone(**inputs)
            feats = (out.pooler_output if hasattr(out, "pooler_output") 
                    else out.last_hidden_state[:, 0, :])
        visual_batches.append(feats.detach().cpu())
        
        # Text features
        texts = [load_ocr(str(p)) for p in ocr_files[i:i+batch_size]]
        embeddings = text_model.encode(
            texts, 
            batch_size=len(texts), 
            convert_to_numpy=True, 
            show_progress_bar=False
        )
        text_batches.append(torch.from_numpy(embeddings))
    
    # Concatenate batches
    visual_tensor = torch.cat(visual_batches, dim=0)
    text_tensor = torch.cat(text_batches, dim=0)
    
    # Save to storage backend (S3, Azure Blob, etc.)
    visual_key = f"visual/{book_id}.pt"
    text_key = f"text/{book_id}.pt"
    
    # Serialize tensors
    import io
    visual_buffer = io.BytesIO()
    text_buffer = io.BytesIO()
    torch.save(visual_tensor, visual_buffer)
    torch.save(text_tensor, text_buffer)
    
    # Upload to cloud storage
    storage.put_object(
        bucket=book_data['output_bucket'],
        key=visual_key,
        body=visual_buffer.getvalue()
    )
    storage.put_object(
        bucket=book_data['output_bucket'],
        key=text_key,
        body=text_buffer.getvalue()
    )
    
    return {
        'book_id': book_id,
        'status': 'success',
        'pages': len(img_files),
        'visual_shape': list(visual_tensor.shape),
        'text_shape': list(text_tensor.shape),
        'visual_key': visual_key,
        'text_key': text_key
    }


def run_lithops_precompute(books_root: str, annotations_path: str, 
                           output_bucket: str, backend: str = 'aws_lambda',
                           workers: int = 100):
    """
    Run precomputation using Lithops for parallel processing.
    
    Args:
        books_root: Root directory containing book subdirectories
        annotations_path: Path to comics_train.json
        output_bucket: S3/Azure/GCS bucket name for output
        backend: Lithops backend ('aws_lambda', 'aws_batch', 'azure_functions', etc.)
        workers: Maximum number of parallel workers
    
    Example:
        >>> run_lithops_precompute(
        ...     books_root='/data/books',
        ...     annotations_path='/data/annotations/v1/comics_train.json',
        ...     output_bucket='cosmo-embeddings',
        ...     backend='aws_lambda',
        ...     workers=100
        ... )
    """
    import lithops
    
    # Load annotations
    with open(annotations_path, 'r', encoding='utf-8') as f:
        books = json.load(f)
    
    print(f"Processing {len(books)} books using Lithops with {backend} backend")
    
    # Prepare book data for workers
    book_tasks = [
        {
            'hash_code': book['hash_code'],
            'books_root': books_root,
            'output_bucket': output_bucket
        }
        for book in books
    ]
    
    # Initialize Lithops executor
    executor = lithops.FunctionExecutor(backend=backend)
    
    # Map process_book_lithops across all books
    futures = executor.map(process_book_lithops, book_tasks)
    
    # Wait for all tasks to complete
    results = executor.get_result(futures)
    
    # Print summary
    successful = sum(1 for r in results if r['status'] == 'success')
    total_pages = sum(r.get('pages', 0) for r in results)
    
    print(f"\nProcessing complete:")
    print(f"  Successful: {successful}/{len(books)} books")
    print(f"  Total pages: {total_pages}")
    print(f"  Output bucket: {output_bucket}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Run CoSMo PSS embedding precomputation with Lithops'
    )
    parser.add_argument('--books-root', required=True,
                       help='Root directory containing book subdirectories')
    parser.add_argument('--annotations', required=True,
                       help='Path to annotations JSON file')
    parser.add_argument('--output-bucket', required=True,
                       help='S3/Azure/GCS bucket name for output embeddings')
    parser.add_argument('--backend', default='aws_lambda',
                       choices=['aws_lambda', 'aws_batch', 'azure_functions', 
                               'gcp_functions', 'ibm_cf', 'code_engine'],
                       help='Lithops backend to use')
    parser.add_argument('--workers', type=int, default=100,
                       help='Maximum number of parallel workers')
    
    args = parser.parse_args()
    
    run_lithops_precompute(
        books_root=args.books_root,
        annotations_path=args.annotations,
        output_bucket=args.output_bucket,
        backend=args.backend,
        workers=args.workers
    )
