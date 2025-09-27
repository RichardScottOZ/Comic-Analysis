#!/usr/bin/env python3
"""
Quick test for the fixed comic search script. - generic idea for search
Tests the directory mapping and caption loading.
"""

from pathlib import Path
import json

def test_directory_mapping():
    """Test that we can map images to captions correctly."""
    print("Testing directory mapping...")
    
    IMAGES_DIR = Path("E:/CalibreComics_extracted")
    ANALYSIS_DIR = Path("E:/CalibreComics_analysis")
    
    if not IMAGES_DIR.exists():
        print(f"❌ Images directory not found: {IMAGES_DIR}")
        return False
    
    if not ANALYSIS_DIR.exists():
        print(f"❌ Analysis directory not found: {ANALYSIS_DIR}")
        return False
    
    # Find a few comic folders
    comic_folders = []
    for folder in IMAGES_DIR.iterdir():
        if folder.is_dir():
            image_files = list(folder.glob("*.jpg")) + list(folder.glob("*.png"))
            if len(image_files) >= 3:
                comic_folders.append(folder)
                if len(comic_folders) >= 2:
                    break
    
    if not comic_folders:
        print("❌ No comic folders found with images")
        return False
    
    print(f"✅ Found {len(comic_folders)} comic folders")
    
    # Test mapping for each folder
    total_matches = 0
    total_images = 0
    
    for folder in comic_folders:
        comic_name = folder.name
        print(f"\nTesting folder: {comic_name}")
        
        # Check if analysis folder exists
        analysis_folder = ANALYSIS_DIR / comic_name
        if not analysis_folder.exists():
            print(f"  ⚠️  No analysis folder: {analysis_folder}")
            continue
        
        # Test image-to-caption mapping
        image_files = list(folder.glob("*.jpg")) + list(folder.glob("*.png"))
        matches = 0
        
        for img_file in image_files[:5]:  # Test first 5 images
            img_filename = img_file.stem
            json_path = analysis_folder / f"{img_filename}.json"
            
            if json_path.exists():
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # Test caption extraction
                    caption = ""
                    if isinstance(data, dict):
                        if "overall_summary" in data:
                            caption = data["overall_summary"]
                        elif "caption" in data:
                            caption = data["caption"]
                        elif "summary" in data and isinstance(data["summary"], dict) and "plot" in data["summary"]:
                            caption = data["summary"]["plot"]
                    
                    if caption.strip():
                        matches += 1
                        print(f"  ✅ {img_filename}.jpg → {img_filename}.json (caption: {len(caption)} chars)")
                    else:
                        print(f"  ⚠️  {img_filename}.jpg → {img_filename}.json (no caption found)")
                        
                except Exception as e:
                    print(f"  ❌ {img_filename}.jpg → {img_filename}.json (error: {e})")
            else:
                print(f"  ❌ {img_filename}.jpg → {img_filename}.json (not found)")
        
        total_images += len(image_files[:5])
        total_matches += matches
        print(f"  📊 {matches}/{len(image_files[:5])} images have captions")
    
    print(f"\n📊 Overall: {total_matches}/{total_images} images have captions")
    
    if total_matches > 0:
        print("✅ Directory mapping test passed!")
        return True
    else:
        print("❌ No image-caption matches found")
        return False

def test_clip_availability():
    """Test if CLIP is available."""
    print("\nTesting CLIP availability...")
    
    try:
        import clip
        import torch
        print("✅ CLIP imported successfully")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"✅ Using device: {device}")
        
        model, preprocess = clip.load("ViT-B/32", device=device)
        print("✅ CLIP model loaded successfully")
        
        return True
    except ImportError as e:
        print(f"❌ CLIP import error: {e}")
        return False
    except Exception as e:
        print(f"❌ CLIP loading error: {e}")
        return False

def test_faiss_availability():
    """Test if FAISS is available."""
    print("\nTesting FAISS availability...")
    
    try:
        import faiss
        import numpy as np
        print("✅ FAISS imported successfully")
        
        # Quick test
        dimension = 512
        index = faiss.IndexFlatIP(dimension)
        test_vectors = np.random.random((5, dimension)).astype('float32')
        index.add(test_vectors)
        print(f"✅ FAISS test passed (index size: {index.ntotal})")
        
        return True
    except ImportError as e:
        print(f"❌ FAISS import error: {e}")
        return False
    except Exception as e:
        print(f"❌ FAISS test error: {e}")
        return False

def main():
    """Run all tests."""
    print("=== Fixed Comic Search Test ===")
    
    tests = [
        ("Directory Mapping", test_directory_mapping),
        ("CLIP Availability", test_clip_availability),
        ("FAISS Availability", test_faiss_availability)
    ]
    
    results = {}
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"Running: {test_name}")
        print('='*50)
        
        try:
            result = test_func()
            results[test_name] = result
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"{test_name}: {status}")
        except Exception as e:
            results[test_name] = False
            print(f"{test_name}: ❌ FAILED - {e}")
    
    # Summary
    print(f"\n{'='*50}")
    print("TEST SUMMARY")
    print('='*50)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! You can now run the fixed comic search script.")
        print("\nNext steps:")
        print("1. Run: python benchmarks/detections/openrouter/comic_search_with_captions_fixed.py")
        print("2. Wait for indexing to complete")
        print("3. Try: search_text('action scene', hybrid=True)")
    else:
        print("\n⚠️  Some tests failed. Please fix the issues before running the script.")

if __name__ == "__main__":
    main() 