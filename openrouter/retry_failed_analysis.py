#!/usr/bin/env python3
"""
Retry failed comic analysis with sophisticated retry logic.
Handles probabilistic failures in VLM/LLM analysis and provides detailed reporting.
"""

import argparse
import subprocess
import time
import json
from pathlib import Path
from datetime import datetime
import sqlite3
from tqdm import tqdm
import os

def count_existing_results(output_dir):
    """Count existing JSON results in output directory."""
    output_path = Path(output_dir)
    if not output_path.exists():
        return 0
    
    json_files = list(output_path.rglob("*.json"))
    return len(json_files)

def count_total_images(input_dir):
    """Count total images that should be processed."""
    input_path = Path(input_dir)
    if not input_path.exists():
        return 0
    
    image_count = 0
    for subdir in input_path.iterdir():
        if subdir.is_dir():
            image_count += len(list(subdir.glob("*.jpg")))
    
    return image_count

def run_analysis_attempt(input_dir, output_dir, max_workers=4, max_images=None, model=None):
    """Run a single analysis attempt and return results."""
    cmd = [
        "python", "benchmarks/detections/openrouter/batch_comic_analysis_multi.py",
        "--input-dir", str(input_dir),
        "--output-dir", str(output_dir),
        "--skip-existing",
        "--max-workers", str(max_workers)
    ]
    
    if max_images:
        cmd.extend(["--max-images", str(max_images)])
    
    if model:
        cmd.extend(["--model", model])
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)  # 1 hour timeout
        return {
            'success': result.returncode == 0,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'returncode': result.returncode
        }
    except subprocess.TimeoutExpired:
        return {
            'success': False,
            'stdout': '',
            'stderr': 'Timeout after 1 hour',
            'returncode': -1
        }
    except Exception as e:
        return {
            'success': False,
            'stdout': '',
            'stderr': str(e),
            'returncode': -1
        }

def analyze_failures(output_dir):
    """Analyze what types of failures occurred."""
    output_path = Path(output_dir)
    if not output_path.exists():
        return {}
    
    # Look for error patterns in the output
    error_patterns = {
        'api_auth': ['401', 'auth', 'authentication'],
        'rate_limit': ['429', 'rate limit', 'too many requests'],
        'timeout': ['timeout', 'timed out'],
        'json_parse': ['json parse', 'json error'],
        'network': ['connection', 'network', 'dns'],
        'model_error': ['model', 'inference', 'generation']
    }
    
    error_counts = {category: 0 for category in error_patterns.keys()}
    error_counts['other'] = 0
    
    # This is a simplified analysis - in practice you'd parse the actual error logs
    return error_counts

def create_retry_report(attempts, input_dir, output_dir, start_time):
    """Create a detailed retry report."""
    end_time = datetime.now()
    total_time = (end_time - start_time).total_seconds()
    
    initial_count = count_existing_results(output_dir)
    final_count = count_existing_results(output_dir)
    total_images = count_total_images(input_dir)
    
    successful_attempts = sum(1 for attempt in attempts if attempt['success'])
    
    report = {
        'timestamp': end_time.isoformat(),
        'total_time_seconds': total_time,
        'total_attempts': len(attempts),
        'successful_attempts': successful_attempts,
        'input_directory': str(input_dir),
        'output_directory': str(output_dir),
        'total_images': total_images,
        'initial_results': initial_count,
        'final_results': final_count,
        'new_results': final_count - initial_count,
        'success_rate': (successful_attempts / len(attempts)) * 100 if attempts else 0,
        'attempts': attempts
    }
    
    return report

def save_retry_report(report, output_path):
    """Save retry report to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)

def main():
    parser = argparse.ArgumentParser(description='Retry failed comic analysis with sophisticated retry logic')
    parser.add_argument('--input-dir', type=str, 
                       default=r'C:\Users\Richard\OneDrive\GIT\CoMix\data\datasets.unify\2000ad\images',
                       help='Input directory containing comic images')
    parser.add_argument('--output-dir', type=str, 
                       default='benchmarks/detections/openrouter/analysis_results_multi',
                       help='Output directory for analysis results')
    parser.add_argument('--max-retries', type=int, default=3,
                       help='Maximum number of retry attempts')
    parser.add_argument('--max-workers', type=int, default=4,
                       help='Number of worker processes')
    parser.add_argument('--retry-delay', type=int, default=30,
                       help='Delay between retry attempts (seconds)')
    parser.add_argument('--max-images', type=int, default=None,
                       help='Maximum number of images to process (for testing)')
    parser.add_argument('--model', type=str, default=None,
                       help='Model to use for analysis')
    parser.add_argument('--report-file', type=str, default='retry_report.json',
                       help='File to save retry report')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("RETRY FAILED COMIC ANALYSIS")
    print("=" * 60)
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Max retries: {args.max_retries}")
    print(f"Max workers: {args.max_workers}")
    print(f"Retry delay: {args.retry_delay} seconds")
    print(f"Max images: {args.max_images or 'All'}")
    print(f"Model: {args.model or 'Default'}")
    print("=" * 60)
    
    # Initial setup
    start_time = datetime.now()
    attempts = []
    
    # Count initial results
    initial_count = count_existing_results(args.output_dir)
    total_images = count_total_images(args.input_dir)
    
    print(f"\nInitial analysis results: {initial_count}")
    print(f"Total images to process: {total_images}")
    print(f"Remaining images: {total_images - initial_count}")
    
    # Retry loop
    for attempt_num in range(1, args.max_retries + 1):
        print(f"\n{'='*20} ATTEMPT {attempt_num} {'='*20}")
        
        # Run analysis
        print(f"Running analysis attempt {attempt_num}...")
        result = run_analysis_attempt(
            args.input_dir, 
            args.output_dir, 
            args.max_workers, 
            args.max_images, 
            args.model
        )
        
        # Record attempt
        attempts.append({
            'attempt_number': attempt_num,
            'timestamp': datetime.now().isoformat(),
            'success': result['success'],
            'returncode': result['returncode'],
            'stdout': result['stdout'][-500:] if result['stdout'] else '',  # Last 500 chars
            'stderr': result['stderr'][-500:] if result['stderr'] else ''   # Last 500 chars
        })
        
        # Report results
        current_count = count_existing_results(args.output_dir)
        new_results = current_count - initial_count
        
        print(f"Attempt {attempt_num} {'SUCCESS' if result['success'] else 'FAILED'}")
        print(f"Return code: {result['returncode']}")
        print(f"New results in this attempt: {new_results}")
        print(f"Total results so far: {current_count}")
        
        if result['stderr']:
            print(f"Errors: {result['stderr'][-200:]}")  # Last 200 chars
        
        # Check if we should continue
        if attempt_num < args.max_retries:
            print(f"\nWaiting {args.retry_delay} seconds before next attempt...")
            time.sleep(args.retry_delay)
        else:
            print(f"\nReached maximum retry attempts ({args.max_retries})")
    
    # Final analysis
    print(f"\n{'='*20} FINAL SUMMARY {'='*20}")
    
    final_count = count_existing_results(args.output_dir)
    total_new = final_count - initial_count
    
    print(f"Initial results: {initial_count}")
    print(f"Final results: {final_count}")
    print(f"New results: {total_new}")
    print(f"Success rate: {(sum(1 for a in attempts if a['success']) / len(attempts)) * 100:.1f}%")
    
    # Create and save report
    report = create_retry_report(attempts, args.input_dir, args.output_dir, start_time)
    save_retry_report(report, args.report_file)
    
    print(f"\nDetailed report saved to: {args.report_file}")
    
    # Optional: Update database
    if total_new > 0:
        print(f"\nUpdating database with {total_new} new results...")
        try:
            subprocess.run([
                "python", "benchmarks/detections/openrouter/organize_analysis_results.py",
                "--input-dir", args.output_dir,
                "--output-db", "comic_analysis_retry.db"
            ], check=True)
            print("Database updated successfully!")
        except subprocess.CalledProcessError as e:
            print(f"Database update failed: {e}")
    
    print(f"\nRetry analysis completed in {report['total_time_seconds']:.1f} seconds")

if __name__ == "__main__":
    main() 