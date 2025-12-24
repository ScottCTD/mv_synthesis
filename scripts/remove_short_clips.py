#!/usr/bin/env python3
"""
Script to remove all video clips with duration less than 1 second.
Scans datasets/explore/videos recursively for all .mp4 files.
Uses multiprocessing to check video durations in parallel.
"""

import subprocess
import sys
from pathlib import Path
from multiprocessing import Pool, cpu_count


def get_video_duration(video_path):
    """
    Get video duration in seconds using ffprobe.
    Returns (video_path, duration) or (video_path, None) if error.
    """
    try:
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            str(video_path)
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=30  # Add timeout to prevent hanging
        )
        duration = float(result.stdout.strip())
        return (video_path, duration)
    except (subprocess.CalledProcessError, ValueError, FileNotFoundError, subprocess.TimeoutExpired) as e:
        return (video_path, None)


def remove_short_clips(directory, min_duration=1.0, dry_run=False, num_workers=None):
    """
    Remove all video clips with duration less than min_duration seconds.
    
    Args:
        directory: Root directory to scan for video files
        min_duration: Minimum duration in seconds (default: 1.0)
        dry_run: If True, only print what would be deleted without actually deleting
        num_workers: Number of worker processes (default: cpu_count())
    """
    directory = Path(directory)
    if not directory.exists():
        print(f"Error: Directory {directory} does not exist", file=sys.stderr)
        return
    
    video_files = list(directory.rglob('*.mp4'))
    print(f"Found {len(video_files)} video files to check...")
    
    if not video_files:
        print("No video files found.")
        return
    
    # Use multiprocessing to check durations in parallel
    if num_workers is None:
        num_workers = cpu_count()
    
    print(f"Using {num_workers} worker processes to check durations...")
    
    removed_count = 0
    error_count = 0
    kept_count = 0
    
    # Process video files in parallel
    with Pool(processes=num_workers) as pool:
        results = pool.map(get_video_duration, video_files)
    
    # Process results and delete files
    for video_path, duration in results:
        if duration is None:
            print(f"Error reading: {video_path}")
            error_count += 1
            continue
        
        if duration < min_duration:
            print(f"Removing: {video_path} (duration: {duration:.3f}s)")
            if not dry_run:
                try:
                    video_path.unlink()
                    removed_count += 1
                except OSError as e:
                    print(f"Error deleting {video_path}: {e}", file=sys.stderr)
                    error_count += 1
        else:
            print(f"Keeping: {video_path} (duration: {duration:.3f}s)")
            kept_count += 1
    
    print(f"\nSummary:")
    print(f"  Removed: {removed_count}")
    print(f"  Errors: {error_count}")
    print(f"  Kept: {kept_count}")
    
    if dry_run:
        print("\n[DRY RUN] No files were actually deleted.")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Remove video clips with duration less than specified seconds'
    )
    parser.add_argument(
        'directory',
        nargs='?',
        default='datasets/explore/videos',
        help='Directory to scan for video files (default: datasets/explore/videos)'
    )
    parser.add_argument(
        '--min-duration',
        type=float,
        default=1.0,
        help='Minimum duration in seconds (default: 1.0)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be deleted without actually deleting'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=None,
        help='Number of worker processes (default: number of CPU cores)'
    )
    
    args = parser.parse_args()
    
    remove_short_clips(args.directory, args.min_duration, args.dry_run, args.workers)

