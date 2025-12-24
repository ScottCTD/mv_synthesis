#!/usr/bin/env python3
"""
Script to count total duration of all videos in a dataset.
Scans a directory recursively for all .mp4 files and calculates total duration.
Uses multiprocessing to check video durations in parallel.
"""

import subprocess
import sys
from pathlib import Path
from multiprocessing import Pool, cpu_count
from collections import defaultdict


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


def format_duration(total_seconds):
    """
    Format duration in seconds to a human-readable string.
    Returns formatted string like "2h 30m 45.123s" or "45.123s"
    """
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = total_seconds % 60
    
    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    parts.append(f"{seconds:.3f}s")
    
    return " ".join(parts)


def count_video_duration(directory, show_breakdown=False, num_workers=None):
    """
    Count total duration of all video files in a directory.
    
    Args:
        directory: Root directory to scan for video files
        show_breakdown: If True, show duration breakdown by subdirectory
        num_workers: Number of worker processes (default: cpu_count())
    """
    directory = Path(directory)
    if not directory.exists():
        print(f"Error: Directory {directory} does not exist", file=sys.stderr)
        return
    
    video_files = list(directory.rglob('*.mp4'))
    print(f"Found {len(video_files)} video files...")
    
    if not video_files:
        print("No video files found.")
        return
    
    # Use multiprocessing to check durations in parallel
    if num_workers is None:
        num_workers = cpu_count()
    
    print(f"Using {num_workers} worker processes to check durations...")
    
    total_duration = 0.0
    error_count = 0
    duration_by_dir = defaultdict(float)
    
    # Process video files in parallel
    with Pool(processes=num_workers) as pool:
        results = pool.map(get_video_duration, video_files)
    
    # Process results and accumulate durations
    for video_path, duration in results:
        if duration is None:
            print(f"Error reading: {video_path}", file=sys.stderr)
            error_count += 1
            continue
        
        total_duration += duration
        
        if show_breakdown:
            # Group by parent directory
            parent_dir = video_path.parent
            duration_by_dir[parent_dir] += duration
    
    # Print results
    print(f"\n{'='*60}")
    print(f"Total Duration: {format_duration(total_duration)}")
    print(f"Total Duration (seconds): {total_duration:.3f}")
    print(f"Total Videos: {len(video_files) - error_count}")
    if error_count > 0:
        print(f"Errors: {error_count}")
    print(f"{'='*60}")
    
    if show_breakdown and duration_by_dir:
        print("\nBreakdown by directory:")
        print("-" * 60)
        # Sort by duration (descending)
        sorted_dirs = sorted(duration_by_dir.items(), key=lambda x: x[1], reverse=True)
        for dir_path, duration in sorted_dirs:
            relative_path = dir_path.relative_to(directory)
            print(f"  {relative_path}: {format_duration(duration)} ({duration:.3f}s)")
        print("-" * 60)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Count total duration of all videos in a dataset'
    )
    parser.add_argument(
        'directory',
        nargs='?',
        default='datasets/explore/videos',
        help='Directory to scan for video files (default: datasets/explore/videos)'
    )
    parser.add_argument(
        '--breakdown',
        action='store_true',
        help='Show duration breakdown by subdirectory'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=None,
        help='Number of worker processes (default: number of CPU cores)'
    )
    
    args = parser.parse_args()
    
    count_video_duration(args.directory, args.breakdown, args.workers)

