#!/usr/bin/env python3
"""
Script to count total duration of all videos in a dataset.
Scans a directory recursively for all .mp4 and .mkv files and calculates total duration.
Uses multiprocessing to check video durations in parallel.
"""

import subprocess
import sys
from pathlib import Path
from multiprocessing import Pool, cpu_count
from collections import defaultdict
import statistics


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


def find_video_files(directory, max_depth=None):
    """
    Find all video files (.mp4 and .mkv) in a directory with optional depth limit.
    
    Args:
        directory: Root directory to search
        max_depth: Maximum recursion depth (None for unlimited)
    
    Returns:
        List of Path objects for video files
    """
    directory = Path(directory)
    video_files = []
    
    def _search_recursive(path, current_depth):
        """Recursive helper function to search with depth tracking."""
        if max_depth is not None and current_depth > max_depth:
            return
        
        # Check for video files in current directory
        for ext in ['*.mp4', '*.mkv']:
            video_files.extend(path.glob(ext))
        
        # Recurse into subdirectories
        if max_depth is None or current_depth < max_depth:
            try:
                for item in path.iterdir():
                    if item.is_dir():
                        _search_recursive(item, current_depth + 1)
            except PermissionError:
                # Skip directories we don't have permission to access
                pass
    
    _search_recursive(directory, 0)
    return video_files


def count_video_duration(directory, show_breakdown=False, num_workers=None, remove_errors=False, max_depth=None):
    """
    Count total duration of all video files in a directory.
    
    Args:
        directory: Root directory to scan for video files
        show_breakdown: If True, show duration breakdown by subdirectory
        num_workers: Number of worker processes (default: cpu_count())
        remove_errors: If True, delete errored video files and print their paths
        max_depth: Maximum recursion depth (None for unlimited)
    """
    directory = Path(directory)
    if not directory.exists():
        print(f"Error: Directory {directory} does not exist", file=sys.stderr)
        return
    
    video_files = find_video_files(directory, max_depth)
    depth_info = f" (max depth: {max_depth})" if max_depth is not None else " (unlimited depth)"
    print(f"Found {len(video_files)} video files{depth_info}...")
    
    if not video_files:
        print("No video files found.")
        return
    
    # Use multiprocessing to check durations in parallel
    if num_workers is None:
        num_workers = cpu_count()
    
    print(f"Using {num_workers} worker processes to check durations...")
    
    total_duration = 0.0
    error_count = 0
    errored_videos = []  # Store paths of errored videos
    duration_by_dir = defaultdict(float)
    durations = []  # Store all durations for statistics
    
    # Process video files in parallel
    with Pool(processes=num_workers) as pool:
        results = pool.map(get_video_duration, video_files)
    
    # Process results and accumulate durations
    for video_path, duration in results:
        if duration is None:
            print(f"Error reading: {video_path}", file=sys.stderr)
            error_count += 1
            errored_videos.append(video_path)
            continue
        
        total_duration += duration
        durations.append(duration)
        
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
    
    # Calculate and print statistics
    if durations:
        durations_sorted = sorted(durations)
        mean_duration = statistics.mean(durations)
        median_duration = statistics.median(durations)
        min_duration = min(durations)
        max_duration = max(durations)
        stdev_duration = statistics.stdev(durations) if len(durations) > 1 else 0.0
        
        # Calculate percentiles
        n = len(durations_sorted)
        p25 = durations_sorted[n // 4] if n > 0 else 0
        p75 = durations_sorted[3 * n // 4] if n > 0 else 0
        p90 = durations_sorted[int(0.9 * n)] if n > 0 else 0
        p95 = durations_sorted[int(0.95 * n)] if n > 0 else 0
        
        print(f"\n{'='*60}")
        print("Duration Statistics:")
        print("-" * 60)
        print(f"  Mean:   {format_duration(mean_duration)} ({mean_duration:.3f}s)")
        print(f"  Median: {format_duration(median_duration)} ({median_duration:.3f}s)")
        print(f"  Min:    {format_duration(min_duration)} ({min_duration:.3f}s)")
        print(f"  Max:    {format_duration(max_duration)} ({max_duration:.3f}s)")
        print(f"  StdDev: {format_duration(stdev_duration)} ({stdev_duration:.3f}s)")
        print(f"\n  Percentiles:")
        print(f"    25th: {format_duration(p25)} ({p25:.3f}s)")
        print(f"    75th: {format_duration(p75)} ({p75:.3f}s)")
        print(f"    90th: {format_duration(p90)} ({p90:.3f}s)")
        print(f"    95th: {format_duration(p95)} ({p95:.3f}s)")
        print(f"{'='*60}")
        
        # Print distribution histogram
        print(f"\n{'='*60}")
        print("Duration Distribution:")
        print("-" * 60)
        
        # Create bins for histogram with finer granularity for < 10 seconds
        if max_duration > 0:
            # Use 1-second bins for 0-10 seconds (10 bins)
            # Then use larger bins for > 10 seconds
            bin_edges = []
            
            # Fine-grained bins for 0-10 seconds (1 second each)
            for i in range(11):  # 0, 1, 2, ..., 10
                bin_edges.append(i)
            
            # Coarser bins for > 10 seconds (5 seconds each)
            if max_duration > 10:
                current_edge = 10
                while current_edge < max_duration:
                    current_edge += 5
                    bin_edges.append(current_edge)
            
            # Ensure the last bin edge covers max_duration
            if bin_edges[-1] < max_duration:
                bin_edges.append(max_duration)
            
            num_bins = len(bin_edges) - 1
            bins = [0] * num_bins
            
            # Assign durations to bins
            for duration in durations:
                bin_idx = num_bins - 1  # Default to last bin
                for i in range(num_bins):
                    if duration < bin_edges[i + 1]:
                        bin_idx = i
                        break
                bins[bin_idx] += 1
            
            # Find max count for scaling
            max_count = max(bins) if bins else 1
            bar_length = 50  # Maximum bar length in characters
            
            # Print bins
            for i in range(num_bins):
                bin_start = bin_edges[i]
                bin_end = bin_edges[i + 1]
                count = bins[i]
                bar = '█' * int((count / max_count) * bar_length) if max_count > 0 else ''
                print(f"  {bin_start:8.2f}s - {bin_end:8.2f}s: {count:5d} {bar}")
        
        print(f"{'='*60}")
    
    # Remove errored videos if requested
    if remove_errors and errored_videos:
        print(f"\n{'='*60}")
        print(f"Removing {len(errored_videos)} errored video(s):")
        print("-" * 60)
        removed_count = 0
        failed_removals = []
        
        for video_path in errored_videos:
            try:
                video_path.unlink()  # Delete the file
                print(f"  Removed: {video_path}")
                removed_count += 1
            except Exception as e:
                print(f"  Failed to remove {video_path}: {e}", file=sys.stderr)
                failed_removals.append(video_path)
        
        print("-" * 60)
        print(f"Successfully removed: {removed_count}")
        if failed_removals:
            print(f"Failed to remove: {len(failed_removals)}")
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
        default=32,
        help='Number of worker processes (default: number of CPU cores)'
    )
    parser.add_argument(
        '--remove-errors',
        action='store_true',
        help='Remove all errored video files and print their paths'
    )
    parser.add_argument(
        '--max-depth',
        type=int,
        default=None,
        metavar='N',
        help='Maximum recursion depth (default: unlimited). 0 = current directory only, 1 = one level deep, etc. Use -1 for unlimited depth.'
    )
    
    args = parser.parse_args()
    
    # Convert -1 to None (unlimited depth)
    max_depth = None if args.max_depth == -1 else args.max_depth
    
    count_video_duration(args.directory, args.breakdown, args.workers, args.remove_errors, max_depth)

