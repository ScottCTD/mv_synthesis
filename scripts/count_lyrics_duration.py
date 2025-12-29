#!/usr/bin/env python3
"""
Script to count total duration of all lyrics lines in a clips_and_lyrics directory.
Scans a clips_and_lyrics directory for all .txt files and calculates total duration
based on the timing information in filenames (format: {index}_{start}-{end}.txt).
"""

import re
import sys
from pathlib import Path
from collections import defaultdict
import statistics


# Regex pattern to match lyrics file names: {index}_{start}-{end}.txt
LYRIC_FILE_RE = re.compile(
    r"^(?P<idx>\d+?)_(?P<start>\d+\.\d+)-(?P<end>\d+\.\d+)$"
)


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


def parse_lyrics_file(txt_path):
    """
    Parse a lyrics .txt file to extract timing information from filename.
    Returns (txt_path, index, start, end, duration) or (txt_path, None, None, None, None) if error.
    """
    try:
        match = LYRIC_FILE_RE.match(txt_path.stem)
        if not match:
            return (txt_path, None, None, None, None)
        
        index = int(match.group("idx"))
        start = float(match.group("start"))
        end = float(match.group("end"))
        
        if end <= start:
            return (txt_path, None, None, None, None)
        
        duration = end - start
        return (txt_path, index, start, end, duration)
    except (ValueError, AttributeError) as e:
        return (txt_path, None, None, None, None)


def count_lyrics_duration(clips_and_lyrics_dir, show_breakdown=False):
    """
    Count total duration of all lyrics lines in a clips_and_lyrics directory.
    
    Args:
        clips_and_lyrics_dir: Path to clips_and_lyrics directory
        show_breakdown: If True, show duration breakdown by index ranges
    """
    clips_and_lyrics_dir = Path(clips_and_lyrics_dir)
    if not clips_and_lyrics_dir.exists():
        print(f"Error: Directory {clips_and_lyrics_dir} does not exist", file=sys.stderr)
        return
    
    if not clips_and_lyrics_dir.is_dir():
        print(f"Error: {clips_and_lyrics_dir} is not a directory", file=sys.stderr)
        return
    
    # Find all .txt files
    txt_files = sorted(clips_and_lyrics_dir.glob("*.txt"))
    print(f"Found {len(txt_files)} lyrics files...")
    
    if not txt_files:
        print("No lyrics files found.")
        return
    
    # Parse all lyrics files
    total_duration = 0.0
    error_count = 0
    errored_files = []
    durations = []
    lines_data = []  # Store (index, start, end, duration) tuples
    
    for txt_path in txt_files:
        txt_path, index, start, end, duration = parse_lyrics_file(txt_path)
        if duration is None:
            print(f"Error parsing: {txt_path}", file=sys.stderr)
            error_count += 1
            errored_files.append(txt_path)
            continue
        
        total_duration += duration
        durations.append(duration)
        lines_data.append((index, start, end, duration))
    
    # Print results
    print(f"\n{'='*60}")
    print(f"Total Duration: {format_duration(total_duration)}")
    print(f"Total Duration (seconds): {total_duration:.3f}")
    print(f"Total Lines: {len(txt_files) - error_count}")
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
    
    # Show breakdown by index ranges if requested
    if show_breakdown and lines_data:
        print("\nBreakdown by index ranges:")
        print("-" * 60)
        
        # Group by index ranges (every 10 lines)
        duration_by_range = defaultdict(float)
        for index, start, end, duration in lines_data:
            range_start = (index - 1) // 10 * 10 + 1
            range_end = range_start + 9
            range_key = f"{range_start:04d}-{range_end:04d}"
            duration_by_range[range_key] += duration
        
        # Sort by range start
        sorted_ranges = sorted(duration_by_range.items(), key=lambda x: int(x[0].split('-')[0]))
        for range_key, duration in sorted_ranges:
            print(f"  {range_key}: {format_duration(duration)} ({duration:.3f}s)")
        print("-" * 60)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Count total duration of all lyrics lines in a clips_and_lyrics directory'
    )
    parser.add_argument(
        'clips_and_lyrics_dir',
        nargs='?',
        default='datasets/ds2/songs/sunshine/clips_and_lyrics',
        help='Path to clips_and_lyrics directory (default: datasets/ds2/songs/sunshine/clips_and_lyrics)'
    )
    parser.add_argument(
        '--breakdown',
        action='store_true',
        help='Show duration breakdown by index ranges'
    )
    
    args = parser.parse_args()
    
    count_lyrics_duration(args.clips_and_lyrics_dir, args.breakdown)

