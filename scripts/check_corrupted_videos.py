#!/usr/bin/env python3
"""
Script to scan a directory of .mp4 files and use ffmpeg to find possible corrupted videos.
Scans a directory recursively for all .mp4 files and validates them using ffmpeg.
Uses multiprocessing to check videos in parallel.
"""

import subprocess
import sys
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial


def check_video_corruption(video_path, timeout=60, use_gpu=False):
    """
    Check if a video file is corrupted using ffmpeg.
    
    Uses ffmpeg to attempt to decode the video and checks for errors.
    Returns (video_path, is_corrupted, error_message).
    is_corrupted is True if the video appears to be corrupted, False otherwise.
    error_message contains any error output from ffmpeg, or None if no errors.
    
    Args:
        video_path: Path to the video file to check
        timeout: Maximum time in seconds to wait for ffmpeg to complete
        use_gpu: If True, use hardware-accelerated decoding (videotoolbox on macOS)
    
    Returns:
        Tuple of (video_path, is_corrupted, error_message)
    """
    try:
        # Use ffmpeg to attempt to decode the video
        # -v error: only show errors (not warnings/info)
        # -hwaccel videotoolbox: use GPU hardware acceleration (macOS)
        # -i: input file
        # -f null: output to null format (we don't need the decoded output)
        # -: output to stdout (but we discard it)
        cmd = ['ffmpeg', '-v', 'error']
        
        if use_gpu:
            cmd.extend(['-hwaccel', 'videotoolbox'])
        
        cmd.extend([
            '-i', str(video_path),
            '-f', 'null',
            '-'
        ])
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,  # Don't raise on non-zero exit
            timeout=timeout
        )
        
        # If there's any output to stderr or non-zero exit code, the video may be corrupted
        # Note: exit code 1 from ffmpeg can also mean normal termination in some cases,
        # but combined with stderr output, it usually indicates an error
        if result.returncode != 0 or result.stderr:
            error_msg = result.stderr.strip() if result.stderr else f"Exit code: {result.returncode}"
            return (video_path, True, error_msg)
        else:
            return (video_path, False, None)
            
    except subprocess.TimeoutExpired:
        return (video_path, True, f"Timeout after {timeout} seconds")
    except FileNotFoundError:
        return (video_path, True, "ffmpeg not found. Please install ffmpeg.")
    except Exception as e:
        return (video_path, True, f"Unexpected error: {str(e)}")


def scan_corrupted_videos(directory, show_errors=False, delete_corrupted=False, dry_run=False, 
                          num_workers=None, timeout=60, use_gpu=False):
    """
    Scan a directory for corrupted video files.
    
    Args:
        directory: Root directory to scan for video files
        show_errors: If True, display error messages from corrupted videos
        delete_corrupted: If True, delete corrupted video files
        dry_run: If True, only show what would be deleted without actually deleting
        num_workers: Number of worker processes (default: cpu_count())
        timeout: Maximum time in seconds to wait for each video check (default: 60)
        use_gpu: If True, use hardware-accelerated decoding (videotoolbox on macOS)
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
    
    # Use multiprocessing to check videos in parallel
    if num_workers is None:
        num_workers = cpu_count()
    
    print(f"Using {num_workers} worker processes to check videos...")
    print(f"Timeout per video: {timeout} seconds")
    if use_gpu:
        print(f"GPU acceleration: enabled (videotoolbox)")
    print(f"\nScanning for corrupted videos...\n")
    
    corrupted_videos = []
    valid_videos = []
    
    # Process video files in parallel
    # Use functools.partial to create a picklable function with timeout and gpu parameters
    check_with_params = partial(check_video_corruption, timeout=timeout, use_gpu=use_gpu)
    
    with Pool(processes=num_workers) as pool:
        results = pool.map(check_with_params, video_files)
    
    # Process results
    for video_path, is_corrupted, error_message in results:
        if is_corrupted:
            corrupted_videos.append((video_path, error_message))
            status = "CORRUPTED"
            print(f"[{status}] {video_path}")
            if show_errors and error_message:
                # Indent error message
                for line in error_message.split('\n'):
                    print(f"         {line}")
        else:
            valid_videos.append(video_path)
            # Only print valid videos if verbose mode or if there are few files
            if len(video_files) <= 50:
                print(f"[OK] {video_path}")
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Scan Summary:")
    print(f"  Total videos: {len(video_files)}")
    print(f"  Valid videos: {len(valid_videos)}")
    print(f"  Corrupted videos: {len(corrupted_videos)}")
    print(f"{'='*60}")
    
    # Delete corrupted videos if requested
    if delete_corrupted and corrupted_videos:
        print(f"\n{'='*60}")
        if dry_run:
            print(f"[DRY RUN] Would delete {len(corrupted_videos)} corrupted video(s):")
        else:
            print(f"Deleting {len(corrupted_videos)} corrupted video(s):")
        print("-" * 60)
        
        deleted_count = 0
        failed_deletions = []
        
        for video_path, error_message in corrupted_videos:
            print(f"  {video_path}")
            if not dry_run:
                try:
                    video_path.unlink()
                    deleted_count += 1
                except Exception as e:
                    print(f"    Failed to delete: {e}", file=sys.stderr)
                    failed_deletions.append((video_path, str(e)))
        
        print("-" * 60)
        if dry_run:
            print(f"[DRY RUN] Would delete: {len(corrupted_videos)}")
        else:
            print(f"Successfully deleted: {deleted_count}")
            if failed_deletions:
                print(f"Failed to delete: {len(failed_deletions)}")
        print(f"{'='*60}")
    
    # Exit with error code if corrupted videos were found
    if corrupted_videos:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Scan a directory of .mp4 files and use ffmpeg to find possible corrupted videos'
    )
    parser.add_argument(
        'directory',
        nargs='?',
        default='datasets/raw',
        help='Directory to scan for video files (default: datasets/raw)'
    )
    parser.add_argument(
        '--show-errors',
        action='store_true',
        help='Display error messages from corrupted videos'
    )
    parser.add_argument(
        '--delete',
        action='store_true',
        dest='delete_corrupted',
        help='Delete corrupted video files'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be deleted without actually deleting (requires --delete)'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=None,
        help='Number of worker processes (default: number of CPU cores)'
    )
    parser.add_argument(
        '--timeout',
        type=int,
        default=60,
        help='Maximum time in seconds to wait for each video check (default: 60)'
    )
    parser.add_argument(
        '--use-gpu',
        action='store_true',
        help='Use GPU hardware-accelerated decoding (videotoolbox on macOS)'
    )
    
    args = parser.parse_args()
    
    scan_corrupted_videos(
        args.directory,
        args.show_errors,
        args.delete_corrupted,
        args.dry_run,
        args.workers,
        args.timeout,
        args.use_gpu
    )

