#!/usr/bin/env python3
"""
Script to merge very short lyrics lines (< 1 second) with adjacent lines.
Merges short lines with the next line (or previous if it's the last line)
to ensure all lyrics lines have duration >= 1 second.
"""

import re
import shutil
import sys
from pathlib import Path
from typing import Optional, Tuple


# Regex pattern to match lyrics file names: {index}_{start}-{end}.txt
LYRIC_FILE_RE = re.compile(
    r"^(?P<idx>\d+?)_(?P<start>\d+\.\d+)-(?P<end>\d+\.\d+)$"
)
AUDIO_EXTS = (".wav", ".mp3", ".m4a", ".aac")


def parse_lyrics_file(txt_path: Path) -> Optional[Tuple[int, float, float, float]]:
    """
    Parse a lyrics .txt file to extract timing information from filename.
    Returns (index, start, end, duration) or None if error.
    """
    match = LYRIC_FILE_RE.match(txt_path.stem)
    if not match:
        return None
    
    index = int(match.group("idx"))
    start = float(match.group("start"))
    end = float(match.group("end"))
    
    if end <= start:
        return None
    
    duration = end - start
    return (index, start, end, duration)


def find_audio_file(txt_path: Path) -> Optional[Path]:
    """Find the corresponding audio file for a lyrics .txt file."""
    for ext in AUDIO_EXTS:
        candidate = txt_path.with_suffix(ext)
        if candidate.exists():
            return candidate
    return None


def merge_short_lyrics(clips_and_lyrics_dir: Path, min_duration: float = 1.0, dry_run: bool = False):
    """
    Merge very short lyrics lines with adjacent lines.
    
    Args:
        clips_and_lyrics_dir: Path to clips_and_lyrics directory
        min_duration: Minimum duration threshold in seconds (default: 1.0)
        dry_run: If True, only show what would be done without making changes
    """
    clips_and_lyrics_dir = Path(clips_and_lyrics_dir)
    if not clips_and_lyrics_dir.exists():
        print(f"Error: Directory {clips_and_lyrics_dir} does not exist", file=sys.stderr)
        return
    
    if not clips_and_lyrics_dir.is_dir():
        print(f"Error: {clips_and_lyrics_dir} is not a directory", file=sys.stderr)
        return
    
    # Load all lyrics files
    txt_files = sorted(clips_and_lyrics_dir.glob("*.txt"))
    if not txt_files:
        print("No lyrics files found.")
        return
    
    # Parse all files
    lines = []
    for txt_path in txt_files:
        parsed = parse_lyrics_file(txt_path)
        if parsed is None:
            print(f"Warning: Could not parse {txt_path.name}, skipping", file=sys.stderr)
            continue
        
        index, start, end, duration = parsed
        text = txt_path.read_text(encoding="utf-8").strip()
        audio_path = find_audio_file(txt_path)
        
        lines.append({
            'index': index,
            'start': start,
            'end': end,
            'duration': duration,
            'text': text,
            'txt_path': txt_path,
            'audio_path': audio_path,
        })
    
    # Sort by index
    lines.sort(key=lambda x: x['index'])
    
    # Find short lines and plan merges iteratively until all lines >= min_duration
    merge_plan = []
    iteration = 0
    max_iterations = len(lines)  # Safety limit
    
    # Simulated state for dry run
    simulated_lines = lines.copy() if dry_run else None
    
    def get_current_lines():
        """Helper to get current state of lines."""
        if dry_run and simulated_lines is not None:
            return simulated_lines
        else:
            current_txt_files = sorted(clips_and_lyrics_dir.glob("*.txt"))
            current_lines = []
            for txt_path in current_txt_files:
                parsed = parse_lyrics_file(txt_path)
                if parsed is None:
                    continue
                index, start, end, duration = parsed
                text = txt_path.read_text(encoding="utf-8").strip()
                audio_path = find_audio_file(txt_path)
                current_lines.append({
                    'index': index,
                    'start': start,
                    'end': end,
                    'duration': duration,
                    'text': text,
                    'txt_path': txt_path,
                    'audio_path': audio_path,
                })
            current_lines.sort(key=lambda x: x['index'])
            return current_lines
    
    def apply_merges_to_simulated_state(current_lines, iteration_plan):
        """Apply merges to simulated state for dry run."""
        # Remove merged lines and add new merged lines
        indices_to_remove = set()
        new_lines = []
        
        for plan in iteration_plan:
            indices_to_remove.add(plan['from_index'])
            indices_to_remove.add(plan['to_index'])
            new_lines.append({
                'index': min(plan['from_index'], plan['to_index']),
                'start': plan['new_start'],
                'end': plan['new_end'],
                'duration': plan['new_end'] - plan['new_start'],
                'text': plan['new_text'],
                'txt_path': plan['to_txt'],  # Use target as placeholder
                'audio_path': plan['to_audio'],
            })
        
        # Filter out removed lines and add new ones
        filtered_lines = [line for line in current_lines if line['index'] not in indices_to_remove]
        filtered_lines.extend(new_lines)
        filtered_lines.sort(key=lambda x: (x['start'], x['index']))
        
        # Re-index
        for i, line in enumerate(filtered_lines, start=1):
            line['index'] = i
        
        return filtered_lines
    
    while iteration < max_iterations:
        iteration += 1
        current_lines = get_current_lines()
        
        # Find short lines
        short_lines = [i for i, line in enumerate(current_lines) if line['duration'] < min_duration]
        
        if not short_lines:
            break  # All lines are long enough
        
        # Plan merges for this iteration
        merged_indices = set()
        iteration_plan = []
        
        for i in short_lines:
            if i in merged_indices:
                continue
            
            line = current_lines[i]
            
            # Try to merge with next line first
            if i + 1 < len(current_lines) and (i + 1) not in merged_indices:
                next_line = current_lines[i + 1]
                merged_text = f"{line['text']} {next_line['text']}".strip()
                merged_start = line['start']
                merged_end = next_line['end']
                
                iteration_plan.append({
                    'type': 'merge_with_next',
                    'from_index': line['index'],
                    'to_index': next_line['index'],
                    'new_start': merged_start,
                    'new_end': merged_end,
                    'new_text': merged_text,
                    'from_txt': line['txt_path'],
                    'from_audio': line['audio_path'],
                    'to_txt': next_line['txt_path'],
                    'to_audio': next_line['audio_path'],
                })
                merged_indices.add(i)
                merged_indices.add(i + 1)
            # If no next line or next already merged, merge with previous
            elif i > 0 and (i - 1) not in merged_indices:
                prev_line = current_lines[i - 1]
                merged_text = f"{prev_line['text']} {line['text']}".strip()
                merged_start = prev_line['start']
                merged_end = line['end']
                
                iteration_plan.append({
                    'type': 'merge_with_prev',
                    'from_index': line['index'],
                    'to_index': prev_line['index'],
                    'new_start': merged_start,
                    'new_end': merged_end,
                    'new_text': merged_text,
                    'from_txt': line['txt_path'],
                    'from_audio': line['audio_path'],
                    'to_txt': prev_line['txt_path'],
                    'to_audio': prev_line['audio_path'],
                })
                merged_indices.add(i)
                merged_indices.add(i - 1)
        
        if not iteration_plan:
            print("Warning: Could not merge all short lines (some may be isolated).")
            break
        
        merge_plan.extend(iteration_plan)
        
        # Update simulated state for dry run
        if dry_run:
            simulated_lines = apply_merges_to_simulated_state(current_lines, iteration_plan)
        else:
            # Perform merges for this iteration
            import subprocess
            files_to_delete = set()
            
            for plan in iteration_plan:
                new_index = min(plan['from_index'], plan['to_index'])
                new_start_str = f"{plan['new_start']:07.3f}"
                new_end_str = f"{plan['new_end']:07.3f}"
                new_base = f"{new_index:04d}_{new_start_str}-{new_end_str}"
                
                new_txt_path = clips_and_lyrics_dir / f"{new_base}.txt"
                new_audio_path = clips_and_lyrics_dir / f"{new_base}.wav"
                
                files_to_delete.add(plan['from_txt'])
                files_to_delete.add(plan['to_txt'])
                if plan['from_audio']:
                    files_to_delete.add(plan['from_audio'])
                if plan['to_audio']:
                    files_to_delete.add(plan['to_audio'])
                
                new_txt_path.write_text(plan['new_text'], encoding="utf-8")
                
                # Merge audio files
                if plan['from_audio'] and plan['to_audio'] and plan['from_audio'].exists() and plan['to_audio'].exists():
                    concat_list_path = clips_and_lyrics_dir / f"temp_concat_{new_index}.txt"
                    try:
                        with open(concat_list_path, 'w') as f:
                            f.write(f"file '{plan['from_audio'].absolute()}'\n")
                            f.write(f"file '{plan['to_audio'].absolute()}'\n")
                        
                        cmd = [
                            'ffmpeg', '-y', '-f', 'concat', '-safe', '0',
                            '-i', str(concat_list_path),
                            '-c', 'copy',
                            str(new_audio_path)
                        ]
                        result = subprocess.run(cmd, capture_output=True, text=True)
                        if result.returncode != 0:
                            print(f"Warning: Failed to merge audio for {new_base}: {result.stderr}", file=sys.stderr)
                    except Exception as e:
                        print(f"Warning: Failed to merge audio for {new_base}: {e}", file=sys.stderr)
                    finally:
                        if concat_list_path.exists():
                            concat_list_path.unlink()
                elif plan['to_audio'] and plan['to_audio'].exists():
                    shutil.copy2(plan['to_audio'], new_audio_path)
                elif plan['from_audio'] and plan['from_audio'].exists():
                    shutil.copy2(plan['from_audio'], new_audio_path)
            
            # Delete old files
            for file_path in files_to_delete:
                if file_path and file_path.exists():
                    file_path.unlink()
            
            # Re-index after this iteration
            remaining_txt_files = sorted([f for f in clips_and_lyrics_dir.glob("*.txt")])
            remaining_lines = []
            for txt_path in remaining_txt_files:
                parsed = parse_lyrics_file(txt_path)
                if parsed is None:
                    continue
                index, start, end, duration = parsed
                remaining_lines.append({
                    'txt_path': txt_path,
                    'start': start,
                    'end': end,
                })
            
            remaining_lines.sort(key=lambda x: x['start'])
            
            for new_index, line_info in enumerate(remaining_lines, start=1):
                txt_path = line_info['txt_path']
                start = line_info['start']
                end = line_info['end']
                
                new_base = f"{new_index:04d}_{start:07.3f}-{end:07.3f}"
                new_txt_path = clips_and_lyrics_dir / f"{new_base}.txt"
                
                if txt_path != new_txt_path:
                    txt_path.rename(new_txt_path)
                    audio_path = find_audio_file(txt_path)
                    if audio_path and audio_path.exists():
                        new_audio_path = clips_and_lyrics_dir / f"{new_base}.wav"
                        audio_path.rename(new_audio_path)
    
    if not merge_plan:
        print("No short lines found to merge.")
        return
    
    # Print merge plan
    print(f"Found {len(merge_plan)} merge(s) to perform:")
    print("-" * 80)
    for plan in merge_plan:
        duration = plan['new_end'] - plan['new_start']
        print(f"  Merge line {plan['from_index']:04d} ({plan['from_txt'].name}) "
              f"with line {plan['to_index']:04d} ({plan['to_txt'].name})")
        print(f"    Duration: {duration:.3f}s {'✓' if duration >= min_duration else '⚠ Still < ' + str(min_duration) + 's'}")
        print(f"    Text: {plan['new_text'][:60]}...")
    
    if dry_run:
        print("\n[DRY RUN] No changes made. Use without --dry-run to apply changes.")
        return
    
    # Confirm
    print(f"\nThis will perform {len(merge_plan)} merge(s) across {iteration} iteration(s).")
    response = input("Continue? (y/N): ").strip().lower()
    if response != 'y':
        print("Aborted.")
        return
    
    # Merges are already performed in the loop above
    print("\nDone!")
    print(f"Total merges performed: {len(merge_plan)}")
    print(f"Iterations: {iteration}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Merge very short lyrics lines (< 1 second) with adjacent lines'
    )
    parser.add_argument(
        'clips_and_lyrics_dir',
        nargs='?',
        default='datasets/ds2/songs/sunshine/clips_and_lyrics',
        help='Path to clips_and_lyrics directory'
    )
    parser.add_argument(
        '--min-duration',
        type=float,
        default=1.0,
        help='Minimum duration threshold in seconds (default: 1.0)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without making changes'
    )
    
    args = parser.parse_args()
    
    merge_short_lyrics(Path(args.clips_and_lyrics_dir), args.min_duration, args.dry_run)

