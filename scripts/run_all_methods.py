#!/usr/bin/env python3
"""
Run all synthesis methods for all songs in a dataset with multiprocessing support.

This script runs multiple methods (random, text_video, fused) for all songs
in a dataset, allowing per-song argument overrides and configurable concurrency.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

# Add src to path to import config
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from synthesis.config import PROJECT_ROOT


@dataclass
class CommandConfig:
    """Configuration for a single synthesis command."""

    dataset_name: str
    song_name: str
    args: dict[str, Any]

    def to_command(self) -> list[str]:
        """Convert to command-line arguments."""
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "src" / "synthesis" / "synthesize_pipeline.py"),
            self.dataset_name,
            self.song_name,
        ]
        for key, value in self.args.items():
            if value is None:
                continue
            if isinstance(value, bool):
                if value:
                    cmd.append(f"--{key}")
            else:
                cmd.extend([f"--{key}", str(value)])
        return cmd


def discover_songs(dataset_name: str) -> list[str]:
    """Discover all songs in a dataset."""
    songs_dir = PROJECT_ROOT / "datasets" / dataset_name / "songs"
    if not songs_dir.exists():
        raise FileNotFoundError(f"Songs directory not found: {songs_dir}")

    songs = []
    for item in songs_dir.iterdir():
        if item.is_dir() and (item / "lyrics_lines.json").exists():
            songs.append(item.name)

    return sorted(songs)


def get_random_method_commands(
    dataset_name: str, songs: list[str]
) -> list[CommandConfig]:
    """Generate commands for random method."""
    commands = []
    for song in songs:
        commands.append(
            CommandConfig(
                dataset_name=dataset_name,
                song_name=song,
                args={
                    "run-name": "random",
                    "query-source": "random",
                    "selection-strategy": "top_video_duration",
                },
            )
        )
    return commands


def get_text_video_method_commands(
    dataset_name: str, songs: list[str]
) -> list[CommandConfig]:
    """Generate commands for text_video method."""
    commands = []
    for song in songs:
        commands.append(
            CommandConfig(
                dataset_name=dataset_name,
                song_name=song,
                args={
                    "run-name": "text_video",
                    "query-source": "text",
                    "selection-strategy": "top_video_duration",
                },
            )
        )
    return commands


def get_fused_method_commands(
    dataset_name: str,
    songs: list[str],
    song_overrides: Optional[dict[str, dict[str, Any]]] = None,
) -> list[CommandConfig]:
    """Generate commands for fused method with per-song configurations."""
    if song_overrides is None:
        song_overrides = {}

    # Default fused configuration
    default_fused_args = {
        "run-name": "fused_rank",
        "selection-strategy": "fused_rank",
        "fused-text-source": "combined",
        "fused-weight-tv": 0.4,
        "fused-weight-tvc": 0.4,
        "fused-weight-av": 0.1,
        "fused-weight-avc": 0.1,
        "fused-anti-repeat": -1,  # disallow all repetitions
    }

    # Per-song overrides from README
    default_overrides = {
        "countingStars": {
            "fused-weight-tv": 0.4,
            "fused-weight-tvc": 0.4,
            "fused-weight-av": 0.1,
            "fused-weight-avc": 0.1,
        },
        "sunshine": {
            "fused-weight-tv": 0.4,
            "fused-weight-tvc": 0.4,
            "fused-weight-av": 0.1,
            "fused-weight-avc": 0.1,
        },
        "happy": {
            "fused-weight-tv": 0.4,
            "fused-weight-tvc": 0.4,
            "fused-weight-av": 0.1,
            "fused-weight-avc": 0.1,
        },
        "payphone": {
            "fused-weight-tv": 0.4,
            "fused-weight-tvc": 0.4,
            "fused-weight-av": 0.1,
            "fused-weight-avc": 0.1,
        },
        "firework": {
            "fused-weight-tv": 0.4,
            "fused-weight-tvc": 0.4,
            "fused-weight-av": 0.1,
            "fused-weight-avc": 0.1,
        },
    }

    commands = []
    for song in songs:
        # Start with default args
        args = default_fused_args.copy()

        # Apply default overrides if available
        if song in default_overrides:
            args.update(default_overrides[song])

        # Apply user-provided overrides (highest priority)
        if song in song_overrides:
            args.update(song_overrides[song])

        commands.append(
            CommandConfig(dataset_name=dataset_name, song_name=song, args=args)
        )

    return commands


def run_command(
    config: CommandConfig, stream_output: bool = False
) -> tuple[str, bool, str]:
    """Run a single synthesis command and return (song_name, success, output)."""
    cmd = config.to_command()
    try:
        # Set PYTHONPATH to include src directory
        env = os.environ.copy()
        src_path = str(PROJECT_ROOT / "src")
        if "PYTHONPATH" in env:
            env["PYTHONPATH"] = f"{src_path}:{env['PYTHONPATH']}"
        else:
            env["PYTHONPATH"] = src_path

        if stream_output:
            # Stream output in real-time for tqdm progress bars
            # Use unbuffered mode to ensure tqdm updates are visible
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
                cwd=str(PROJECT_ROOT),
                bufsize=1,  # Line buffered for real-time output
            )

            output_lines = []
            # Read and print line by line for real-time display
            for line in iter(process.stdout.readline, ""):
                if not line:
                    break
                sys.stdout.write(line)
                sys.stdout.flush()  # Ensure immediate output
                output_lines.append(line)

            process.wait()
            success = process.returncode == 0
            output = "".join(output_lines)
        else:
            # Capture output for multiprocessing
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
                env=env,
                cwd=str(PROJECT_ROOT),
            )
            success = result.returncode == 0
            output = result.stdout + result.stderr

        return (config.song_name, success, output)
    except Exception as e:
        return (config.song_name, False, f"Error: {str(e)}")


def run_method(
    method_name: str,
    commands: list[CommandConfig],
    concurrency: int,
    verbose: bool = False,
    dry_run: bool = False,
) -> dict[str, tuple[bool, str]]:
    """Run all commands for a method with multiprocessing."""
    print(f"\n{'='*60}")
    print(f"Running method: {method_name}")
    print(f"Total commands: {len(commands)}")
    if not dry_run:
        print(f"Concurrency: {concurrency}")
    else:
        print("DRY RUN MODE - No commands will be executed")
    print(f"{'='*60}\n")

    if dry_run:
        # Just print commands
        for i, cmd in enumerate(commands, 1):
            cmd_str = " ".join(cmd.to_command())
            print(f"[{i}/{len(commands)}] {cmd.song_name}")
            print(f"  {cmd_str}\n")
        # Return fake success results
        return {cmd.song_name: (True, "dry-run") for cmd in commands}

    results = {}

    # When concurrency is 1, run sequentially with streaming output
    if concurrency == 1:
        for i, cmd in enumerate(commands, 1):
            print(f"\n[{i}/{len(commands)}] Running {cmd.song_name}...")
            print("-" * 60)
            song_name, success, output = run_command(cmd, stream_output=True)
            status = "✓" if success else "✗"
            print("-" * 60)
            print(f"[{i}/{len(commands)}] {status} {cmd.song_name}")

            if not success:
                # Always print errors, but show more detail if verbose
                error_limit = len(output) if verbose else 500
                print(f"  Error output:\n{output[:error_limit]}\n")

            results[song_name] = (success, output)
    else:
        # Use multiprocessing for concurrency > 1
        with ProcessPoolExecutor(max_workers=concurrency) as executor:
            # Submit all tasks
            future_to_config = {
                executor.submit(run_command, cmd, stream_output=False): cmd
                for cmd in commands
            }

            # Process completed tasks
            completed = 0
            for future in as_completed(future_to_config):
                completed += 1
                config = future_to_config[future]
                song_name, success, output = future.result()

                status = "✓" if success else "✗"
                print(f"[{completed}/{len(commands)}] {status} {config.song_name}")

                if not success:
                    # Always print errors, but show more detail if verbose
                    error_limit = len(output) if verbose else 500
                    print(f"  Error output:\n{output[:error_limit]}\n")

                results[song_name] = (success, output)

    return results


def parse_song_overrides(override_str: str) -> dict[str, dict[str, Any]]:
    """
    Parse song-specific overrides from string format.
    Format: "song1:key1=value1,key2=value2;song2:key1=value1"
    """
    if not override_str:
        return {}

    overrides = {}
    for song_override in override_str.split(";"):
        if ":" not in song_override:
            continue
        song_name, args_str = song_override.split(":", 1)
        song_name = song_name.strip()
        overrides[song_name] = {}

        for arg_pair in args_str.split(","):
            if "=" not in arg_pair:
                continue
            key, value = arg_pair.split("=", 1)
            key = key.strip().replace("_", "-")
            value = value.strip()

            # Try to convert to appropriate type
            if value.lower() == "true":
                value = True
            elif value.lower() == "false":
                value = False
            else:
                try:
                    if "." in value:
                        value = float(value)
                    else:
                        value = int(value)
                except ValueError:
                    pass  # Keep as string

            overrides[song_name][key] = value

    return overrides


def main():
    parser = argparse.ArgumentParser(
        description="Run all synthesis methods for all songs in a dataset with multiprocessing."
    )
    parser.add_argument(
        "dataset_name",
        help="Dataset name (e.g., 'ds2') under datasets/ directory.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["random", "text_video", "fused", "all"],
        default=["all"],
        help="Methods to run. Default: all",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Number of concurrent processes. Default: 4",
    )
    parser.add_argument(
        "--songs",
        nargs="+",
        help="Specific songs to run (default: all songs in dataset).",
    )
    parser.add_argument(
        "--fused-overrides",
        type=str,
        help=(
            "Song-specific fused method overrides. "
            "Format: 'song1:key1=value1,key2=value2;song2:key1=value1'"
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed error output for failed commands.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands that would be run without executing them.",
    )

    args = parser.parse_args()

    # Discover songs
    all_songs = discover_songs(args.dataset_name)
    if args.songs:
        songs = [s for s in all_songs if s in args.songs]
        if not songs:
            print(
                f"Error: None of the specified songs found. Available: {', '.join(all_songs)}"
            )
            sys.exit(1)
    else:
        songs = all_songs

    print(f"Dataset: {args.dataset_name}")
    print(f"Songs: {', '.join(songs)}")
    print(f"Total songs: {len(songs)}\n")

    # Determine which methods to run
    methods_to_run = []
    if "all" in args.methods:
        methods_to_run = ["random", "text_video", "fused"]
    else:
        methods_to_run = args.methods

    # Parse fused overrides
    fused_overrides = {}
    if args.fused_overrides:
        fused_overrides = parse_song_overrides(args.fused_overrides)

    # Run each method
    all_results = {}
    for method_name in methods_to_run:
        if method_name == "random":
            commands = get_random_method_commands(args.dataset_name, songs)
        elif method_name == "text_video":
            commands = get_text_video_method_commands(args.dataset_name, songs)
        elif method_name == "fused":
            commands = get_fused_method_commands(
                args.dataset_name, songs, song_overrides=fused_overrides
            )
        else:
            continue

        results = run_method(
            method_name, commands, args.concurrency, args.verbose, args.dry_run
        )
        all_results[method_name] = results

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}\n")

    for method_name, results in all_results.items():
        total = len(results)
        successful = sum(1 for success, _ in results.values() if success)
        failed = total - successful

        print(f"{method_name}:")
        print(f"  Total: {total}")
        print(f"  Successful: {successful}")
        print(f"  Failed: {failed}")

        if failed > 0:
            failed_songs = [
                song for song, (success, _) in results.items() if not success
            ]
            print(f"  Failed songs: {', '.join(failed_songs)}")
        print()

    # Exit with error code if any failed
    total_failed = sum(
        sum(1 for success, _ in results.values() if not success)
        for results in all_results.values()
    )

    if total_failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
