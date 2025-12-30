#!/usr/bin/env python3
"""
Plot evaluation results comparing different synthesis methods.

This script analyzes JSON result files from the results/ directory and creates
visualizations showing winning rates between methods.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def load_results(results_dir: Path) -> list[dict[str, Any]]:
    """Load all JSON result files from the results directory."""
    results = []
    for json_file in results_dir.glob("*.json"):
        try:
            with open(json_file, "r") as f:
                data = json.load(f)
                results.append(data)
        except Exception as e:
            print(f"Warning: Failed to load {json_file}: {e}")
    return results


def analyze_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Analyze results and compute statistics."""
    stats = {
        "overall": defaultdict(int),
        "by_user": defaultdict(lambda: defaultdict(int)),
        "by_song": defaultdict(lambda: defaultdict(int)),
        "by_method_pair": defaultdict(lambda: defaultdict(int)),
        "total_comparisons": 0,
        "ties": 0,
        "line_positions": defaultdict(lambda: defaultdict(int)),
        # Track wins/losses/ties per method per song for segmented bar charts
        "method_outcomes_by_song": defaultdict(lambda: defaultdict(lambda: defaultdict(int))),
    }

    for result in results:
        metadata = result.get("metadata", {})
        username = metadata.get("username", "unknown")
        song = metadata.get("song", "unknown")
        methods = metadata.get("methods", [])

        comparisons = result.get("comparisons", [])
        for comp in comparisons:
            winner = comp.get("winner")
            decision = comp.get("decision")
            line_index = comp.get("line_index")
            line_position = comp.get("line_position")
            
            # Get the methods being compared
            left_method = comp.get("left", {}).get("method")
            right_method = comp.get("right", {}).get("method")

            # Check if this is a tie
            is_tie = winner is None or decision == "tie"

            if is_tie:
                stats["ties"] += 1
                stats["total_comparisons"] += 1
                stats["overall"]["tie"] += 1
                stats["by_user"][username]["tie"] += 1
                stats["by_song"][song]["tie"] += 1

                # Track outcomes for segmented bar chart
                if left_method and right_method:
                    stats["method_outcomes_by_song"][song][left_method]["tie"] += 1
                    stats["method_outcomes_by_song"][song][right_method]["tie"] += 1

                # Track by method pair
                if len(methods) == 2:
                    method_pair = tuple(sorted(methods))
                    stats["by_method_pair"][method_pair]["tie"] += 1

                # Track by line position
                if line_position is not None:
                    stats["line_positions"][line_position]["tie"] += 1
            else:
                stats["total_comparisons"] += 1
                stats["overall"][winner] += 1
                stats["by_user"][username][winner] += 1
                stats["by_song"][song][winner] += 1

                # Track outcomes for segmented bar chart
                if left_method and right_method:
                    # Winner gets a win
                    stats["method_outcomes_by_song"][song][winner]["win"] += 1
                    # Loser gets a loss
                    loser = right_method if winner == left_method else left_method
                    stats["method_outcomes_by_song"][song][loser]["loss"] += 1

                # Track by method pair
                if len(methods) == 2:
                    method_pair = tuple(sorted(methods))
                    stats["by_method_pair"][method_pair][winner] += 1

                # Track by line position
                if line_position is not None:
                    stats["line_positions"][line_position][winner] += 1

    return stats


def plot_overall_winning_rates(stats: dict[str, Any], output_dir: Path):
    """Plot overall winning rates for all methods including ties."""
    overall = stats["overall"]
    if not overall:
        print("No valid comparisons found for overall plot")
        return

    # Sort methods, putting "tie" last
    methods = sorted([m for m in overall.keys() if m != "tie"]) + (["tie"] if "tie" in overall else [])
    wins = [overall[m] for m in methods]
    total = sum(wins)
    win_rates = [w / total * 100 for w in wins]

    # Color scheme: methods get colors, tie gets gray
    colors = ["#2ecc71", "#e74c3c", "#3498db", "#f39c12", "#95a5a6"]
    bar_colors = []
    for i, method in enumerate(methods):
        if method == "tie":
            bar_colors.append("#95a5a6")  # Gray for ties
        else:
            bar_colors.append(colors[i % (len(colors) - 1)])

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(methods, win_rates, color=bar_colors)

    # Add value labels on bars
    for bar, rate, win in zip(bars, win_rates, wins):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{rate:.1f}%\n({win}/{total})",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    ax.set_ylabel("Rate (%)", fontsize=12)
    ax.set_xlabel("Method / Outcome", fontsize=12)
    ax.set_title(f"Overall Results Distribution\n(Total comparisons: {total})", fontsize=14, fontweight="bold")
    ax.set_ylim(0, max(win_rates) * 1.2 if win_rates else 100)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    plt.tight_layout()
    plt.savefig(output_dir / "overall_winning_rates.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir / 'overall_winning_rates.png'}")


def plot_by_user(stats: dict[str, Any], output_dir: Path):
    """Plot winning rates broken down by user including ties."""
    by_user = stats["by_user"]
    if not by_user:
        print("No user-specific data found")
        return

    users = sorted(by_user.keys())
    all_methods = set()
    for user_stats in by_user.values():
        all_methods.update(user_stats.keys())
    # Sort methods, putting "tie" last
    methods = sorted([m for m in all_methods if m != "tie"]) + (["tie"] if "tie" in all_methods else [])

    if not methods:
        print("No methods found in user data")
        return

    # Prepare data for grouped bar chart
    x = np.arange(len(users))
    width = 0.35
    colors = ["#2ecc71", "#e74c3c", "#3498db", "#f39c12", "#95a5a6"]

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, method in enumerate(methods):
        win_rates = []
        win_counts = []
        totals = []
        for user in users:
            wins = by_user[user].get(method, 0)
            total = sum(by_user[user].values())
            win_rates.append(wins / total * 100 if total > 0 else 0)
            win_counts.append(wins)
            totals.append(total)

        offset = (i - len(methods) / 2 + 0.5) * width / len(methods)
        method_color = "#95a5a6" if method == "tie" else colors[i % (len(colors) - 1)]
        bars = ax.bar(x + offset, win_rates, width / len(methods), label=method, color=method_color)

        # Add value labels
        for bar, rate, win, tot in zip(bars, win_rates, win_counts, totals):
            if win > 0:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{rate:.1f}%\n({win}/{tot})",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

    ax.set_ylabel("Rate (%)", fontsize=12)
    ax.set_xlabel("User", fontsize=12)
    ax.set_title("Results Distribution by User", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(users)
    ax.legend()
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(output_dir / "winning_rates_by_user.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir / 'winning_rates_by_user.png'}")


def plot_by_song(stats: dict[str, Any], output_dir: Path):
    """Plot winning rates broken down by song including ties."""
    by_song = stats["by_song"]
    if not by_song:
        print("No song-specific data found")
        return

    songs = sorted(by_song.keys())
    all_methods = set()
    for song_stats in by_song.values():
        all_methods.update(song_stats.keys())
    # Sort methods, putting "tie" last
    methods = sorted([m for m in all_methods if m != "tie"]) + (["tie"] if "tie" in all_methods else [])

    if not methods:
        print("No methods found in song data")
        return

    # Prepare data for grouped bar chart
    x = np.arange(len(songs))
    width = 0.35
    colors = ["#2ecc71", "#e74c3c", "#3498db", "#f39c12", "#95a5a6"]

    fig, ax = plt.subplots(figsize=(max(12, len(songs) * 2), 6))

    for i, method in enumerate(methods):
        win_rates = []
        win_counts = []
        totals = []
        for song in songs:
            wins = by_song[song].get(method, 0)
            total = sum(by_song[song].values())
            win_rates.append(wins / total * 100 if total > 0 else 0)
            win_counts.append(wins)
            totals.append(total)

        offset = (i - len(methods) / 2 + 0.5) * width / len(methods)
        method_color = "#95a5a6" if method == "tie" else colors[i % (len(colors) - 1)]
        bars = ax.bar(x + offset, win_rates, width / len(methods), label=method, color=method_color)

        # Add value labels
        for bar, rate, win, tot in zip(bars, win_rates, win_counts, totals):
            if win > 0:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{rate:.1f}%\n({win}/{tot})",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

    ax.set_ylabel("Rate (%)", fontsize=12)
    ax.set_xlabel("Song", fontsize=12)
    ax.set_title("Results Distribution by Song", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(songs, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(output_dir / "winning_rates_by_song.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir / 'winning_rates_by_song.png'}")


def plot_cumulative_winning_rates(stats: dict[str, Any], output_dir: Path):
    """Plot cumulative winning rates over line positions including ties."""
    line_positions = stats["line_positions"]
    if not line_positions:
        print("No line position data found")
        return

    positions = sorted(line_positions.keys())
    all_methods = set()
    for pos_stats in line_positions.values():
        all_methods.update(pos_stats.keys())
    # Sort methods, putting "tie" last
    methods = sorted([m for m in all_methods if m != "tie"]) + (["tie"] if "tie" in all_methods else [])

    if not methods:
        print("No methods found in line position data")
        return

    # Calculate cumulative wins
    cumulative_wins = {method: [] for method in methods}
    cumulative_totals = []

    for pos in positions:
        total = sum(line_positions[pos].values())
        cumulative_totals.append(total)
        for method in methods:
            wins = line_positions[pos].get(method, 0)
            prev_wins = cumulative_wins[method][-1] if cumulative_wins[method] else 0
            cumulative_wins[method].append(prev_wins + wins)

    # Calculate cumulative rates
    cumulative_rates = {method: [] for method in methods}
    running_total = 0
    for i, pos in enumerate(positions):
        running_total += cumulative_totals[i]
        for method in methods:
            rate = cumulative_wins[method][i] / running_total * 100 if running_total > 0 else 0
            cumulative_rates[method].append(rate)

    # Plot
    fig, ax = plt.subplots(figsize=(14, 6))
    colors = ["#2ecc71", "#e74c3c", "#3498db", "#f39c12", "#95a5a6"]

    for i, method in enumerate(methods):
        method_color = "#95a5a6" if method == "tie" else colors[i % (len(colors) - 1)]
        linestyle = "--" if method == "tie" else "-"
        ax.plot(
            positions,
            cumulative_rates[method],
            marker="o",
            label=method,
            linewidth=2,
            color=method_color,
            linestyle=linestyle,
        )

    ax.set_xlabel("Line Position", fontsize=12)
    ax.set_ylabel("Cumulative Rate (%)", fontsize=12)
    ax.set_title("Cumulative Results Distribution Over Line Positions", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(output_dir / "cumulative_winning_rates.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir / 'cumulative_winning_rates.png'}")


def plot_segmented_by_song(stats: dict[str, Any], output_dir: Path):
    """Plot segmented bar charts showing wins/losses/ties for each method per song."""
    method_outcomes = stats["method_outcomes_by_song"]
    if not method_outcomes:
        print("No method outcome data found for segmented plots")
        return

    songs = sorted(method_outcomes.keys())
    
    for song in songs:
        song_outcomes = method_outcomes[song]
        methods = sorted(song_outcomes.keys())
        
        if not methods:
            continue
        
        # Prepare data for stacked bar chart
        wins = []
        losses = []
        ties = []
        
        for method in methods:
            outcomes = song_outcomes[method]
            wins.append(outcomes.get("win", 0))
            losses.append(outcomes.get("loss", 0))
            ties.append(outcomes.get("tie", 0))
        
        # Create stacked bar chart
        fig, ax = plt.subplots(figsize=(max(10, len(methods) * 2), 6))
        
        x = np.arange(len(methods))
        width = 0.6
        
        # Colors: green for wins, red for losses, gray for ties
        colors = {"win": "#2ecc71", "loss": "#e74c3c", "tie": "#95a5a6"}
        
        # Create stacked bars
        bars1 = ax.bar(x, wins, width, label="Win", color=colors["win"])
        bars2 = ax.bar(x, losses, width, bottom=wins, label="Loss", color=colors["loss"])
        bars3 = ax.bar(x, ties, width, bottom=np.array(wins) + np.array(losses), label="Tie", color=colors["tie"])
        
        # Add value labels on each segment
        for i, method in enumerate(methods):
            total = wins[i] + losses[i] + ties[i]
            if total == 0:
                continue
            
            # Win segment
            if wins[i] > 0:
                win_y = wins[i] / 2
                ax.text(
                    x[i],
                    win_y,
                    f"{wins[i]}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    fontweight="bold",
                    color="white",
                )
            
            # Loss segment
            if losses[i] > 0:
                loss_y = wins[i] + losses[i] / 2
                ax.text(
                    x[i],
                    loss_y,
                    f"{losses[i]}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    fontweight="bold",
                    color="white",
                )
            
            # Tie segment
            if ties[i] > 0:
                tie_y = wins[i] + losses[i] + ties[i] / 2
                ax.text(
                    x[i],
                    tie_y,
                    f"{ties[i]}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    fontweight="bold",
                    color="white",
                )
            
            # Total on top
            ax.text(
                x[i],
                total,
                f"Total: {total}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )
        
        ax.set_ylabel("Number of Comparisons", fontsize=12)
        ax.set_xlabel("Method", fontsize=12)
        ax.set_title(f"Results Distribution by Method - {song}\n(Win / Loss / Tie)", fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        ax.legend(loc="upper right")
        ax.grid(axis="y", alpha=0.3, linestyle="--", which="major")
        
        # Set y-axis to show a bit above the max total
        max_total = max([w + l + t for w, l, t in zip(wins, losses, ties)], default=1)
        ax.set_ylim(0, max_total * 1.15)
        
        plt.tight_layout()
        filename = f"segmented_bar_{song}.png"
        plt.savefig(output_dir / filename, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved: {output_dir / filename}")


def print_summary(stats: dict[str, Any]):
    """Print a text summary of the statistics."""
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS SUMMARY")
    print("=" * 60)

    print(f"\nTotal Comparisons: {stats['total_comparisons']}")
    print(f"Ties: {stats['ties']}")

    print("\n--- Overall Results Distribution ---")
    overall = stats["overall"]
    total = sum(overall.values())
    # Sort methods, putting "tie" last
    methods_sorted = sorted([m for m in overall.keys() if m != "tie"]) + (["tie"] if "tie" in overall else [])
    for method in methods_sorted:
        wins = overall[method]
        rate = wins / total * 100 if total > 0 else 0
        print(f"  {method}: {wins}/{total} ({rate:.2f}%)")

    print("\n--- By User ---")
    for user in sorted(stats["by_user"].keys()):
        user_stats = stats["by_user"][user]
        user_total = sum(user_stats.values())
        print(f"\n  {user}:")
        user_methods_sorted = sorted([m for m in user_stats.keys() if m != "tie"]) + (
            ["tie"] if "tie" in user_stats else []
        )
        for method in user_methods_sorted:
            wins = user_stats[method]
            rate = wins / user_total * 100 if user_total > 0 else 0
            print(f"    {method}: {wins}/{user_total} ({rate:.2f}%)")

    print("\n--- By Song ---")
    for song in sorted(stats["by_song"].keys()):
        song_stats = stats["by_song"][song]
        song_total = sum(song_stats.values())
        print(f"\n  {song}:")
        song_methods_sorted = sorted([m for m in song_stats.keys() if m != "tie"]) + (
            ["tie"] if "tie" in song_stats else []
        )
        for method in song_methods_sorted:
            wins = song_stats[method]
            rate = wins / song_total * 100 if song_total > 0 else 0
            print(f"    {method}: {wins}/{song_total} ({rate:.2f}%)")

    print("\n" + "=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Plot evaluation results comparing synthesis methods")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).parent.parent / "results",
        help="Directory containing result JSON files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent.parent / "results" / "plots",
        help="Directory to save plot images",
    )
    parser.add_argument("--no-plots", action="store_true", help="Only print summary, don't generate plots")

    args = parser.parse_args()

    # Load results
    print(f"Loading results from {args.results_dir}...")
    results = load_results(args.results_dir)
    print(f"Loaded {len(results)} result file(s)")

    if not results:
        print("No result files found!")
        return

    # Analyze
    stats = analyze_results(results)

    # Print summary
    print_summary(stats)

    # Generate plots
    if not args.no_plots:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nGenerating plots in {args.output_dir}...")

        plot_overall_winning_rates(stats, args.output_dir)
        plot_by_user(stats, args.output_dir)
        plot_by_song(stats, args.output_dir)
        plot_cumulative_winning_rates(stats, args.output_dir)
        plot_segmented_by_song(stats, args.output_dir)

        print("\nAll plots generated successfully!")


if __name__ == "__main__":
    main()

