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


# Mapping from internal method names to display names
METHOD_DISPLAY_NAMES = {
    "fused_rank": "Multi-Stream Synergy Retrieval",
    "random": "Random",
    "text_video": "Naive",
}


def get_method_display_name(method: str) -> str:
    """Get the display name for a method, or return the method name if not found."""
    return METHOD_DISPLAY_NAMES.get(method, method)


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
        # Track fused_rank outcomes by method pair per song for horizontal segmented bar chart
        "fused_rank_pair_outcomes": defaultdict(lambda: defaultdict(lambda: defaultdict(int))),
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

            # Track fused_rank comparisons with random or text_video
            if left_method and right_method:
                fused_method = "fused_rank"
                other_methods = ["random", "text_video"]
                
                # Check if fused_rank is being compared with random or text_video
                if left_method == fused_method and right_method in other_methods:
                    pair_key = f"{fused_method}_vs_{right_method}"
                    if is_tie:
                        stats["fused_rank_pair_outcomes"][song][pair_key]["tie"] += 1
                    elif winner == fused_method:
                        stats["fused_rank_pair_outcomes"][song][pair_key]["win"] += 1
                    else:
                        stats["fused_rank_pair_outcomes"][song][pair_key]["loss"] += 1
                elif right_method == fused_method and left_method in other_methods:
                    pair_key = f"{fused_method}_vs_{left_method}"
                    if is_tie:
                        stats["fused_rank_pair_outcomes"][song][pair_key]["tie"] += 1
                    elif winner == fused_method:
                        stats["fused_rank_pair_outcomes"][song][pair_key]["win"] += 1
                    else:
                        stats["fused_rank_pair_outcomes"][song][pair_key]["loss"] += 1

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
    """Plot overall winning rates for all methods (excluding ties)."""
    overall = stats["overall"]
    if not overall:
        print("No valid comparisons found for overall plot")
        return

    # Filter to only show actual methods (exclude "tie" which is an outcome, not a method)
    # Expected methods: fused_rank, random, text_video
    expected_methods = ["fused_rank", "random", "text_video"]
    methods = [m for m in expected_methods if m in overall]
    
    if not methods:
        print("No valid methods found in overall stats")
        return
    
    # Sort methods in expected order
    methods = sorted(methods, key=lambda x: expected_methods.index(x) if x in expected_methods else len(expected_methods))
    wins = [overall[m] for m in methods]
    # Total includes all comparisons (wins + ties) for proper percentage calculation
    total = sum(overall.values())
    win_rates = [w / total * 100 for w in wins]

    # Get display names for methods
    method_display_names = [get_method_display_name(m) for m in methods]

    # Color scheme: one color per method
    colors = ["#2ecc71", "#e74c3c", "#3498db"]
    bar_colors = [colors[i % len(colors)] for i in range(len(methods))]

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(method_display_names, win_rates, color=bar_colors)

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

    ax.set_ylabel("Rate (%)", fontsize=14)
    ax.set_xlabel("Method", fontsize=14)
    ax.set_title(f"Win Rate Distribution\n(Total preference entries: {total})", fontsize=12, fontweight="bold")
    ax.set_ylim(0, max(win_rates) * 1.2 if win_rates else 100)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    plt.tight_layout()
    plt.savefig(output_dir / "overall_win_rates.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir / 'overall_win_rates.png'}")


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

    fig, ax = plt.subplots(figsize=(8, 6))

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
        method_label = method if method == "tie" else get_method_display_name(method)
        bars = ax.bar(x + offset, win_rates, width / len(methods), label=method_label, color=method_color)

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

    ax.set_ylabel("Rate (%)", fontsize=14)
    ax.set_xlabel("User", fontsize=14)
    ax.set_title("Results Distribution by User", fontsize=12, fontweight="bold")
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

    fig, ax = plt.subplots(figsize=(8, 6))

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
        method_label = method if method == "tie" else get_method_display_name(method)
        bars = ax.bar(x + offset, win_rates, width / len(methods), label=method_label, color=method_color)

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

    ax.set_ylabel("Rate (%)", fontsize=14)
    ax.set_xlabel("Song", fontsize=14)
    ax.set_title("Results Distribution by Song", fontsize=12, fontweight="bold")
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
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#2ecc71", "#e74c3c", "#3498db", "#f39c12", "#95a5a6"]

    for i, method in enumerate(methods):
        method_color = "#95a5a6" if method == "tie" else colors[i % (len(colors) - 1)]
        linestyle = "--" if method == "tie" else "-"
        method_label = method if method == "tie" else get_method_display_name(method)
        ax.plot(
            positions,
            cumulative_rates[method],
            marker="o",
            label=method_label,
            linewidth=2,
            color=method_color,
            linestyle=linestyle,
        )

    ax.set_xlabel("Line Position", fontsize=14)
    ax.set_ylabel("Cumulative Rate (%)", fontsize=14)
    ax.set_title("Cumulative Results Distribution Over Line Positions", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(output_dir / "cumulative_winning_rates.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir / 'cumulative_winning_rates.png'}")


def plot_segmented_by_song(stats: dict[str, Any], output_dir: Path):
    """Plot horizontal segmented bar chart showing fused_rank vs random and fused_rank vs text_video for all songs."""
    fused_rank_outcomes = stats["fused_rank_pair_outcomes"]
    if not fused_rank_outcomes:
        print("No fused_rank comparison data found for segmented plots")
        return

    songs = sorted(fused_rank_outcomes.keys())
    if not songs:
        print("No songs found with fused_rank comparisons")
        return
    
    # Define the two comparison pairs we want to plot
    pair_labels = ["fused_rank_vs_random", "fused_rank_vs_text_video"]
    pair_display_names = [
        f"{get_method_display_name('fused_rank')} vs {get_method_display_name('random')}",
        f"{get_method_display_name('fused_rank')} vs {get_method_display_name('text_video')}"
    ]
    
    # Prepare data for horizontal stacked bar chart
    # Structure: [song][pair][outcome] = count
    song_data = {}
    for song in songs:
        song_data[song] = {}
        for pair_label in pair_labels:
            outcomes = fused_rank_outcomes[song].get(pair_label, {})
            song_data[song][pair_label] = {
                "win": outcomes.get("win", 0),
                "loss": outcomes.get("loss", 0),
                "tie": outcomes.get("tie", 0),
            }
    
    # Create horizontal bar chart
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Use double spacing for each song (one for each bar)
    y_pos = np.arange(len(songs)) * 2
    bar_height = 0.6
    spacing = 0.2  # Space between the two bars for each song
    
    # Single color scheme for all bars (distinguished by position: top/bottom)
    colors = {"win": "#2ecc71", "loss": "#e74c3c", "tie": "#95a5a6"}
    
    # Track legend labels
    legend_added = {"win": False, "loss": False, "tie": False}
    
    # Plot bars for each song
    for i, song in enumerate(songs):
        y_base = y_pos[i]
        
        # Plot two bars for each song (fused vs random, fused vs text_video)
        # j=0: top bar (fused vs random), j=1: bottom bar (fused vs text_video)
        for j, (pair_label, pair_display) in enumerate(zip(pair_labels, pair_display_names)):
            outcomes = song_data[song][pair_label]
            wins = outcomes["win"]
            losses = outcomes["loss"]
            ties = outcomes["tie"]
            total = wins + losses + ties
            
            if total == 0:
                continue
            
            # Calculate percentages
            win_pct = (wins / total * 100) if total > 0 else 0
            loss_pct = (losses / total * 100) if total > 0 else 0
            tie_pct = (ties / total * 100) if total > 0 else 0
            
            # Calculate bar position (offset for second bar)
            # j=0: top bar (positive offset), j=1: bottom bar (negative offset)
            y_offset = (j - 0.5) * (bar_height + spacing)
            y_center = y_base + y_offset
            
            # Create horizontal stacked bars normalized to 100%
            # Order: win (left), loss (middle), tie (right)
            left_edge = 0
            if win_pct > 0:
                label = "Win" if not legend_added["win"] else ""
                ax.barh(y_center, win_pct, bar_height, left=left_edge, color=colors["win"], label=label)
                legend_added["win"] = True
                # Add label with percentage
                if win_pct >= 2:  # Only show label if segment is large enough (2%)
                    ax.text(left_edge + win_pct / 2, y_center, f"{win_pct:.1f}%", ha="center", va="center", 
                           fontsize=8, fontweight="bold", color="white")
                left_edge += win_pct
            
            if loss_pct > 0:
                label = "Loss" if not legend_added["loss"] else ""
                ax.barh(y_center, loss_pct, bar_height, left=left_edge, color=colors["loss"], label=label)
                legend_added["loss"] = True
                # Add label with percentage
                if loss_pct >= 2:  # Only show label if segment is large enough (2%)
                    ax.text(left_edge + loss_pct / 2, y_center, f"{loss_pct:.1f}%", ha="center", va="center",
                           fontsize=8, fontweight="bold", color="white")
                left_edge += loss_pct
            
            if tie_pct > 0:
                label = "Tie" if not legend_added["tie"] else ""
                ax.barh(y_center, tie_pct, bar_height, left=left_edge, color=colors["tie"], label=label)
                legend_added["tie"] = True
                # Add label with percentage
                if tie_pct >= 2:  # Only show label if segment is large enough (2%)
                    ax.text(left_edge + tie_pct / 2, y_center, f"{tie_pct:.1f}%", ha="center", va="center",
                           fontsize=8, fontweight="bold", color="white")
                left_edge += tie_pct
    
    # Set labels and title
    ax.set_xlabel("Percentage (%)", fontsize=14)
    ax.set_ylabel("Song", fontsize=14)
    ax.set_title(f"Multi-Stream Synergy Retrieval Performance: Win / Loss / Tie (Normalized)\n(Top bar: vs. {get_method_display_name('random')} | Bottom bar: vs. {get_method_display_name('text_video')})", 
                 fontsize=12, fontweight="bold")
    
    # Set y-axis ticks and labels (one per song, positioned between the two bars)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(songs)
    
    # Add legend
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    
    # Set x-axis limits (bars start at 0, with padding for labels)
    ax.set_xlim(0, 120)
    
    plt.tight_layout()
    filename = "win_loss_by_song.png"
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
        method_display = method if method == "tie" else get_method_display_name(method)
        print(f"  {method_display}: {wins}/{total} ({rate:.2f}%)")

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
            method_display = method if method == "tie" else get_method_display_name(method)
            print(f"    {method_display}: {wins}/{user_total} ({rate:.2f}%)")

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
            method_display = method if method == "tie" else get_method_display_name(method)
            print(f"    {method_display}: {wins}/{song_total} ({rate:.2f}%)")

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

