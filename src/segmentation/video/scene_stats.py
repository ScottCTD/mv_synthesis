import statistics
from typing import List, Tuple


def _percentile(values: List[float], percentile: float) -> float:
    if not values:
        return 0.0
    k = (len(values) - 1) * percentile
    f = int(k)
    c = min(f + 1, len(values) - 1)
    if f == c:
        return values[f]
    d0 = values[f] * (c - k)
    d1 = values[c] * (k - f)
    return d0 + d1


def _duration_histogram(durations: List[float], bins: int = 20) -> List[Tuple[float, float, int]]:
    if not durations:
        return []
    max_duration = max(durations)
    bin_size = max_duration / bins if max_duration else 0
    histogram: List[Tuple[float, float, int]] = []
    for idx in range(bins):
        start = idx * bin_size
        end = start + bin_size
        count = sum(
            1 for d in durations if (d >= start and (d < end or (idx == bins - 1 and d <= end)))
        )
        histogram.append((start, end, count))
    return histogram


def _format_seconds(value: float) -> str:
    minutes, seconds = divmod(int(round(value)), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m {seconds}s"


def build_duration_report(durations: List[float]) -> str:
    if not durations:
        return "No scenes to report."

    sorted_durations = sorted(durations)
    total_secs = sum(durations)
    lines = []
    lines.append("============================================================")
    lines.append(f"Total Scenes: {len(durations)}")
    lines.append(f"Total Duration: {_format_seconds(total_secs)}")
    lines.append(f"Total Duration (seconds): {total_secs:.3f}")
    lines.append("============================================================")
    lines.append("\nDuration Statistics:")
    lines.append("------------------------------------------------------------")
    lines.append(f"Mean:   {statistics.fmean(sorted_durations):.3f}s")
    lines.append(f"Median: {statistics.median(sorted_durations):.3f}s")
    lines.append(f"Min:    {min(sorted_durations):.3f}s")
    lines.append(f"Max:    {max(sorted_durations):.3f}s")
    if len(sorted_durations) > 1:
        lines.append(f"StdDev: {statistics.stdev(sorted_durations):.3f}s")
    else:
        lines.append("StdDev: N/A (only one scene)")
    lines.append("\nPercentiles:")
    for pct in (0.25, 0.75, 0.90, 0.95):
        pct_value = _percentile(sorted_durations, pct)
        lines.append(f"{int(pct * 100):>3}th: {pct_value:.3f}s")
    lines.append("============================================================")
    lines.append("\nDuration Distribution:")
    histogram = _duration_histogram(sorted_durations)
    if not histogram:
        lines.append("No histogram available.")
        return "\n".join(lines)
    max_count = max(count for _, _, count in histogram) or 1
    for start, end, count in histogram:
        bar = "#" * int(40 * (count / max_count))
        lines.append(f"{start:6.2f}s - {end:6.2f}s: {count:4d} {bar}")
    lines.append("============================================================\n")
    return "\n".join(lines)
