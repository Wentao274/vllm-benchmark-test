import os
import re
import glob
import yaml
from pathlib import Path
from datetime import datetime
from collections import defaultdict

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None
    cm = None
    print("matplotlib not available, skipping chart generation")


TEST_SUITES = ["test_01"]

RUN_ID = "01"

MODEL_NAME = "MiniMax-M2.5"


def load_models_scenarios(config_path="config/models_scenarios.yaml"):
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


CHIP_BASE_PATHS = {}


def load_chip_base_paths():
    paths = {}
    scenarios = load_models_scenarios()
    models = scenarios.get("models", {})
    chip_key_map = {
        "Hygon_BW1000": "hygon_bw1000",
        "Kunlun_P800": "kunlun_p800",
        "NVIDIA_H100": "nvidia_h100",
    }
    for chip_name, chip_key in chip_key_map.items():
        chip_models = models.get(chip_key, [])
        if chip_models:
            model_info = chip_models[0]
            model_path = model_info.get("model_path", "")
            if model_path:
                model_name = Path(model_path).name
                paths[chip_name] = f"reports/{chip_key}/benchmark/{model_name}"
    return paths


CHIP_BASE_PATHS = load_chip_base_paths()


def get_chip_configs(chip_name, test_suite, run_id):
    base_path = CHIP_BASE_PATHS.get(chip_name, "")
    full_path = f"{base_path}/{test_suite}/{run_id}"

    if not os.path.exists(full_path):
        print(f"Error: No data found at {full_path}")
        return []

    return [
        {
            "name": chip_name,
            "base_path": full_path,
        }
    ]


def load_chip_config(config_path="config/chip_conf.yaml"):
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


def load_vllm_config(config_path="config/model_deployment.yaml"):
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


def parse_benchmark_log(log_file):
    with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    lines = content.split("\n")
    metrics = {}

    in_results = False
    for line in lines:
        if "=========== Serving Benchmark Result" in line:
            in_results = True
            continue
        if in_results and line.strip().startswith("==========="):
            break
        if in_results:
            match = re.match(r"(.+?):\s+(.+)$", line.strip())
            if match:
                key = match.group(1).strip()
                value = match.group(2).strip()
                key_normalized = key.replace("Token ", "token ")
                metrics[key] = value
                if key != key_normalized:
                    metrics[key_normalized] = value

    return metrics


def extract_concurrency_from_dir(dir_name):
    match = re.match(r"^(\d+)-", dir_name)
    if match:
        return match.group(1)
    return None


def get_all_concurrencies(chip_config):
    concurrency_set = set()
    base_path = chip_config["base_path"]

    if not os.path.exists(base_path):
        return []

    for item in os.listdir(base_path):
        item_path = os.path.join(base_path, item)
        if os.path.isdir(item_path):
            conc = extract_concurrency_from_dir(item)
            if conc:
                concurrency_set.add(conc)

    return sorted(concurrency_set, key=lambda x: int(x))


def get_chip_metrics(chip_config, concurrency):
    base_path = chip_config["base_path"]
    chip_name = chip_config["name"]

    dir_pattern = os.path.join(base_path, f"{concurrency}-*")
    matching_dirs = glob.glob(dir_pattern)

    if not matching_dirs:
        return None

    log_pattern = os.path.join(matching_dirs[0], "*.log")
    log_files = glob.glob(log_pattern)

    if not log_files:
        return None

    metrics = parse_benchmark_log(log_files[0])
    return metrics


def generate_comparison_csv(chip_data, concurrencies, output_dir, chip_name):
    metric_names = [
        ("[Serving Benchmark Result]", ""),
        ("Successful requests", "Successful requests"),
        ("Failed requests", "Failed requests"),
        ("Benchmark duration (s)", "Benchmark duration (s)"),
        ("Total input tokens", "Total input tokens"),
        ("Total generated tokens", "Total generated tokens"),
        ("Request throughput (req/s)", "Request throughput (req/s)"),
        ("Output token throughput (tok/s)", "Output token throughput (tok/s)"),
        (
            "Peak output token throughput (tok/s)",
            "Peak output token throughput (tok/s)",
        ),
        ("Peak concurrent requests", "Peak concurrent requests"),
        ("Total token throughput (tok/s)", "Total token throughput (tok/s)"),
        ("[Time to First Token]", ""),
        ("Mean TTFT (ms)", "Mean TTFT (ms)"),
        ("Median TTFT (ms)", "Median TTFT (ms)"),
        ("P95 TTFT (ms)", "P95 TTFT (ms)"),
        ("P99 TTFT (ms)", "P99 TTFT (ms)"),
        ("[Time per Output Token]", ""),
        ("Mean TPOT (ms)", "Mean TPOT (ms)"),
        ("Median TPOT (ms)", "Median TPOT (ms)"),
        ("P95 TPOT (ms)", "P95 TPOT (ms)"),
        ("P99 TPOT (ms)", "P99 TPOT (ms)"),
        ("[Inter-token Latency]", ""),
        ("Mean ITL (ms)", "Mean ITL (ms)"),
        ("Median ITL (ms)", "Median ITL (ms)"),
        ("P95 ITL (ms)", "P95 ITL (ms)"),
        ("P99 ITL (ms)", "P99 ITL (ms)"),
    ]

    csv_lines = []
    header = ["Metric"] + [f"{conc} concurrency" for conc in concurrencies]
    csv_lines.append(",".join(header))

    for display_name, key_name in metric_names:
        if not key_name:
            csv_lines.append(f"[{display_name}]" + ",," * (len(concurrencies) - 1))
            continue

        row = [display_name]
        for conc in concurrencies:
            value = chip_data.get(chip_name, {}).get(conc, {}).get(key_name, "")
            row.append(value)
        csv_lines.append(",".join(row))

    csv_file = os.path.join(output_dir, "concurrency_comparison.csv")
    with open(csv_file, "w", encoding="utf-8") as f:
        f.write("\n".join(csv_lines))

    print(f"Generated: {csv_file}")
    return [csv_file]


def generate_comparison_charts(
    chip_data, concurrencies, output_dir, chip_name, model_name=None
):
    if not HAS_MATPLOTLIB:
        return None

    actual_model_name = model_name if model_name else MODEL_NAME
    x = range(len(concurrencies))

    def get_values(key):
        values = []
        for conc in concurrencies:
            val = chip_data.get(chip_name, {}).get(conc, {}).get(key, "0")
            try:
                values.append(float(val))
            except:
                values.append(0)
        return values

    colors = ["#3498db", "#2ecc71", "#e74c3c", "#f39c12", "#9b59b6", "#1abc9c"]

    req_throughput = get_values("Request throughput (req/s)")
    total_tput = get_values("Total token throughput (tok/s)")
    output_tput = get_values("Output token throughput (tok/s)")
    ttft_p99 = get_values("P99 TTFT (ms)")
    tpot_p99 = get_values("P99 TPOT (ms)")
    itl_p99 = get_values("P99 ITL (ms)")

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(
        f"{actual_model_name} on {chip_name} - Concurrency Comparison",
        fontsize=14,
        fontweight="bold",
    )

    axes[0, 0].bar(x, req_throughput, color=colors[: len(concurrencies)], alpha=0.8)
    axes[0, 0].set_title("Request Throughput (req/s)", fontsize=11)
    axes[0, 0].set_xlabel("Concurrency")
    axes[0, 0].set_ylabel("req/s")
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(concurrencies, rotation=45)
    for i, v in enumerate(req_throughput):
        axes[0, 0].text(
            i,
            v + 0.02 * max(req_throughput) if max(req_throughput) > 0 else 0.1,
            f"{v:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    axes[0, 0].grid(axis="y", alpha=0.3)

    axes[0, 1].bar(x, output_tput, color=colors[: len(concurrencies)], alpha=0.8)
    axes[0, 1].set_title("Output Token Throughput (tok/s)", fontsize=11)
    axes[0, 1].set_xlabel("Concurrency")
    axes[0, 1].set_ylabel("tok/s")
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(concurrencies, rotation=45)
    for i, v in enumerate(output_tput):
        axes[0, 1].text(
            i,
            v + 0.02 * max(output_tput) if max(output_tput) > 0 else 1,
            f"{v:.0f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    axes[0, 1].grid(axis="y", alpha=0.3)

    axes[0, 2].bar(x, total_tput, color=colors[: len(concurrencies)], alpha=0.8)
    axes[0, 2].set_title("Total Token Throughput (tok/s)", fontsize=11)
    axes[0, 2].set_xlabel("Concurrency")
    axes[0, 2].set_ylabel("tok/s")
    axes[0, 2].set_xticks(x)
    axes[0, 2].set_xticklabels(concurrencies, rotation=45)
    for i, v in enumerate(total_tput):
        axes[0, 2].text(
            i,
            v + 0.02 * max(total_tput) if max(total_tput) > 0 else 100,
            f"{v:.0f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    axes[0, 2].grid(axis="y", alpha=0.3)

    axes[1, 0].bar(x, ttft_p99, color=colors[: len(concurrencies)], alpha=0.8)
    axes[1, 0].set_title("TTFT P99 (ms)", fontsize=11)
    axes[1, 0].set_xlabel("Concurrency")
    axes[1, 0].set_ylabel("ms")
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(concurrencies, rotation=45)
    for i, v in enumerate(ttft_p99):
        axes[1, 0].text(
            i,
            v + 0.02 * max(ttft_p99) if max(ttft_p99) > 0 else 10,
            f"{v:.0f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    axes[1, 0].grid(axis="y", alpha=0.3)

    axes[1, 1].bar(x, tpot_p99, color=colors[: len(concurrencies)], alpha=0.8)
    axes[1, 1].set_title("TPOT P99 (ms)", fontsize=11)
    axes[1, 1].set_xlabel("Concurrency")
    axes[1, 1].set_ylabel("ms")
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(concurrencies, rotation=45)
    for i, v in enumerate(tpot_p99):
        axes[1, 1].text(
            i,
            v + 0.02 * max(tpot_p99) if max(tpot_p99) > 0 else 1,
            f"{v:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    axes[1, 1].grid(axis="y", alpha=0.3)

    axes[1, 2].bar(x, itl_p99, color=colors[: len(concurrencies)], alpha=0.8)
    axes[1, 2].set_title("ITL P99 (ms)", fontsize=11)
    axes[1, 2].set_xlabel("Concurrency")
    axes[1, 2].set_ylabel("ms")
    axes[1, 2].set_xticks(x)
    axes[1, 2].set_xticklabels(concurrencies, rotation=45)
    for i, v in enumerate(itl_p99):
        axes[1, 2].text(
            i,
            v + 0.02 * max(itl_p99) if max(itl_p99) > 0 else 1,
            f"{v:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    axes[1, 2].grid(axis="y", alpha=0.3)

    for ax in axes.flat:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.set_facecolor("#f0f0f0")
    for ax in axes.flat:
        ax.set_facecolor("white")

    plt.tight_layout()

    chart_file = os.path.join(output_dir, "concurrency_comparison.png")
    plt.savefig(chart_file, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Generated chart: {chart_file}")
    return [chart_file]


def generate_performance_trends(
    chip_data, concurrencies, output_dir, chip_name, model_name=None
):
    if not HAS_MATPLOTLIB:
        return None

    actual_model_name = model_name if model_name else MODEL_NAME
    concurrencies_int = [int(c) for c in concurrencies]

    colors = ["#3498db", "#2ecc71", "#e74c3c", "#f39c12", "#9b59b6"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(
        f"{actual_model_name} on {chip_name} - Performance Trends by Concurrency",
        fontsize=14,
        fontweight="bold",
    )

    def get_values(key):
        return [
            float(chip_data.get(chip_name, {}).get(c, {}).get(key, 0) or 0)
            for c in concurrencies
        ]

    values = get_values("Request throughput (req/s)")
    axes[0, 0].plot(
        concurrencies_int,
        values,
        "-o",
        color=colors[0],
        linewidth=2,
        markersize=6,
        label="QPS",
    )
    axes[0, 0].set_title("Request Throughput (req/s)")
    axes[0, 0].set_xlabel("Concurrency")
    axes[0, 0].set_ylabel("req/s")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    values = get_values("Output token throughput (tok/s)")
    axes[0, 1].plot(
        concurrencies_int,
        values,
        "-o",
        color=colors[1],
        linewidth=2,
        markersize=6,
        label="Output",
    )
    axes[0, 1].set_title("Output Token Throughput (tok/s)")
    axes[0, 1].set_xlabel("Concurrency")
    axes[0, 1].set_ylabel("tok/s")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    values = get_values("Total token throughput (tok/s)")
    axes[0, 2].plot(
        concurrencies_int,
        values,
        "-o",
        color=colors[2],
        linewidth=2,
        markersize=6,
        label="Total",
    )
    axes[0, 2].set_title("Total Token Throughput (tok/s)")
    axes[0, 2].set_xlabel("Concurrency")
    axes[0, 2].set_ylabel("tok/s")
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    values_mean = get_values("Mean TTFT (ms)")
    values_p99 = get_values("P99 TTFT (ms)")
    axes[1, 0].plot(
        concurrencies_int,
        values_mean,
        "-o",
        color=colors[0],
        linewidth=2,
        markersize=6,
        label="Mean",
    )
    axes[1, 0].plot(
        concurrencies_int,
        values_p99,
        "--s",
        color=colors[1],
        linewidth=1,
        markersize=4,
        alpha=0.6,
        label="P99",
    )
    axes[1, 0].set_title("TTFT Latency (ms)")
    axes[1, 0].set_xlabel("Concurrency")
    axes[1, 0].set_ylabel("ms")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    values_mean = get_values("Mean TPOT (ms)")
    values_p99 = get_values("P99 TPOT (ms)")
    axes[1, 1].plot(
        concurrencies_int,
        values_mean,
        "-o",
        color=colors[0],
        linewidth=2,
        markersize=6,
        label="Mean",
    )
    axes[1, 1].plot(
        concurrencies_int,
        values_p99,
        "--s",
        color=colors[1],
        linewidth=1,
        markersize=4,
        alpha=0.6,
        label="P99",
    )
    axes[1, 1].set_title("TPOT Latency (ms)")
    axes[1, 1].set_xlabel("Concurrency")
    axes[1, 1].set_ylabel("ms")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    values_mean = get_values("Mean ITL (ms)")
    values_p99 = get_values("P99 ITL (ms)")
    axes[1, 2].plot(
        concurrencies_int,
        values_mean,
        "-o",
        color=colors[0],
        linewidth=2,
        markersize=6,
        label="Mean",
    )
    axes[1, 2].plot(
        concurrencies_int,
        values_p99,
        "--s",
        color=colors[1],
        linewidth=1,
        markersize=4,
        alpha=0.6,
        label="P99",
    )
    axes[1, 2].set_title("ITL Latency (ms)")
    axes[1, 2].set_xlabel("Concurrency")
    axes[1, 2].set_ylabel("ms")
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()

    chart_file = os.path.join(output_dir, "performance_trends.png")
    plt.savefig(chart_file, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Generated performance trends chart: {chart_file}")
    return chart_file


def generate_performance_trends_csv(chip_data, concurrencies, output_dir, chip_name):
    metric_names = [
        ("[Serving Benchmark Result]", ""),
        ("Successful requests", "Successful requests"),
        ("Failed requests", "Failed requests"),
        ("Benchmark duration (s)", "Benchmark duration (s)"),
        ("Total input tokens", "Total input tokens"),
        ("Total generated tokens", "Total generated tokens"),
        ("Request throughput (req/s)", "Request throughput (req/s)"),
        ("Output token throughput (tok/s)", "Output token throughput (tok/s)"),
        (
            "Peak output token throughput (tok/s)",
            "Peak output token throughput (tok/s)",
        ),
        ("Peak concurrent requests", "Peak concurrent requests"),
        ("Total token throughput (tok/s)", "Total token throughput (tok/s)"),
        ("[Time to First Token]", ""),
        ("Mean TTFT (ms)", "Mean TTFT (ms)"),
        ("Median TTFT (ms)", "Median TTFT (ms)"),
        ("P95 TTFT (ms)", "P95 TTFT (ms)"),
        ("P99 TTFT (ms)", "P99 TTFT (ms)"),
        ("[Time per Output Token]", ""),
        ("Mean TPOT (ms)", "Mean TPOT (ms)"),
        ("Median TPOT (ms)", "Median TPOT (ms)"),
        ("P95 TPOT (ms)", "P95 TPOT (ms)"),
        ("P99 TPOT (ms)", "P99 TPOT (ms)"),
        ("[Inter-token Latency]", ""),
        ("Mean ITL (ms)", "Mean ITL (ms)"),
        ("Median ITL (ms)", "Median ITL (ms)"),
        ("P95 ITL (ms)", "P95 ITL (ms)"),
        ("P99 ITL (ms)", "P99 ITL (ms)"),
    ]

    csv_lines = []
    header = ["Metric"] + [f"{conc}" for conc in concurrencies]
    csv_lines.append(",".join(header))

    for display_name, key_name in metric_names:
        if not key_name:
            csv_lines.append(f"[{display_name}]" + ",," * (len(concurrencies) - 1))
            continue

        row = [display_name]
        for conc in concurrencies:
            value = chip_data.get(chip_name, {}).get(conc, {}).get(key_name, "")
            row.append(value)
        csv_lines.append(",".join(row))

    csv_file = os.path.join(output_dir, "performance_trends.csv")
    with open(csv_file, "w", encoding="utf-8") as f:
        f.write("\n".join(csv_lines))

    print(f"Generated performance trends CSV: {csv_file}")
    return csv_file


def generate_analysis_content(chip_data, chip_name, concurrencies):
    analysis_lines = []

    low_conc = [c for c in concurrencies if int(c) <= 4]
    mid_conc = [c for c in concurrencies if 4 < int(c) <= 32]
    high_conc = [c for c in concurrencies if int(c) > 32]

    def get_avg_by_conc_range(key, conc_list):
        vals = []
        for c in conc_list:
            val = chip_data.get(chip_name, {}).get(c, {}).get(key, 0)
            try:
                vals.append(float(val))
            except:
                pass
        return sum(vals) / len(vals) if vals else 0

    def get_max_value(key):
        max_val = 0
        max_conc = None
        for c in concurrencies:
            val = chip_data.get(chip_name, {}).get(c, {}).get(key, 0)
            try:
                fval = float(val)
                if fval > max_val:
                    max_val = fval
                    max_conc = c
            except:
                pass
        return max_val, max_conc

    def get_min_value(key):
        min_val = float("inf")
        min_conc = None
        for c in concurrencies:
            val = chip_data.get(chip_name, {}).get(c, {}).get(key, float("inf"))
            try:
                fval = float(val)
                if fval > 0 and fval < min_val:
                    min_val = fval
                    min_conc = c
            except:
                pass
        return min_val if min_val != float("inf") else 0, min_conc

    throughputs = [
        float(
            chip_data.get(chip_name, {}).get(c, {}).get("Request throughput (req/s)", 0)
            or 0
        )
        for c in concurrencies
    ]
    avg_low_tp = get_avg_by_conc_range("Request throughput (req/s)", low_conc)
    avg_mid_tp = get_avg_by_conc_range("Request throughput (req/s)", mid_conc)
    avg_high_tp = (
        get_avg_by_conc_range("Request throughput (req/s)", high_conc)
        if high_conc
        else 0
    )

    max_tp, max_tp_conc = get_max_value("Request throughput (req/s)")

    analysis_lines.append("### 1. 吞吐量性能分析\n")
    analysis_lines.append(f"**请求吞吐量 (QPS)**: 随着并发级别增加，QPS持续上升。")
    if low_conc:
        analysis_lines.append(
            f"低并发({','.join(low_conc)})平均 QPS: {avg_low_tp:.2f} req/s；"
        )
    if mid_conc:
        analysis_lines.append(
            f"中并发({','.join(mid_conc)})平均 QPS: {avg_mid_tp:.2f} req/s；"
        )
    if high_conc:
        analysis_lines.append(
            f"高并发({','.join(high_conc)})平均 QPS: {avg_high_tp:.2f} req/s；"
        )
    analysis_lines.append(
        f"最高 QPS 出现在 {max_tp_conc} 并发，达到 {max_tp:.2f} req/s。\n"
    )

    total_tput, total_tput_conc = get_max_value("Total token throughput (tok/s)")
    analysis_lines.append(
        f"**Token总吞吐量**: 最高达到 {total_tput:.0f} tok/s ({total_tput_conc} 并发)。\n"
    )

    ttft_p99, ttft_p99_conc = get_max_value("P99 TTFT (ms)")
    ttft_p99_min, _ = get_min_value("P99 TTFT (ms)")
    ttft_avg_low = get_avg_by_conc_range("P99 TTFT (ms)", low_conc)
    ttft_avg_high = (
        get_avg_by_conc_range("P99 TTFT (ms)", high_conc) if high_conc else 0
    )

    analysis_lines.append("### 2. 首Token延迟 (TTFT) 分析\n")
    analysis_lines.append(f"TTFT随并发增加显著上升。")
    if low_conc:
        analysis_lines.append(f"低并发平均 P99 TTFT: {ttft_avg_low:.0f}ms；")
    if high_conc:
        analysis_lines.append(f"高并发平均 P99 TTFT: {ttft_avg_high:.0f}ms；")
    analysis_lines.append(
        f"最高 P99 TTFT 出现在 {ttft_p99_conc} 并发，达到 {ttft_p99:.0f}ms。\n"
    )

    tpot_p99, tpot_p99_conc = get_max_value("P99 TPOT (ms)")
    tpot_avg_low = get_avg_by_conc_range("P99 TPOT (ms)", low_conc)
    tpot_avg_high = (
        get_avg_by_conc_range("P99 TPOT (ms)", high_conc) if high_conc else 0
    )

    analysis_lines.append("### 3. Token生成时间 (TPOT) 分析\n")
    analysis_lines.append(f"TPOT随并发增加也呈上升趋势。")
    if low_conc:
        analysis_lines.append(f"低并发平均 P99 TPOT: {tpot_avg_low:.2f}ms；")
    if high_conc:
        analysis_lines.append(f"高并发平均 P99 TPOT: {tpot_avg_high:.2f}ms；")
    analysis_lines.append(
        f"最高 P99 TPOT 出现在 {tpot_p99_conc} 并发，达到 {tpot_p99:.2f}ms。\n"
    )

    itl_p99, itl_p99_conc = get_max_value("P99 ITL (ms)")
    itl_avg_low = get_avg_by_conc_range("P99 ITL (ms)", low_conc)
    itl_avg_high = get_avg_by_conc_range("P99 ITL (ms)", high_conc) if high_conc else 0

    analysis_lines.append("### 4. Token间延迟 (ITL) 分析\n")
    analysis_lines.append(f"ITL随并发增加呈上升趋势。")
    if low_conc:
        analysis_lines.append(f"低并发平均 P99 ITL: {itl_avg_low:.2f}ms；")
    if high_conc:
        analysis_lines.append(f"高并发平均 P99 ITL: {itl_avg_high:.2f}ms；")
    analysis_lines.append(
        f"最高 P99 ITL 出现在 {itl_p99_conc} 并发，达到 {itl_p99:.2f}ms。\n"
    )

    analysis_lines.append("### 5. 综合评估\n")
    if throughputs:
        growth_rate = (
            (throughputs[-1] / throughputs[0] - 1) * 100 if throughputs[0] > 0 else 0
        )
        analysis_lines.append(
            f"**吞吐量增长**: 从最低并发到最高并发，QPS增长了 {growth_rate:.1f}%。"
        )

    if ttft_p99 > 0 and ttft_avg_low > 0:
        ttft_growth = (ttft_p99 / ttft_avg_low - 1) * 100 if ttft_avg_low > 0 else 0
        analysis_lines.append(
            f"**TTFT延迟恶化**: 高并发相比低并发，TTFT P99增加了 {ttft_growth:.1f}%。"
        )

    if tpot_p99 > 0 and tpot_avg_low > 0:
        tpot_growth = (tpot_p99 / tpot_avg_low - 1) * 100 if tpot_avg_low > 0 else 0
        analysis_lines.append(
            f"**TPOT延迟恶化**: 高并发相比低并发，TPOT P99增加了 {tpot_growth:.1f}%。"
        )

    conclusion = "\n".join(analysis_lines)
    return conclusion


def generate_markdown_report(
    chip_data,
    concurrencies,
    output_dir,
    test_suite,
    chip_name,
    model_name=None,
    scenarios_config=None,
):
    current_date = datetime.now().strftime("%Y-%m-%d")

    actual_model_name = model_name if model_name else MODEL_NAME
    model_key = actual_model_name

    chip_config = load_chip_config()
    vllm_config = load_vllm_config()

    chips_raw = chip_config.get("chips", {})
    vllm_configs_raw = vllm_config.get("vllm_configs", {})

    chip_configs_list = chips_raw.get(chip_name, [])
    if isinstance(chip_configs_list, list):
        for cfg in chip_configs_list:
            if cfg.get("model_name") == model_key:
                chips_info = cfg
                break
        else:
            chips_info = chip_configs_list[0] if chip_configs_list else {}
    else:
        chips_info = chip_configs_list if chip_configs_list else {}

    vllm_cfg_list = vllm_configs_raw.get(chip_name, [])
    if isinstance(vllm_cfg_list, list):
        for cfg in vllm_cfg_list:
            if cfg.get("model_name") == model_key:
                vllm_cfg = cfg
                break
        else:
            vllm_cfg = vllm_cfg_list[0] if vllm_cfg_list else {}
    else:
        vllm_cfg = vllm_cfg_list if vllm_cfg_list else {}

    if scenarios_config is None:
        scenarios_config = load_models_scenarios()

    base_config = scenarios_config.get("base_config", {})
    params = base_config.get("params", {})
    test_cfg = params.get(test_suite, {})

    models_config = scenarios_config.get("models", {})

    chip_key_map = {
        "Hygon_BW1000": "hygon_bw1000",
        "Kunlun_P800": "kunlun_p800",
        "NVIDIA_H100": "nvidia_h100",
    }
    chip_key = chip_key_map.get(chip_name, chip_name.lower())
    chip_models = models_config.get(chip_key, [])
    model_info = chip_models[0] if chip_models else {}
    model_path = model_info.get("model_path", "N/A")

    def make_table_for_metric(key_name):
        values = []
        for conc in concurrencies:
            value = chip_data.get(chip_name, {}).get(conc, {}).get(key_name, "")
            values.append(value)
        return " | ".join(values)

    serving_metrics = [
        ("成功请求数", "Successful requests"),
        ("失败请求数", "Failed requests"),
        ("测试持续时间 (s)", "Benchmark duration (s)"),
        ("总输入 tokens", "Total input tokens"),
        ("总生成 tokens", "Total generated tokens"),
        ("**请求吞吐量 (req/s)**", "Request throughput (req/s)"),
        ("**输出 token 吞吐量 (tok/s)**", "Output token throughput (tok/s)"),
        ("峰值输出 token 吞吐量 (tok/s)", "Peak output token throughput (tok/s)"),
        ("峰值并发请求数", "Peak concurrent requests"),
        ("**总 token 吞吐量 (tok/s)**", "Total token throughput (tok/s)"),
    ]

    ttft_metrics = [
        ("平均 TTFT (ms)", "Mean TTFT (ms)"),
        ("中位 TTFT (ms)", "Median TTFT (ms)"),
        ("P95 TTFT (ms)", "P95 TTFT (ms)"),
        ("P99 TTFT (ms)", "P99 TTFT (ms)"),
    ]

    tpot_metrics = [
        ("平均 TPOT (ms)", "Mean TPOT (ms)"),
        ("中位 TPOT (ms)", "Median TPOT (ms)"),
        ("P95 TPOT (ms)", "P95 TPOT (ms)"),
        ("P99 TPOT (ms)", "P99 TPOT (ms)"),
    ]

    itl_metrics = [
        ("平均 ITL (ms)", "Mean ITL (ms)"),
        ("中位 ITL (ms)", "Median ITL (ms)"),
        ("P95 ITL (ms)", "P95 ITL (ms)"),
        ("P99 ITL (ms)", "P99 ITL (ms)"),
    ]

    header = " | ".join([f"{conc} 并发" for conc in concurrencies])
    separator = " | ".join(["-----------"] * len(concurrencies))

    serving_table = "\n".join(
        [f"| {name} | {make_table_for_metric(key)} |" for name, key in serving_metrics]
    )

    ttft_table = "\n".join(
        [f"| {name} | {make_table_for_metric(key)} |" for name, key in ttft_metrics]
    )

    tpot_table = "\n".join(
        [f"| {name} | {make_table_for_metric(key)} |" for name, key in tpot_metrics]
    )

    itl_table = "\n".join(
        [f"| {name} | {make_table_for_metric(key)} |" for name, key in itl_metrics]
    )

    analysis_content = generate_analysis_content(chip_data, chip_name, concurrencies)

    concurrency_comparison_img = (
        '<img src="./concurrency_comparison.png" width="1000" />'
    )
    performance_trends_img = '<img src="./performance_trends.png" width="1000" />'

    def format_tokens(val):
        try:
            v = int(val)
            if v >= 1024:
                return f"{v // 1024}k"
            else:
                return f"{v / 1024:.2f}k"
        except:
            return str(val)

    dataset = test_cfg.get("dataset-name", "random")
    num_prompts = test_cfg.get("num-prompts", [])
    input_output_lens = test_cfg.get("random-input-output-len", [])
    if (
        input_output_lens
        and isinstance(input_output_lens[0], list)
        and len(input_output_lens[0]) >= 2
    ):
        input_len = [input_output_lens[0][0]]
        output_len = [input_output_lens[0][1]]
    else:
        input_len = test_cfg.get("random-input-len", [])
        output_len = test_cfg.get("random-output-len", [])

    input_ctx = format_tokens(input_len[0]) if input_len else "N/A"
    output_ctx = format_tokens(output_len[0]) if output_len else "N/A"

    chip_info = chips_info
    chip_param_names = [
        "model_name",
        "quantization_config",
        "model_size",
        "max_position_embeddings",
        "temperature",
        "top_k",
        "top_p",
        "transformers_version",
        "vllm_version",
        "python_version",
    ]
    chip_table_rows = []
    for param in chip_param_names:
        val = chip_info.get(param, "N/A")
        chip_table_rows.append(f"| **{param}** | {val} |")
    chip_table = "\n".join(chip_table_rows)

    vllm_param_names = [
        "model_name",
        "max-model-len",
        "max-num-seqs",
        "max-num-batched-tokens",
        "gpu-memory-utilization",
        "dtype",
        "block_size",
        "dp",
        "tp",
        "pp",
        "enable-export-parallel",
        "enable-auto-tool-choice",
        "tool-call-parser",
        "reasoning-parser",
    ]
    vllm_table_rows = []
    for param in vllm_param_names:
        val = vllm_cfg.get(param, "N/A")
        vllm_table_rows.append(f"| {param} | {val} |")
    vllm_table = "\n".join(vllm_table_rows)

    remark = vllm_cfg.get("remarks", "")
    remarks_section = f"- **{chip_name}**: {remark}" if remark else ""

    md_content = f"""# {actual_model_name}模型在{chip_name}上的Benchmark基准测试报告

<div align="center">
**测试日期：** {current_date}

</div>

---

## 测试场景
在固定请求数，输入上下文和输出上下文长度下，使用vllm bench serve工具对并发数逐级增加场景的性能基准验证。分析同一芯片同一模型在不同并发级别下的性能指标变化趋势。

**主要采集指标**：

| 指标                  | 单位         | 含义                                 |
|---------------------|------------|------------------------------------|
| TTFT                | ms         | Time To First Token，首 token 延迟     |
| TPOT                | ms/token   | Time Per Output Token，每 token 生成时间 |
| Throughput          | tokens/s   | 系统总吞吐                              |
| QPS                 | requests/s | 请求吞吐                               |
| P50/P95/P99 Latency | ms         | 延迟分位数                              |


## 📊 测试概览

| 项目            | 配置                                     | 备注  |
|---------------|----------------------------------------|-----|
| **数据集**       | {dataset}                                 |     |
| **并发数**       | {", ".join(concurrencies)}    |     |
| **总请求数**      | {num_prompts[0] if num_prompts else "N/A"}                                    |     |
| **请求输入上下文长度** | {input_len[0] if input_len else "N/A"}（{input_ctx}）                             |     |
| **请求输出上下文长度** | {output_len[0] if output_len else "N/A"}（{output_ctx}）                             |     |
| **模型**        | {actual_model_name}                           |     |
| **被测芯片**      | {chip_name} |     |

---

## 🤖 芯片和模型配置信息

| 参数名称                    | {chip_name} |
|------------------------|-------------|
{chip_table}

---

## 🤖 vLLM启动配置信息

| 参数名称                   | {chip_name} |
|------------------------|-------------|
{vllm_table}

{remarks_section}

---

## 🎯 服务基准结果

| 指标 | {header} |
|------|{separator}|
{serving_table}

---

## ⏱️ 首Token延迟 (TTFT)

| 指标 | {header} |
|------|{separator}|
{ttft_table}

---

## ⚡ 每Token生成时间 (TPOT)

| 指标 | {header} |
|------|{separator}|
{tpot_table}

---

## 🔄 Token间延迟 (ITL)

| 指标 | {header} |
|------|{separator}|
{itl_table}

---

## 📊 各并发级别性能柱状图

{concurrency_comparison_img}

---

## 📈 性能趋势分析

{performance_trends_img}

---

## 📝 分析总结

{analysis_content}

---

<div align="center">
*报告生成时间: {current_date}*
</div>
"""

    md_file = os.path.join(
        output_dir, f"{actual_model_name}_{chip_name}_concurrency.md"
    )
    with open(md_file, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"Generated: {md_file}")
    return md_file


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate single chip benchmark report"
    )
    parser.add_argument(
        "--chip",
        type=str,
        default=None,
        help="Chip name (e.g., Hygon_BW1000, Kunlun_P800, NVIDIA_H100)",
    )
    parser.add_argument(
        "--model", type=str, default=None, help="Model name (e.g., MiniMax-M2.5)"
    )
    parser.add_argument(
        "--test-suite", type=str, default=None, help="Test suite name (e.g., test_01)"
    )
    parser.add_argument("--run-id", type=str, default=None, help="Run ID (e.g., 01)")
    parser.add_argument(
        "--concurrency",
        type=str,
        default=None,
        help="Specific concurrency levels to include, comma-separated (e.g., 1,2,4,8,10)",
    )
    args = parser.parse_args()

    chip_to_use = args.chip.strip() if args.chip else list(CHIP_BASE_PATHS.keys())[0]
    chip_to_use = chip_to_use.lower() if chip_to_use else chip_to_use
    chip_key_map_reverse = {
        "hygon_bw1000": "Hygon_BW1000",
        "kunlun_p800": "Kunlun_P800",
        "nvidia_h100": "NVIDIA_H100",
    }
    chip_to_use = chip_key_map_reverse.get(chip_to_use, chip_to_use)

    model_input = args.model.strip() if args.model else MODEL_NAME
    model_key_map = {
        "minimax-m2.5": "MiniMax-M2.5",
        "qwen3.5": "Qwen3.5",
    }
    model_to_use = model_key_map.get(model_input.lower(), model_input)
    model_default = MODEL_NAME

    if not args.model:
        model_to_use = model_default

    test_suite_to_use = args.test_suite.strip() if args.test_suite else TEST_SUITES[0]
    run_id_to_use = args.run_id.strip() if args.run_id else RUN_ID

    concurrency_filter = None
    if args.concurrency:
        concurrency_filter = [s.strip() for s in args.concurrency.split(",")]

    scenarios_config = load_models_scenarios()

    base_path = f"reports/{chip_to_use.lower()}/benchmark/{model_to_use}"

    if not os.path.exists(base_path):
        print(f"\nError: No data found for chip={chip_to_use}, model={model_to_use}")
        print(f"Expected path: {base_path}")

        available_reports = f"reports/{chip_to_use.lower()}/benchmark/"
        if os.path.exists(available_reports):
            print(f"Available model reports:")
            for item in os.listdir(available_reports):
                print(f"  - {item}")
        return

    full_base_path = f"{base_path}/{test_suite_to_use}/{run_id_to_use}"

    if not os.path.exists(full_base_path):
        print(
            f"\nError: No data found for {chip_to_use}/{model_to_use} test_suite={test_suite_to_use} run_id={run_id_to_use}"
        )
        print(f"Expected path: {full_base_path}")
        return

    chip_configs = [
        {
            "name": chip_to_use,
            "base_path": full_base_path,
        }
    ]

    print(f"\n{'#' * 60}")
    print(
        f"Processing: chip={chip_to_use}, model={model_to_use}, test_suite={test_suite_to_use}, run_id={run_id_to_use}"
    )
    print(f"{'#' * 60}\n")

    for chip in chip_configs:
        chip_name = chip["name"]
        output_base = f"analysis/single_chip/{chip_name}/{model_to_use}/{test_suite_to_use}/{run_id_to_use}"

        all_concurrencies = set()
        concs = get_all_concurrencies(chip)
        all_concurrencies.update(concs)

        if not all_concurrencies:
            print(
                f"No concurrency configurations found for {chip_name} / {test_suite_to_use}!"
            )
            continue

        concurrencies = sorted(all_concurrencies, key=lambda x: int(x))

        if concurrency_filter:
            filtered_concs = [c for c in concurrencies if c in concurrency_filter]
            if filtered_concs:
                concurrencies = filtered_concs
                print(f"Using specified concurrency levels: {', '.join(concurrencies)}")
            else:
                print(
                    f"Warning: None of the specified concurrency levels {concurrency_filter} found, using all"
                )

        print(
            f"Found {len(concurrencies)} concurrency levels: {', '.join(concurrencies)}"
        )

        Path(output_base).mkdir(parents=True, exist_ok=True)

        chip_data = defaultdict(lambda: defaultdict(dict))

        print(f"\nProcessing chip: {chip_name}")

        for conc in concurrencies:
            metrics = get_chip_metrics(chip, conc)
            if metrics:
                chip_data[chip_name][conc] = metrics
                print(f"  - {conc}并发: OK")
            else:
                print(f"  - {conc}并发: No data")

        print("\nGenerating comparison reports...")

        generate_comparison_csv(chip_data, concurrencies, output_base, chip_name)

        if HAS_MATPLOTLIB:
            generate_comparison_charts(
                chip_data, concurrencies, output_base, chip_name, model_to_use
            )
            generate_performance_trends(
                chip_data, concurrencies, output_base, chip_name, model_to_use
            )

        generate_performance_trends_csv(
            chip_data, concurrencies, output_base, chip_name
        )

        generate_markdown_report(
            chip_data,
            concurrencies,
            output_base,
            test_suite_to_use,
            chip_name,
            model_name=model_to_use,
            scenarios_config=scenarios_config,
        )

        print(f"\n{'=' * 50}")
        print(
            f"Single chip analysis for {chip_name} - {test_suite_to_use} generated successfully!"
        )
        print(f"Output directory: {output_base}")
        print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
