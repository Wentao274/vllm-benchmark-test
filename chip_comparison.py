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

RUN_IDS = ["01", "01", "01"]

CHIP_BASE_PATHS = {
    "Hygon_BW1000": "reports/benchmark/hygon_bw1000/MiniMax-M2.5-W8A8",
    "Kunlun_P800": "reports/benchmark/kunlun_p800/MiniMax-M2.5-W8A8-INT8-Dynamic",
    "NVIDIA_H100": "reports/benchmark/nvidia_h100/MiniMax-M2.5",
}

MODEL_NAME = "MiniMax-M2.5"


def get_chip_configs(test_suite, chip_run_ids):
    chip_paths = get_chip_path_from_model_config()
    configs = []
    for i, chip_name in enumerate(chip_paths.keys()):
        base = CHIP_BASE_PATHS.get(chip_name, "")
        model_name = chip_paths.get(chip_name, "")
        run_id = chip_run_ids[i] if i < len(chip_run_ids) else chip_run_ids[0]
        configs.append(
            {
                "name": chip_name,
                "base_path": f"{base}/{model_name}/{test_suite}/{run_id}",
            }
        )
    return configs


def get_chip_path_from_model_config():
    chip_config = load_chip_config()
    chips_raw = chip_config.get("chips", {})

    chip_paths = {}
    for chip_name, model_configs in chips_raw.items():
        if isinstance(model_configs, list):
            for cfg in model_configs:
                model_name = cfg.get("model_name", "")
                if model_name:
                    chip_paths[chip_name] = model_name
                    break
        elif isinstance(model_configs, dict):
            model_name = model_configs.get("model_name", "")
            if model_name:
                chip_paths[chip_name] = model_name

    return chip_paths


def load_chip_config_by_model():
    chip_config = load_chip_config()

    chip_info_map = {}
    chips_raw = chip_config.get("chips", {})

    for chip_name, base_path in CHIP_BASE_PATHS.items():
        if not base_path:
            continue
        model_name = Path(base_path).name
        chip_configs = chips_raw.get(chip_name, [])
        if isinstance(chip_configs, list):
            for cfg in chip_configs:
                if cfg.get("model_name") == model_name:
                    chip_info_map[chip_name] = cfg
                    break
            else:
                chip_info_map[chip_name] = chip_configs[0] if chip_configs else {}
        elif isinstance(chip_configs, dict):
            chip_info_map[chip_name] = chip_configs

    return chip_info_map


def load_vllm_config_by_model():
    vllm_config = load_vllm_config()

    vllm_info_map = {}
    vllm_configs_raw = vllm_config.get("vllm_configs", {})

    for chip_name, base_path in CHIP_BASE_PATHS.items():
        if not base_path:
            continue
        model_name = Path(base_path).name
        config_list = vllm_configs_raw.get(chip_name, [])
        if isinstance(config_list, list):
            for cfg in config_list:
                if cfg.get("model_name") == model_name:
                    vllm_info_map[chip_name] = cfg
                    break
            else:
                vllm_info_map[chip_name] = config_list[0] if config_list else {}
        elif isinstance(config_list, dict):
            vllm_info_map[chip_name] = config_list

    return vllm_info_map


def load_chip_config(config_path="config/chip_conf.yaml"):
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


def get_chip_config_by_base_path(chip_name, base_path):
    chip_config = load_chip_config()
    chips_raw = chip_config.get("chips", {})

    path_model = base_path.split("/")[-1] if "/" in base_path else base_path

    chip_configs = chips_raw.get(chip_name, [])
    if isinstance(chip_configs, list):
        for cfg in chip_configs:
            if cfg.get("model") == path_model:
                return cfg
        return chip_configs[0] if chip_configs else {}
    elif isinstance(chip_configs, dict):
        return chip_configs

    return {}


def load_vllm_config(config_path="config/model_deployment.yaml"):
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


def load_models_scenarios(config_path="config/models_scenarios.yaml"):
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


def get_test_suite_config(test_suite, scenarios_config):
    base_config = scenarios_config.get("base_config", {})
    params = base_config.get("params", {})
    return params.get(test_suite, {})


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


COMPARISON_METRICS = [
    ("请求吞吐量（Request throughput (req/s)）", "Request throughput (req/s)"),
    (
        "输出token吞吐量（Output token throughput (tok/s)）",
        "Output token throughput (tok/s)",
    ),
    (
        "总token吞吐量（Total token throughput (tok/s)）",
        "Total token throughput (tok/s)",
    ),
    ("首token延迟（P99 TTFT (ms)）", "P99 TTFT (ms)"),
    ("每token生成时间（P99 TPOT (ms)）", "P99 TPOT (ms)"),
    ("token间延迟（P99 ITL (ms)）", "P99 ITL (ms)"),
]


def generate_comparison_csv(
    chip_data, concurrencies, output_dir, test_suite, chip_names
):
    chip_suffix = "_vs_".join([c.lower() for c in chip_names])

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

    generated_files = []

    for conc in concurrencies:
        csv_lines = []
        header = ["Metric"] + [f"{conc} 并发" for _ in chip_names]
        csv_lines.append(",".join(header))

        for display_name, key_name in metric_names:
            if not key_name:
                csv_lines.append(f"[{display_name}]" + ",," * (len(chip_names) - 1))
                continue

            row = [display_name]
            for chip in chip_names:
                value = chip_data.get(chip, {}).get(conc, {}).get(key_name, "")
                row.append(value)
            csv_lines.append(",".join(row))

        csv_file = os.path.join(
            output_dir, f"concurrency{conc}_comparison_{test_suite}_{chip_suffix}.csv"
        )
        with open(csv_file, "w", encoding="utf-8") as f:
            f.write("\n".join(csv_lines))

        print(f"Generated: {csv_file}")
        generated_files.append(csv_file)

    return generated_files


def generate_comparison_charts(
    chip_data, concurrencies, output_dir, test_suite, chip_names
):
    chip_suffix = "_vs_".join([c.lower() for c in chip_names])
    x = range(len(chip_names))

    def get_values(key):
        values = []
        for chip in chip_names:
            chip_vals = []
            for conc in concurrencies:
                val = chip_data.get(chip, {}).get(conc, {}).get(key, "0")
                try:
                    chip_vals.append(float(val))
                except:
                    chip_vals.append(0)
            values.append(chip_vals)
        return values

    colors = ["#3498db", "#2ecc71", "#e74c3c"]

    req_throughput = get_values("Request throughput (req/s)")
    total_tput = get_values("Total token throughput (tok/s)")
    ttft_p99 = get_values("P99 TTFT (ms)")
    tpot_p99 = get_values("P99 TPOT (ms)")

    chart_files = []

    for conc in concurrencies:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(
            f"{MODEL_NAME} Chip Comparison @ {conc} Concurrency",
            fontsize=14,
            fontweight="bold",
        )

        idx = concurrencies.index(conc)

        values = [req_throughput[i][idx] for i in range(len(chip_names))]
        axes[0, 0].bar(chip_names, values, color=colors, alpha=0.8)
        axes[0, 0].set_title("Request Throughput (req/s)", fontsize=11)
        axes[0, 0].set_ylabel("req/s")
        axes[0, 0].tick_params(axis="x", rotation=15)
        for i, v in enumerate(values):
            axes[0, 0].text(
                i,
                v + 0.02 * max(values) if max(values) > 0 else 0.1,
                f"{v:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        axes[0, 0].grid(axis="y", alpha=0.3)

        values = [total_tput[i][idx] for i in range(len(chip_names))]
        axes[0, 1].bar(chip_names, values, color=colors, alpha=0.8)
        axes[0, 1].set_title("Total Token Throughput (tok/s)", fontsize=11)
        axes[0, 1].set_ylabel("tok/s")
        axes[0, 1].tick_params(axis="x", rotation=15)
        for i, v in enumerate(values):
            axes[0, 1].text(
                i,
                v + 0.02 * max(values) if max(values) > 0 else 100,
                f"{v:.0f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        axes[0, 1].grid(axis="y", alpha=0.3)

        values = [ttft_p99[i][idx] for i in range(len(chip_names))]
        axes[1, 0].bar(chip_names, values, color=colors, alpha=0.8)
        axes[1, 0].set_title("TTFT P99 (ms)", fontsize=11)
        axes[1, 0].set_ylabel("ms")
        axes[1, 0].tick_params(axis="x", rotation=15)
        for i, v in enumerate(values):
            axes[1, 0].text(
                i,
                v + 0.02 * max(values) if max(values) > 0 else 100,
                f"{v:.0f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        axes[1, 0].grid(axis="y", alpha=0.3)

        values = [tpot_p99[i][idx] for i in range(len(chip_names))]
        axes[1, 1].bar(chip_names, values, color=colors, alpha=0.8)
        axes[1, 1].set_title("TPOT P99 (ms)", fontsize=11)
        axes[1, 1].set_ylabel("ms")
        axes[1, 1].tick_params(axis="x", rotation=15)
        for i, v in enumerate(values):
            axes[1, 1].text(
                i,
                v + 0.02 * max(values) if max(values) > 0 else 5,
                f"{v:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        axes[1, 1].grid(axis="y", alpha=0.3)

        for ax in axes.flat:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        fig.set_facecolor("#f0f0f0")
        for i, ax in enumerate(axes.flat):
            ax.set_facecolor("white")
            if i % 2 == 0 and i < 2:
                for spine in ax.spines.values():
                    spine.set_color("#333333")
                    spine.set_linewidth(2)

        plt.tight_layout()

        chart_file = os.path.join(
            output_dir, f"chip_comparison_c{conc}_{test_suite}_{chip_suffix}.png"
        )
        plt.savefig(chart_file, dpi=150, bbox_inches="tight")
        plt.close()

        chart_files.append(chart_file)
        print(f"Generated chart: {chart_file}")

    return chart_files


def generate_performance_trends(
    chip_data, concurrencies, output_dir, test_suite, chip_names
):
    chip_suffix = "_vs_".join([c.lower() for c in chip_names])
    if not HAS_MATPLOTLIB:
        return None

    concurrencies_int = [int(c) for c in concurrencies]

    colors = ["#3498db", "#2ecc71", "#e74c3c"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(
        f"{MODEL_NAME} Performance Trends by Concurrency",
        fontsize=14,
        fontweight="bold",
    )

    def get_chip_values(chip_name, key):
        return [
            float(chip_data.get(chip_name, {}).get(c, {}).get(key, 0) or 0)
            for c in concurrencies
        ]

    for idx, chip_name in enumerate(chip_names):
        color = colors[idx % len(colors)]

        values = get_chip_values(chip_name, "Request throughput (req/s)")
        axes[0, 0].plot(
            concurrencies_int,
            values,
            "-o",
            color=color,
            linewidth=2,
            markersize=6,
            label=chip_name,
        )

    axes[0, 0].set_title("Request Throughput (req/s)")
    axes[0, 0].set_xlabel("Concurrency")
    axes[0, 0].set_ylabel("req/s")
    axes[0, 0].set_xticks(concurrencies_int)
    axes[0, 0].set_xticklabels(concurrencies)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    for idx, chip_name in enumerate(chip_names):
        color = colors[idx % len(colors)]
        values = get_chip_values(chip_name, "Output token throughput (tok/s)")
        axes[0, 1].plot(
            concurrencies_int,
            values,
            "-o",
            color=color,
            linewidth=2,
            markersize=6,
            label=chip_name,
        )

    axes[0, 1].set_title("Output Token Throughput (tok/s)")
    axes[0, 1].set_xlabel("Concurrency")
    axes[0, 1].set_ylabel("tok/s")
    axes[0, 1].set_xticks(concurrencies_int)
    axes[0, 1].set_xticklabels(concurrencies)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    for idx, chip_name in enumerate(chip_names):
        color = colors[idx % len(colors)]
        values = get_chip_values(chip_name, "Total token throughput (tok/s)")
        axes[0, 2].plot(
            concurrencies_int,
            values,
            "-o",
            color=color,
            linewidth=2,
            markersize=6,
            label=chip_name,
        )

    axes[0, 2].set_title("Total Token Throughput (tok/s)")
    axes[0, 2].set_xlabel("Concurrency")
    axes[0, 2].set_ylabel("tok/s")
    axes[0, 2].set_xticks(concurrencies_int)
    axes[0, 2].set_xticklabels(concurrencies)
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    for idx, chip_name in enumerate(chip_names):
        color = colors[idx % len(colors)]
        values = get_chip_values(chip_name, "Mean TTFT (ms)")
        axes[1, 0].plot(
            concurrencies_int,
            values,
            "-o",
            color=color,
            linewidth=2,
            markersize=6,
            label=f"{chip_name} Mean",
        )
        values_p99 = get_chip_values(chip_name, "P99 TTFT (ms)")
        axes[1, 0].plot(
            concurrencies_int,
            values_p99,
            "--s",
            color=color,
            linewidth=1,
            markersize=4,
            alpha=0.6,
            label=f"{chip_name} P99",
        )

    axes[1, 0].set_title("TTFT Latency (ms)")
    axes[1, 0].set_xlabel("Concurrency")
    axes[1, 0].set_ylabel("ms")
    axes[1, 0].set_xticks(concurrencies_int)
    axes[1, 0].set_xticklabels(concurrencies)
    axes[1, 0].legend(fontsize=7)
    axes[1, 0].grid(True, alpha=0.3)

    for idx, chip_name in enumerate(chip_names):
        color = colors[idx % len(colors)]
        values = get_chip_values(chip_name, "Mean TPOT (ms)")
        axes[1, 1].plot(
            concurrencies_int,
            values,
            "-o",
            color=color,
            linewidth=2,
            markersize=6,
            label=f"{chip_name} Mean",
        )
        values_p99 = get_chip_values(chip_name, "P99 TPOT (ms)")
        axes[1, 1].plot(
            concurrencies_int,
            values_p99,
            "--s",
            color=color,
            linewidth=1,
            markersize=4,
            alpha=0.6,
            label=f"{chip_name} P99",
        )

    axes[1, 1].set_title("TPOT Latency (ms)")
    axes[1, 1].set_xlabel("Concurrency")
    axes[1, 1].set_ylabel("ms")
    axes[1, 1].set_xticks(concurrencies_int)
    axes[1, 1].set_xticklabels(concurrencies)
    axes[1, 1].legend(fontsize=7)
    axes[1, 1].grid(True, alpha=0.3)

    for idx, chip_name in enumerate(chip_names):
        color = colors[idx % len(colors)]
        values = get_chip_values(chip_name, "Mean ITL (ms)")
        axes[1, 2].plot(
            concurrencies_int,
            values,
            "-o",
            color=color,
            linewidth=2,
            markersize=6,
            label=f"{chip_name} Mean",
        )
        values_p99 = get_chip_values(chip_name, "P99 ITL (ms)")
        axes[1, 2].plot(
            concurrencies_int,
            values_p99,
            "--s",
            color=color,
            linewidth=1,
            markersize=4,
            alpha=0.6,
            label=f"{chip_name} P99",
        )

    axes[1, 2].set_title("ITL Latency (ms)")
    axes[1, 2].set_xlabel("Concurrency")
    axes[1, 2].set_ylabel("ms")
    axes[1, 2].set_xticks(concurrencies_int)
    axes[1, 2].set_xticklabels(concurrencies)
    axes[1, 2].legend(fontsize=7)
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()

    chart_file = os.path.join(
        output_dir, f"performance_trends_{test_suite}_{chip_suffix}.png"
    )
    plt.savefig(chart_file, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Generated performance trends chart: {chart_file}")
    return chart_file


def generate_performance_trends_csv(
    chip_data, concurrencies, output_dir, test_suite, chip_names
):
    chip_suffix = "_vs_".join([c.lower() for c in chip_names])
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

    chip_names = list(CHIP_BASE_PATHS.keys())

    csv_lines = []
    header = ["Metric"] + [
        f"{chip}-{conc}" for chip in chip_names for conc in concurrencies
    ]
    csv_lines.append(",".join(header))

    for display_name, key_name in metric_names:
        if not key_name:
            csv_lines.append(
                f"[{display_name}]" + ",," * (len(concurrencies) * len(chip_names) - 1)
            )
            continue

        row = [display_name]
        for conc in concurrencies:
            for chip in chip_names:
                value = chip_data.get(chip, {}).get(conc, {}).get(key_name, "")
                row.append(value)
        csv_lines.append(",".join(row))

    csv_file = os.path.join(
        output_dir, f"performance_trends_{test_suite}_{chip_suffix}.csv"
    )
    with open(csv_file, "w", encoding="utf-8") as f:
        f.write("\n".join(csv_lines))

    print(f"Generated performance trends CSV: {csv_file}")
    return csv_file


def generate_analysis_content(chip_data, chip_names, concurrencies):
    analysis_lines = []

    throughputs_by_conc = {}
    for conc in concurrencies:
        throughputs = [
            (
                name,
                float(
                    chip_data.get(name, {})
                    .get(conc, {})
                    .get("Request throughput (req/s)", 0)
                    or 0
                ),
            )
            for name in chip_names
        ]
        throughputs_by_conc[conc] = throughputs

    max_throughput_chip = None
    max_throughput_value = 0
    for conc, tps in throughputs_by_conc.items():
        for chip, tp in tps:
            if tp > max_throughput_value:
                max_throughput_value = tp
                max_throughput_chip = chip

    num_conc = len(concurrencies)
    if num_conc >= 3:
        third = num_conc // 3
        low_conc = concurrencies[:third]
        mid_conc = concurrencies[third : 2 * third]
        high_conc = concurrencies[2 * third :]
    elif num_conc == 2:
        low_conc = concurrencies[:1]
        mid_conc = concurrencies[1:]
        high_conc = []
    else:
        low_conc = concurrencies
        mid_conc = []
        high_conc = []

    def get_avg_throughput(chip_name, conc_list):
        if not conc_list:
            return 0
        vals = [throughputs_by_conc.get(c, []) for c in conc_list]
        chip_vals = []
        for v in vals:
            for chip, tp in v:
                if chip == chip_name:
                    chip_vals.append(tp)
        return sum(chip_vals) / len(chip_vals) if chip_vals else 0

    avg_low = {chip: get_avg_throughput(chip, low_conc) for chip in chip_names}
    avg_mid = {chip: get_avg_throughput(chip, mid_conc) for chip in chip_names}
    avg_high = {chip: get_avg_throughput(chip, high_conc) for chip in chip_names}

    best_low = max(avg_low.items(), key=lambda x: x[1])
    best_mid = max(avg_mid.items(), key=lambda x: x[1])
    best_high = max(avg_high.items(), key=lambda x: x[1])

    low_range = f"{min(low_conc)}-{max(low_conc)}" if low_conc else "N/A"
    mid_range = f"{min(mid_conc)}-{max(mid_conc)}" if mid_conc else "N/A"
    high_range = f"{min(high_conc)}-{max(high_conc)}" if high_conc else "N/A"

    analysis_lines.append("### 1. 吞吐量性能对比\n")
    if low_conc:
        analysis_lines.append(
            f"**请求吞吐量 (QPS)**: 在低并发({low_range})场景下，{best_low[0]} 表现最佳，平均 {best_low[1]:.2f} req/s；"
        )
    if mid_conc:
        analysis_lines.append(
            f"在中并发({mid_range})场景下，{best_mid[0]} 表现最佳，平均 {best_mid[1]:.2f} req/s；"
        )
    if high_conc:
        analysis_lines.append(
            f"在高并发({high_range})场景下，{best_high[0]} 表现最佳，平均 {best_high[1]:.2f} req/s。\n"
        )

    total_tputs_by_conc = {}
    for conc in concurrencies:
        tputs = [
            (
                name,
                float(
                    chip_data.get(name, {})
                    .get(conc, {})
                    .get("Total token throughput (tok/s)", 0)
                    or 0
                ),
            )
            for name in chip_names
        ]
        total_tputs_by_conc[conc] = tputs

    best_total_tput_chip = None
    best_total_tput_value = 0
    for conc, tts in total_tputs_by_conc.items():
        for chip, tt in tts:
            if tt > best_total_tput_value:
                best_total_tput_value = tt
                best_total_tput_chip = chip

    analysis_lines.append(
        f"**Token吞吐量**: {best_total_tput_chip} 在{max(concurrencies)}并发时达到最高吞吐量 {best_total_tput_value:.0f} tok/s。\n"
    )

    ttft_by_conc = {}
    for conc in concurrencies:
        ttft_vals = [
            (
                name,
                float(
                    chip_data.get(name, {})
                    .get(conc, {})
                    .get("P99 TTFT (ms)", float("inf"))
                    or float("inf")
                ),
            )
            for name in chip_names
        ]
        ttft_by_conc[conc] = ttft_vals

    avg_ttft_low = {}
    for chip in chip_names:
        vals = []
        for c in low_conc:
            for name, val in ttft_by_conc.get(c, []):
                if name == chip and val < float("inf"):
                    vals.append(val)
        avg_ttft_low[chip] = sum(vals) / len(vals) if vals else float("inf")

    avg_ttft_high = {}
    for chip in chip_names:
        vals = []
        for c in high_conc:
            for name, val in ttft_by_conc.get(c, []):
                if name == chip and val < float("inf"):
                    vals.append(val)
        avg_ttft_high[chip] = sum(vals) / len(vals) if vals else float("inf")

    best_ttft_low = min(avg_ttft_low.items(), key=lambda x: x[1])
    best_ttft_high = min(avg_ttft_high.items(), key=lambda x: x[1])

    ttft_low_range = f"{min(low_conc)}-{max(low_conc)}" if low_conc else "N/A"
    ttft_high_range = f"{min(high_conc)}-{max(high_conc)}" if high_conc else "N/A"

    analysis_lines.append("### 2. 首Token延迟 (TTFT) 对比\n")
    if low_conc:
        analysis_lines.append(
            f"**低并发({ttft_low_range})**: {best_ttft_low[0]} TTFT最优，平均 {best_ttft_low[1]:.0f}ms\n"
        )
    if high_conc:
        analysis_lines.append(
            f"**高并发({ttft_high_range})**: {best_ttft_high[0]} TTFT最优，平均 {best_ttft_high[1]:.0f}ms\n"
        )

    if (
        avg_ttft_high.get("NVIDIA_H100", 0) > 0
        and avg_ttft_high.get("Kunlun_P800", 0) > 0
    ):
        ratio = avg_ttft_high.get("Hygon_BW1000", 1) / max(
            avg_ttft_high.get("NVIDIA_H100", 1), 1
        )
        if ratio > 10:
            analysis_lines.append(
                f"⚠️ **注意**: 海光芯片在高并发下TTFT延迟显著高于其他芯片，约为NVIDIA的{ratio:.0f}倍\n"
            )

    tpot_by_conc = {}
    for conc in concurrencies:
        tpot_vals = [
            (
                name,
                float(
                    chip_data.get(name, {})
                    .get(conc, {})
                    .get("P99 TPOT (ms)", float("inf"))
                    or float("inf")
                ),
            )
            for name in chip_names
        ]
        tpot_by_conc[conc] = tpot_vals

    best_tpot_chip = None
    best_tpot_value = float("inf")
    best_tpot_conc_val = None
    for conc, tps in tpot_by_conc.items():
        for chip, tp in tps:
            if tp < best_tpot_value and tp > 0:
                best_tpot_value = tp
                best_tpot_chip = chip
                best_tpot_conc_val = conc

    analysis_lines.append("### 3. Token生成时间 (TPOT) 对比\n")
    best_conc = best_tpot_conc_val if best_tpot_conc_val else max(concurrencies)
    analysis_lines.append(
        f"**最优表现**: {best_tpot_chip} 在各并发下TPOT表现最佳，{best_conc}并发时仅为 {best_tpot_value:.2f}ms\n"
    )

    try:
        max_conc = max(concurrencies)
        hygon_val = float(
            chip_data.get("Hygon_BW1000", {}).get(max_conc, {}).get("P99 TPOT (ms)", 0)
            or 0
        )
        nvidia_val = float(
            chip_data.get("NVIDIA_H100", {}).get(max_conc, {}).get("P99 TPOT (ms)", 0)
            or 0
        )
        if hygon_val > 0 and nvidia_val > 0:
            ratio_tpot = hygon_val / nvidia_val
            if ratio_tpot > 5:
                analysis_lines.append(
                    f"⚠️ **注意**: 海光芯片TPOT延迟明显高于NVIDIA，约为 {ratio_tpot:.1f} 倍\n"
                )
    except:
        pass

    analysis_lines.append("### 4. 综合评估\n")

    scores = {chip: 0 for chip in chip_names}

    for conc in concurrencies:
        tps = throughputs_by_conc.get(conc, [])
        if tps:
            max_tp = max(tps, key=lambda x: x[1])[1]
            for chip, tp in tps:
                if max_tp > 0:
                    scores[chip] += tp / max_tp

    best_overall = max(scores.items(), key=lambda x: x[1])
    analysis_lines.append(
        f"**综合性能**: {best_overall[0]} 在所有测试场景中综合表现最优\n"
    )

    conclusion = "\n".join(analysis_lines)
    return conclusion


def generate_markdown_report(
    chip_data,
    concurrencies,
    output_dir,
    test_suite,
    scenarios_config,
    chip_names=None,
    model_display=None,
):
    current_date = datetime.now().strftime("%Y-%m-%d")
    if chip_names is None:
        chip_names = list(CHIP_BASE_PATHS.keys())

    if model_display is None:
        model_display = MODEL_NAME

    chip_suffix = "_vs_".join([c.lower() for c in chip_names])

    chip_config = load_chip_config()
    vllm_config = load_vllm_config()

    chips_raw = chip_config.get("chips", {})
    chips_info = {}
    for chip_name, configs in chips_raw.items():
        if isinstance(configs, list) and configs:
            chips_info[chip_name] = configs[0]
        else:
            chips_info[chip_name] = configs or {}

    vllm_configs = vllm_config.get("vllm_configs", {})

    base_config = scenarios_config.get("base_config", {})
    test_params = get_test_suite_config(test_suite, scenarios_config)
    dataset_name = test_params.get("dataset-name", "random")
    max_concurrency = test_params.get("max-concurrency", [])
    num_prompts = test_params.get("num-prompts", [])

    input_output_lens = test_params.get("random-input-output-len", [])
    if (
        input_output_lens
        and isinstance(input_output_lens, list)
        and len(input_output_lens) > 0
    ):
        first_pair = input_output_lens[0]
        if isinstance(first_pair, list) and len(first_pair) >= 2:
            input_len = first_pair[0]
            output_len = first_pair[1]
        else:
            input_len = first_pair if first_pair else 0
            output_len = 0
    else:
        input_len = test_params.get("random-input-len", [0])
        output_len = test_params.get("random-output-len", [0])
        input_len = input_len[0] if isinstance(input_len, list) else input_len
        output_len = output_len[0] if isinstance(output_len, list) else output_len

    models_config = scenarios_config.get("models", {})
    chip_key_map = {
        "Hygon_BW1000": "hygon_bw1000",
        "Kunlun_P800": "kunlun_p800",
        "NVIDIA_H100": "nvidia_h100",
    }

    def format_tokens(val):
        try:
            v = int(val)
            if v >= 1024:
                return f"{v // 1024}k"
            else:
                return f"{v / 1024:.2f}k"
        except:
            return str(val)

    def make_table_for_concurrency(
        conc, key_name, highlight_max=False, highlight_min=False
    ):
        values = []
        for chip in chip_names:
            value = chip_data.get(chip, {}).get(conc, {}).get(key_name, "")
            if value == "" or value is None:
                value = "0"
            values.append(value)

        if highlight_max or highlight_min:
            try:
                numeric = [(i, float(v)) for i, v in enumerate(values) if v]
                if numeric:
                    if highlight_max:
                        best_idx = max(numeric, key=lambda x: x[1])[0]
                    else:
                        best_idx = min(numeric, key=lambda x: x[1])[0]
                    for i in range(len(values)):
                        if i == best_idx and values[i]:
                            values[i] = f"**{values[i]}** ⭐"
            except:
                pass
        return " | ".join(values)

    serving_metrics = [
        ("成功请求数", "Successful requests", False, False),
        ("失败请求数", "Failed requests", False, False),
        ("测试持续时间 (s)", "Benchmark duration (s)", False, False),
        ("总输入 tokens", "Total input tokens", False, False),
        ("总生成 tokens", "Total generated tokens", False, False),
        ("**请求吞吐量 (req/s)**", "Request throughput (req/s)", True, False),
        (
            "**输出 token 吞吐量 (tok/s)**",
            "Output token throughput (tok/s)",
            True,
            False,
        ),
        (
            "峰值输出 token 吞吐量 (tok/s)",
            "Peak output token throughput (tok/s)",
            True,
            False,
        ),
        ("峰值并发请求数", "Peak concurrent requests", False, False),
        ("**总 token 吞吐量 (tok/s)**", "Total token throughput (tok/s)", True, False),
    ]

    ttft_metrics = [
        ("平均 TTFT (ms)", "Mean TTFT (ms)", False, True),
        ("中位 TTFT (ms)", "Median TTFT (ms)", False, True),
        ("P95 TTFT (ms)", "P95 TTFT (ms)", False, True),
        ("P99 TTFT (ms)", "P99 TTFT (ms)", False, True),
    ]

    tpot_metrics = [
        ("平均 TPOT (ms)", "Mean TPOT (ms)", False, True),
        ("中位 TPOT (ms)", "Median TPOT (ms)", False, True),
        ("P95 TPOT (ms)", "P95 TPOT (ms)", False, True),
        ("P99 TPOT (ms)", "P99 TPOT (ms)", False, True),
    ]

    itl_metrics = [
        ("平均 ITL (ms)", "Mean ITL (ms)", False, True),
        ("中位 ITL (ms)", "Median ITL (ms)", False, True),
        ("P95 ITL (ms)", "P95 ITL (ms)", False, True),
        ("P99 ITL (ms)", "P99 ITL (ms)", False, True),
    ]

    concurrency_tables = ""

    for conc in concurrencies:
        header = " | ".join(chip_names)
        separator = " | ".join(["-----------"] * len(chip_names))

        serving_rows = "\n".join(
            [
                f"| {name} | {make_table_for_concurrency(conc, key, hmax, hmin)} |"
                for name, key, hmax, hmin in serving_metrics
            ]
        )

        ttft_rows = "\n".join(
            [
                f"| {name} | {make_table_for_concurrency(conc, key, hmax, hmin)} |"
                for name, key, hmax, hmin in ttft_metrics
            ]
        )

        tpot_rows = "\n".join(
            [
                f"| {name} | {make_table_for_concurrency(conc, key, hmax, hmin)} |"
                for name, key, hmax, hmin in tpot_metrics
            ]
        )

        itl_rows = "\n".join(
            [
                f"| {name} | {make_table_for_concurrency(conc, key, hmax, hmin)} |"
                for name, key, hmax, hmin in itl_metrics
            ]
        )

        concurrency_tables += f"""
### {conc} 并发

#### 服务基准结果

| 指标 | {header} |
|------|{separator}|
{serving_rows}

#### 首Token延迟 (TTFT)

| 指标 | {header} |
|------|{separator}|
{ttft_rows}

#### 每Token生成时间 (TPOT)

| 指标 | {header} |
|------|{separator}|
{tpot_rows}

#### Token间延迟 (ITL)

| 指标 | {header} |
|------|{separator}|
{itl_rows}

---
"""

    throughput_table_rows = []
    ttft_table_rows = []
    tpot_table_rows = []
    total_tput_table_rows = []

    for conc in concurrencies:
        throughputs = [
            (
                name,
                float(
                    chip_data.get(name, {})
                    .get(conc, {})
                    .get("Request throughput (req/s)", 0)
                    or 0
                ),
            )
            for name in chip_names
        ]
        if throughputs and any(t[1] > 0 for t in throughputs):
            max_tp = max(throughputs, key=lambda x: x[1])
            throughput_table_rows.append(
                f"| {conc} | {max_tp[0]} | {max_tp[1]:.2f} req/s |"
            )

    for conc in concurrencies:
        ttft_p99 = [
            (
                name,
                float(
                    chip_data.get(name, {})
                    .get(conc, {})
                    .get("P99 TTFT (ms)", float("inf"))
                    or float("inf")
                ),
            )
            for name in chip_names
        ]
        if ttft_p99 and any(t[1] < float("inf") for t in ttft_p99):
            min_ttft = min(ttft_p99, key=lambda x: x[1])
            ttft_table_rows.append(f"| {conc} | {min_ttft[0]} | {min_ttft[1]:.2f} ms |")

    for conc in concurrencies:
        tpot_p99 = [
            (
                name,
                float(
                    chip_data.get(name, {})
                    .get(conc, {})
                    .get("P99 TPOT (ms)", float("inf"))
                    or float("inf")
                ),
            )
            for name in chip_names
        ]
        if tpot_p99 and any(t[1] < float("inf") for t in tpot_p99):
            min_tpot = min(tpot_p99, key=lambda x: x[1])
            tpot_table_rows.append(f"| {conc} | {min_tpot[0]} | {min_tpot[1]:.2f} ms |")

    for conc in concurrencies:
        total_tputs = [
            (
                name,
                float(
                    chip_data.get(name, {})
                    .get(conc, {})
                    .get("Total token throughput (tok/s)", 0)
                    or 0
                ),
            )
            for name in chip_names
        ]
        if total_tputs and any(t[1] > 0 for t in total_tputs):
            max_tt = max(total_tputs, key=lambda x: x[1])
            total_tput_table_rows.append(
                f"| {conc} | {max_tt[0]} | {max_tt[1]:.0f} tok/s |"
            )

    analysis_content = generate_analysis_content(chip_data, chip_names, concurrencies)

    metric_trends_section = ""

    for display_name, key_name in COMPARISON_METRICS:
        is_throughput = key_name in [
            "Request throughput (req/s)",
            "Output token throughput (tok/s)",
            "Total token throughput (tok/s)",
        ]
        find_max = is_throughput

        chip_header = " | ".join(chip_names)
        chip_separator = " | ".join(["-----------"] * len(chip_names))
        header = f"{chip_header} | 差值 | 百分比"
        separator = f"{chip_separator} | ----------- | -----------"

        metric_rows = []
        for conc in concurrencies:
            base_value = None
            values = []
            best_chip = None
            for chip in chip_names:
                value = chip_data.get(chip, {}).get(conc, {}).get(key_name, "")
                if value == "" or value is None:
                    value = "N/A"
                values.append(value)

                if chip == chip_names[0] and value != "N/A":
                    try:
                        base_value = float(value)
                    except:
                        base_value = None

            if len(chip_names) > 1 and base_value is not None and base_value != 0:
                last_value_str = values[-1] if values[-1] != "N/A" else None
                if last_value_str:
                    try:
                        last_value = float(last_value_str)
                        diff = last_value - base_value
                        pct = (diff / base_value) * 100
                        diff_str = f"+{diff:.2f}" if diff >= 0 else f"{diff:.2f}"
                        pct_str = f"+{pct:.1f}%" if pct >= 0 else f"{pct:.1f}%"
                        extra_cols = f" | {diff_str} | {pct_str}"
                    except:
                        extra_cols = " | N/A | N/A"
                else:
                    extra_cols = " | N/A | N/A"
            else:
                extra_cols = " | N/A | N/A"

            metric_rows.append(f"| {conc}   | {' | '.join(values)}{extra_cols} |")

        metric_trends_section += f"""
#### {display_name}

| 并发数 | {header} |
|-----|{separator}|
{chr(10).join(metric_rows)}

"""

    chart_images = "\n".join(
        [
            f'\n**{conc}并发**\n\n<img src="./chip_comparison_c{conc}_{test_suite}_{chip_suffix}.png" width="1000" />'
            for conc in concurrencies
        ]
    )

    performance_trends_img = f'<img src="./performance_trends_{test_suite}_{chip_suffix}.png" width="1000" />'

    chip_info_map = load_chip_config_by_model()
    chip_table_rows = []
    all_params = set()
    for chip in chip_names:
        cfg = chip_info_map.get(chip, {})
        all_params.update(cfg.keys())
    all_params = sorted([p for p in all_params if p != "remark"])

    for param in all_params:
        row = f"| **{param}** |"
        for chip in chip_names:
            chip_info = chip_info_map.get(chip, {})
            val = chip_info.get(param, "N/A")
            row += f" {val} |"
        chip_table_rows.append(row)
    chip_table = "\n".join(chip_table_rows)

    vllm_info_map = load_vllm_config_by_model()
    all_sglang_params = set()
    for chip in chip_names:
        cfg = vllm_info_map.get(chip, {})
        all_sglang_params.update(cfg.keys())
    all_sglang_params = sorted([p for p in all_sglang_params if p != "remarks"])

    param_rows = []
    for param in all_sglang_params:
        display_name = param.replace("-", " ").replace("_", " ").title()
        row = f"| **{display_name}** |"
        for chip_name in chip_names:
            cfg = vllm_info_map.get(chip_name, {})
            val = cfg.get(param, "N/A")
            row += f" {val} |"
        param_rows.append(row)
    vllm_table = "\n".join(param_rows)

    remarks = []
    for chip_name in chip_names:
        cfg = vllm_info_map.get(chip_name, {})
        remark = cfg.get("remarks", "")
        if remark:
            remarks.append(f"- **{chip_name}**: {remark}")
    remarks_section = "\n".join(remarks) if remarks else ""

    concurrency_str = (
        ", ".join(str(c) for c in max_concurrency)
        if max_concurrency
        else ", ".join(concurrencies)
    )
    input_ctx = format_tokens(input_len) if input_len else "N/A"
    output_ctx = format_tokens(output_len) if output_len else "N/A"
    total_requests = num_prompts[0] if num_prompts else "N/A"
    input_len_val = input_len if input_len else "N/A"
    output_len_val = output_len if output_len else "N/A"

    md_content = f"""# {MODEL_NAME}模型在不同芯片下的benchmark基准测试报告

<div align="center">
**测试日期：** {current_date}

</div>

---

## 测试场景
在固定请求数，输入上下文和输出上下文长度下，使用vllm bench serve工具对并发数逐级增加场景的性能基准验证。并对比同一模型在不同芯片环境上的性能指标。

**主要采集指标**：

| 指标                  | 单位         | 含义                                 |
|---------------------|------------|------------------------------------|
| TTFT                | ms         | Time To First Token，首 token 延迟     |
| TPOT                | ms/token   | Time Per Output Token，每 token 生成时间 |
| Throughput          | tokens/s   | 系统总吞吐                              |
| QPS                 | requests/s | 请求吞吐                               |
| P50/P95/P99 Latency | ms         | 延迟分位数                              |
    
### 📊 测试概览

| 项目            | 配置                                     | 备注  |
|---------------|----------------------------------------|-----|
| **数据集**       | {dataset_name}                                 |     |
| **并发数**       | {concurrency_str}    |     |
| **总请求数**      | {total_requests}                                    |     |
| **请求输入上下文长度** | {input_len_val}（{input_ctx}）                             |     |
| **请求输出上下文长度** | {output_len_val}（{output_ctx}）                             |     |
| **被测芯片**      | {", ".join(chip_names)} |     |
| **被测模型**      | {MODEL_NAME} |     |

---

### 🤖 芯片和模型配置信息

| 参数名称 | **{"** | **".join(chip_names)}** |
|----------|{"----------|" * len(chip_names)}
{chip_table}

---

### ⚙️ vLLM启动配置信息

| 参数名称 | **{"** | **".join(chip_names)}** |
|----------|{"----------|" * len(chip_names)}
{vllm_table}

{remarks_section}

---

### 📊 芯片性能对比柱状图

{chart_images}


### 📈 性能趋势对比图 (所有芯片)

{performance_trends_img}

---

### 📈 各指标随并发级别性能对比详情

{metric_trends_section}

### 📈 各并发级别性能对比详情

{concurrency_tables}

---

<div align="center">
*报告生成时间: {current_date}*
</div>
"""

    md_file = os.path.join(
        output_dir, f"{model_display}_chip_comparison_{test_suite}_{chip_suffix}.md"
    )
    with open(md_file, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"Generated: {md_file}")
    return md_file


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate chip comparison report")
    parser.add_argument(
        "--chip",
        type=str,
        default=None,
        help="Chip names to compare, comma-separated (e.g., hygon_bw1000,nvidia_h100)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model names for each chip, comma-separated and must match chip order (e.g., MiniMax-M2.5-bf16,MiniMax-M2.5)",
    )
    parser.add_argument(
        "--test-suite", type=str, default=None, help="Test suite name (e.g., test_01)"
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Run IDs, can be '01' for all chips or '01,02,03' for each chip",
    )
    parser.add_argument(
        "--concurrency",
        "-c",
        type=str,
        default=None,
        help="Specific concurrency levels to compare, comma-separated (e.g., 1,2,4,8,10)",
    )
    args = parser.parse_args()

    scenarios_config = load_models_scenarios()
    models = scenarios_config.get("models", {})
    chip_key_map = {
        "hygon_bw1000": "hygon_bw1000",
        "kunlun_p800": "kunlun_p800",
        "nvidia_h100": "nvidia_h100",
    }
    chip_key_map_reverse = {
        "hygon_bw1000": "Hygon_BW1000",
        "kunlun_p800": "Kunlun_P800",
        "nvidia_h100": "NVIDIA_H100",
    }

    if args.chip:
        chip_list = [s.strip() for s in args.chip.split(",")]
        chip_list = [s.lower() for s in chip_list]
        chip_names = [chip_key_map_reverse.get(s, s) for s in chip_list]
    else:
        chip_names = list(chip_key_map_reverse.values())

    num_chips = len(chip_names)

    # 处理模型参数 - 支持多个模型，与芯片一一对应
    if args.model:
        model_input_list = [s.strip() for s in args.model.split(",")]
        model_key_map = {
            "minimax-m2.5": "MiniMax-M2.5",
            "qwen3.5": "Qwen3.5",
        }
        models_to_use = [model_key_map.get(m.lower(), m) for m in model_input_list]

        # 如果只指定一个模型，所有芯片使用相同模型
        if len(models_to_use) == 1:
            models_to_use = [models_to_use[0]] * num_chips
            print(f"Using model '{models_to_use[0]}' for all {num_chips} chips")
        elif len(models_to_use) != num_chips:
            print(
                f"Error: Number of models ({len(models_to_use)}) must match number of chips ({num_chips})"
            )
            print(f"Chips: {', '.join(chip_names)}")
            print(f"Models: {', '.join(models_to_use)}")
            return
    else:
        # 使用默认模型
        model_to_use = MODEL_NAME
        models_to_use = [model_to_use] * num_chips
        print(f"No model specified, using default: {model_to_use} for all chips")

    test_suite_input = args.test_suite.strip() if args.test_suite else TEST_SUITES[0]
    test_suite_to_use = (
        test_suite_input.lower() if test_suite_input else test_suite_input
    )

    run_ids_input = args.run_id.strip() if args.run_id else None
    if run_ids_input:
        run_ids_list = [s.strip() for s in run_ids_input.split(",")]
        if len(run_ids_list) == 1:
            run_id_to_use = [run_ids_list[0]] * num_chips
        elif len(run_ids_list) == num_chips:
            run_id_to_use = run_ids_list
        else:
            print(
                f"Error: Number of run-ids ({len(run_ids_list)}) must match number of chips ({num_chips}) or be 1"
            )
            return
    else:
        run_id_to_use = [RUN_IDS[0]] * num_chips

    # 验证每个芯片是否有对应的模型报告文件
    chip_base_paths = {}
    chip_key_map_for_path = {
        "Hygon_BW1000": "hygon_bw1000",
        "Kunlun_P800": "kunlun_p800",
        "NVIDIA_H100": "nvidia_h100",
    }

    for i, chip_name in enumerate(chip_names):
        model_name = models_to_use[i]
        chip_key = chip_key_map_for_path.get(chip_name, chip_name.lower())

        # 动态构建 base path
        base = f"reports/benchmark/{chip_key}/{model_name}"

        # 检查模型目录是否存在
        if not os.path.exists(base):
            print(f"Error: Model '{model_name}' not found for chip '{chip_name}'")
            print(f"Expected path: {base}")
            print(f"Available models for {chip_name}:")
            # 列出 reports 目录下的模型
            reports_chip_path = f"reports/benchmark/{chip_key}"
            if os.path.exists(reports_chip_path):
                available_models = [
                    d
                    for d in os.listdir(reports_chip_path)
                    if os.path.isdir(os.path.join(reports_chip_path, d))
                ]
                for m in available_models:
                    print(f"  - {m}")
            else:
                print(f"  No reports found for {chip_name}")
            return

        # 检查测试套件目录是否存在
        test_path = f"{base}/{test_suite_to_use}"
        if not os.path.exists(test_path):
            print(
                f"Error: Test suite '{test_suite_to_use}' not found for chip '{chip_name}' model '{model_name}'"
            )
            print(f"Expected path: {test_path}")
            return

        # 检查 run-id 目录是否存在（只检查当前芯片对应的 run-id）
        rid = run_id_to_use[i]
        runid_path = f"{test_path}/{rid}"
        if not os.path.exists(runid_path):
            print(
                f"Error: Run-id '{rid}' not found for chip '{chip_name}' model '{model_name}' test-suite '{test_suite_to_use}'"
            )
            print(f"Expected path: {runid_path}")
            # 列出可用的 run-id
            if os.path.exists(test_path):
                available_runids = [
                    d
                    for d in os.listdir(test_path)
                    if os.path.isdir(os.path.join(test_path, d))
                ]
                print(
                    f"Available run-ids: {', '.join(available_runids) if available_runids else 'None'}"
                )
            return

        chip_base_paths[chip_name] = base
        print(
            f"Validated: {chip_name} - {model_name} - {test_suite_to_use} - run-id {rid}"
        )

    print(f"\nComparing {num_chips} chips with {len(set(models_to_use))} model(s)")

    def get_chip_configs_for_chips(chip_names, test_suite, chip_run_ids, chip_models):
        configs = []
        for i, chip_name in enumerate(chip_names):
            base_path = chip_base_paths.get(chip_name, "")
            run_id = chip_run_ids[i] if i < len(chip_run_ids) else chip_run_ids[0]
            model_name = chip_models[i] if i < len(chip_models) else chip_models[0]
            if base_path:
                configs.append(
                    {
                        "name": chip_name,
                        "model": model_name,
                        "base_path": f"{base_path}/{test_suite}/{run_id}",
                    }
                )
        return configs

    for test_suite in [test_suite_to_use]:
        print(f"\n{'#' * 60}")
        print(f"Processing test suite: {test_suite}")
        print(f"Chips: {', '.join(chip_names)}")
        print(f"Models: {', '.join(models_to_use)}")
        print(f"Run IDs: {', '.join(run_id_to_use)}")
        print(f"{'#' * 60}\n")

        chip_configs = get_chip_configs_for_chips(
            chip_names, test_suite, run_id_to_use, models_to_use
        )
        run_id_display = "_".join(run_id_to_use)

        # 使用模型名称组合作为输出目录名
        # 提取公共前缀来简化目录名
        def get_common_prefix(names):
            if not names:
                return ""
            if len(names) == 1:
                return names[0]
            prefix = names[0]
            for name in names[1:]:
                while not name.startswith(prefix):
                    prefix = prefix[:-1]
                    if not prefix:
                        break
            return prefix

        common_prefix = get_common_prefix(models_to_use)

        if len(set(models_to_use)) > 1 and common_prefix and len(common_prefix) > 3:
            # 有公共前缀且长度大于3，使用公共前缀
            model_display = common_prefix
        else:
            # 没有足够长的公共前缀或有重复模型，使用完整名称
            model_display = "_vs_".join(models_to_use)

        output_base = (
            f"analysis/chip_comparison/{model_display}/{test_suite}/{run_id_display}"
        )
        Path(output_base).mkdir(parents=True, exist_ok=True)

        all_concurrencies = set()
        for chip in chip_configs:
            concs = get_all_concurrencies(chip)
            all_concurrencies.update(concs)

        if not all_concurrencies:
            print(f"No concurrency configurations found for {test_suite}!")
            continue

        concurrencies = sorted(all_concurrencies, key=lambda x: int(x))

        if args.concurrency:
            conc_list = [s.strip() for s in args.concurrency.split(",")]
            filtered_concs = [c for c in concurrencies if c in conc_list]
            if filtered_concs:
                concurrencies = filtered_concs
                print(f"Using specified concurrency levels: {', '.join(concurrencies)}")
            else:
                print(
                    f"Warning: None of the specified concurrency levels {conc_list} found, using all"
                )

        print(
            f"Found {len(concurrencies)} concurrency levels: {', '.join(concurrencies)}"
        )

        chip_data = defaultdict(lambda: defaultdict(dict))

        for chip in chip_configs:
            chip_name = chip["name"]
            print(f"\nProcessing chip: {chip_name}")

            for conc in concurrencies:
                metrics = get_chip_metrics(chip, conc)
                if metrics:
                    chip_data[chip_name][conc] = metrics
                    print(f"  - {conc}并发: OK")
                else:
                    print(f"  - {conc}并发: No data")

        print("\nGenerating comparison reports...")

        generate_comparison_csv(
            chip_data, concurrencies, output_base, test_suite_to_use, chip_names
        )

        if HAS_MATPLOTLIB:
            generate_comparison_charts(
                chip_data, concurrencies, output_base, test_suite_to_use, chip_names
            )
            generate_performance_trends(
                chip_data, concurrencies, output_base, test_suite_to_use, chip_names
            )

        generate_performance_trends_csv(
            chip_data, concurrencies, output_base, test_suite_to_use, chip_names
        )

        generate_markdown_report(
            chip_data,
            concurrencies,
            output_base,
            test_suite_to_use,
            scenarios_config,
            chip_names,
            model_display,
        )

        print(f"\n{'=' * 50}")
        print(f"Chip comparison for {test_suite} generated successfully!")
        print(f"Output directory: {output_base}")
        print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
