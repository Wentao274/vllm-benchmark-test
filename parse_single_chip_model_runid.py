import os
import re
import glob
import yaml
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None
    print("matplotlib not available, skipping chart generation")


TEST_SUITES = ["test_03"]

RUN_IDS = ["01", "02"]

CHIP_BASE_PATHS = {
    "Hygon_BW1000": "reports/hygon_bw1000/benchmark/MiniMax-M2.5-bf16",
}

MODEL_NAME = "MiniMax-M2.5"


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


def load_chip_config_by_model(chip_name, model_name):
    chip_config = load_chip_config()
    chips_raw = chip_config.get("chips", {})

    base_path = CHIP_BASE_PATHS.get(chip_name, "")
    if base_path:
        model_name = Path(base_path).name

    chip_configs = chips_raw.get(chip_name, [])
    if isinstance(chip_configs, list):
        for cfg in chip_configs:
            if cfg.get("model_name") == model_name:
                return cfg
        return chip_configs[0] if chip_configs else {}
    elif isinstance(chip_configs, dict):
        return chip_configs
    return {}


def load_vllm_config_by_model(chip_name, model_name):
    vllm_config = load_vllm_config()
    vllm_configs_raw = vllm_config.get("vllm_configs", {})

    base_path = CHIP_BASE_PATHS.get(chip_name, "")
    if base_path:
        model_name = Path(base_path).name

    config_list = vllm_configs_raw.get(chip_name, [])
    if isinstance(config_list, list):
        for cfg in config_list:
            if cfg.get("model_name") == model_name:
                return cfg
        return config_list[0] if config_list else {}
    elif isinstance(config_list, dict):
        return config_list
    return {}


VLLM_CONFIG = {
    "max-model-len": {"01": "196608", "02": "196608"},
    "max-num-seqs": {"01": "64", "02": "64"},
    "max-num-batched-tokens": {"01": "8192", "02": "N/A"},
    "gpu-memory-utilization": {"01": "0.95", "02": "0.9"},
    "dp": {"01": "1", "02": "1"},
    "tp": {"01": "8", "02": "8"},
    "pp": {"01": "1", "02": "1"},
    "enable-export-parallel": {"01": "False", "02": "N/A"},
    "tool-call-parser": {"01": "minimax_m2", "02": "minimax_m2"},
    "reasoning-parser": {"01": "minimax_m2", "02": "N/A"},
    "-cc": {"01": "N/A", "02": '{"pass_config": {"fuse_act_quant": false}}'},
}


def load_test_config(config_path="config/models_scenarios.yaml"):
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


def get_test_overview(test_suite):
    config = load_test_config()
    params = config.get("base_config", {}).get("params", {}).get(test_suite, {})

    input_output_lens = params.get("random-input-output-len", [])
    if (
        input_output_lens
        and isinstance(input_output_lens[0], list)
        and len(input_output_lens[0]) >= 2
    ):
        input_len = [input_output_lens[0][0]]
        output_len = [input_output_lens[0][1]]
    else:
        input_len = params.get("random-input-len", [])
        output_len = params.get("random-output-len", [])

    if params:
        return {
            "dataset": params.get("dataset-name", "random"),
            "concurrency": params.get("max-concurrency", []),
            "total_requests": params.get("num-prompts", []),
            "input_context_length": input_len,
            "output_context_length": output_len,
        }
    return {
        "dataset": "random",
        "concurrency": [],
        "total_requests": [],
        "input_context_length": [],
        "output_context_length": [],
    }


def get_chip_configs(test_suite):
    configs = []
    for chip_name, base_path in CHIP_BASE_PATHS.items():
        configs.append(
            {
                "name": chip_name,
                "base_path": f"{base_path}/{test_suite}",
            }
        )
    return configs


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

    if "Failed requests" not in metrics:
        metrics["Failed requests"] = "0"

    return metrics


def extract_concurrency_from_dir(dir_name):
    match = re.match(r"^(\d+)-", dir_name)
    if match:
        return match.group(1)
    return None


def get_all_concurrencies(base_path, run_id):
    concurrency_set = set()
    full_path = os.path.join(base_path, run_id)

    if not os.path.exists(full_path):
        return []

    for item in os.listdir(full_path):
        item_path = os.path.join(full_path, item)
        if os.path.isdir(item_path):
            conc = extract_concurrency_from_dir(item)
            if conc:
                concurrency_set.add(conc)

    return sorted(concurrency_set, key=lambda x: int(x))


def get_chip_metrics(base_path, run_id, concurrency):
    full_path = os.path.join(base_path, run_id)

    dir_pattern = os.path.join(full_path, f"{concurrency}-*")
    matching_dirs = glob.glob(dir_pattern)

    if not matching_dirs:
        return None

    log_pattern = os.path.join(matching_dirs[0], "*.log")
    log_files = glob.glob(log_pattern)

    if not log_files:
        return None

    metrics = parse_benchmark_log(log_files[0])
    return metrics


def calculate_diff(val1, val2):
    try:
        f1 = float(val1)
        f2 = float(val2)
        diff = f2 - f1
        if f1 != 0:
            pct = (diff / f1) * 100
            return diff, pct
        return diff, 0
    except:
        return None, None


def format_diff(diff, pct):
    if diff is None:
        return "N/A", "N/A"
    if pct is not None:
        sign = "+" if diff > 0 else ""
        return f"{sign}{diff:.2f}", f"{sign}{pct:.1f}%"
    return f"{diff:.2f}", "N/A"


def generate_comparison_csv(runid_data, concurrencies, output_dir, chip_name):
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

    run_id1 = RUN_IDS[0]
    run_id2 = RUN_IDS[1]

    csv_lines = []

    header_parts = ["Metric"]
    for conc in concurrencies:
        header_parts.append(f"{conc}-{run_id1}")
        header_parts.append(f"{conc}-{run_id2}")
        header_parts.append(f"{conc}-Diff")
        header_parts.append(f"{conc}-%")
    csv_lines.append(",".join(header_parts))

    for display_name, key_name in metric_names:
        if not key_name:
            csv_lines.append(f"[{display_name}]" + ",," * (len(concurrencies) * 4))
            continue

        row = [display_name]

        for conc in concurrencies:
            val1 = (
                runid_data.get(run_id1, {})
                .get(chip_name, {})
                .get(conc, {})
                .get(key_name, "")
            )
            val2 = (
                runid_data.get(run_id2, {})
                .get(chip_name, {})
                .get(conc, {})
                .get(key_name, "")
            )

            row.append(val1)
            row.append(val2)

            diff, pct = calculate_diff(val1, val2)
            diff_str, pct_str = format_diff(diff, pct)
            row.append(diff_str)
            row.append(pct_str)

        csv_lines.append(",".join(row))

    csv_file = os.path.join(output_dir, "runid_comparison.csv")
    with open(csv_file, "w", encoding="utf-8") as f:
        f.write("\n".join(csv_lines))

    print(f"Generated: {csv_file}")
    return [csv_file]


def generate_comparison_charts(runid_data, concurrencies, output_dir, chip_name):
    if not HAS_MATPLOTLIB:
        return None

    run_id1 = RUN_IDS[0]
    run_id2 = RUN_IDS[1]

    x = range(len(concurrencies))
    width = 0.35

    def get_values(run_id, key):
        values = []
        for conc in concurrencies:
            val = (
                runid_data.get(run_id, {})
                .get(chip_name, {})
                .get(conc, {})
                .get(key, "0")
            )
            try:
                values.append(float(val))
            except:
                values.append(0)
        return values

    colors = ["#3498db", "#2ecc71"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(
        f"{MODEL_NAME} on {chip_name} - Run ID Comparison ({run_id1} vs {run_id2})",
        fontsize=14,
        fontweight="bold",
    )

    req_throughput_1 = get_values(run_id1, "Request throughput (req/s)")
    req_throughput_2 = get_values(run_id2, "Request throughput (req/s)")

    axes[0, 0].bar(
        [i - width / 2 for i in x],
        req_throughput_1,
        width,
        label=run_id1,
        color=colors[0],
        alpha=0.8,
    )
    axes[0, 0].bar(
        [i + width / 2 for i in x],
        req_throughput_2,
        width,
        label=run_id2,
        color=colors[1],
        alpha=0.8,
    )
    axes[0, 0].set_title("Request Throughput (req/s)", fontsize=11)
    axes[0, 0].set_xlabel("Concurrency")
    axes[0, 0].set_ylabel("req/s")
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(concurrencies, rotation=45)
    axes[0, 0].legend()
    axes[0, 0].grid(axis="y", alpha=0.3)

    output_tput_1 = get_values(run_id1, "Output token throughput (tok/s)")
    output_tput_2 = get_values(run_id2, "Output token throughput (tok/s)")

    axes[0, 1].bar(
        [i - width / 2 for i in x],
        output_tput_1,
        width,
        label=run_id1,
        color=colors[0],
        alpha=0.8,
    )
    axes[0, 1].bar(
        [i + width / 2 for i in x],
        output_tput_2,
        width,
        label=run_id2,
        color=colors[1],
        alpha=0.8,
    )
    axes[0, 1].set_title("Output Token Throughput (tok/s)", fontsize=11)
    axes[0, 1].set_xlabel("Concurrency")
    axes[0, 1].set_ylabel("tok/s")
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(concurrencies, rotation=45)
    axes[0, 1].legend()
    axes[0, 1].grid(axis="y", alpha=0.3)

    total_tput_1 = get_values(run_id1, "Total token throughput (tok/s)")
    total_tput_2 = get_values(run_id2, "Total token throughput (tok/s)")

    axes[0, 2].bar(
        [i - width / 2 for i in x],
        total_tput_1,
        width,
        label=run_id1,
        color=colors[0],
        alpha=0.8,
    )
    axes[0, 2].bar(
        [i + width / 2 for i in x],
        total_tput_2,
        width,
        label=run_id2,
        color=colors[1],
        alpha=0.8,
    )
    axes[0, 2].set_title("Total Token Throughput (tok/s)", fontsize=11)
    axes[0, 2].set_xlabel("Concurrency")
    axes[0, 2].set_ylabel("tok/s")
    axes[0, 2].set_xticks(x)
    axes[0, 2].set_xticklabels(concurrencies, rotation=45)
    axes[0, 2].legend()
    axes[0, 2].grid(axis="y", alpha=0.3)

    ttft_p99_1 = get_values(run_id1, "P99 TTFT (ms)")
    ttft_p99_2 = get_values(run_id2, "P99 TTFT (ms)")

    axes[1, 0].bar(
        [i - width / 2 for i in x],
        ttft_p99_1,
        width,
        label=run_id1,
        color=colors[0],
        alpha=0.8,
    )
    axes[1, 0].bar(
        [i + width / 2 for i in x],
        ttft_p99_2,
        width,
        label=run_id2,
        color=colors[1],
        alpha=0.8,
    )
    axes[1, 0].set_title("TTFT P99 (ms)", fontsize=11)
    axes[1, 0].set_xlabel("Concurrency")
    axes[1, 0].set_ylabel("ms")
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(concurrencies, rotation=45)
    axes[1, 0].legend()
    axes[1, 0].grid(axis="y", alpha=0.3)

    tpot_p99_1 = get_values(run_id1, "P99 TPOT (ms)")
    tpot_p99_2 = get_values(run_id2, "P99 TPOT (ms)")

    axes[1, 1].bar(
        [i - width / 2 for i in x],
        tpot_p99_1,
        width,
        label=run_id1,
        color=colors[0],
        alpha=0.8,
    )
    axes[1, 1].bar(
        [i + width / 2 for i in x],
        tpot_p99_2,
        width,
        label=run_id2,
        color=colors[1],
        alpha=0.8,
    )
    axes[1, 1].set_title("TPOT P99 (ms)", fontsize=11)
    axes[1, 1].set_xlabel("Concurrency")
    axes[1, 1].set_ylabel("ms")
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(concurrencies, rotation=45)
    axes[1, 1].legend()
    axes[1, 1].grid(axis="y", alpha=0.3)

    itl_p99_1 = get_values(run_id1, "P99 ITL (ms)")
    itl_p99_2 = get_values(run_id2, "P99 ITL (ms)")

    axes[1, 2].bar(
        [i - width / 2 for i in x],
        itl_p99_1,
        width,
        label=run_id1,
        color=colors[0],
        alpha=0.8,
    )
    axes[1, 2].bar(
        [i + width / 2 for i in x],
        itl_p99_2,
        width,
        label=run_id2,
        color=colors[1],
        alpha=0.8,
    )
    axes[1, 2].set_title("ITL P99 (ms)", fontsize=11)
    axes[1, 2].set_xlabel("Concurrency")
    axes[1, 2].set_ylabel("ms")
    axes[1, 2].set_xticks(x)
    axes[1, 2].set_xticklabels(concurrencies, rotation=45)
    axes[1, 2].legend()
    axes[1, 2].grid(axis="y", alpha=0.3)

    for ax in axes.flat:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.set_facecolor("#f0f0f0")
    for ax in axes.flat:
        ax.set_facecolor("white")

    plt.tight_layout()

    chart_file = os.path.join(output_dir, "runid_comparison.png")
    plt.savefig(chart_file, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Generated chart: {chart_file}")
    return [chart_file]


def generate_markdown_report(
    runid_data,
    concurrencies,
    output_dir,
    test_suite,
    chip_name,
    vllm_config=None,
    test_overview=None,
):
    current_date = datetime.now().strftime("%Y-%m-%d")
    run_id1 = RUN_IDS[0]
    run_id2 = RUN_IDS[1]

    if vllm_config is None:
        vllm_config = {}
    if test_overview is None:
        test_overview = {}

    def make_table_for_conc(conc, key_name):
        val1 = (
            runid_data.get(run_id1, {})
            .get(chip_name, {})
            .get(conc, {})
            .get(key_name, "")
        )
        val2 = (
            runid_data.get(run_id2, {})
            .get(chip_name, {})
            .get(conc, {})
            .get(key_name, "")
        )

        diff, pct = calculate_diff(val1, val2)
        diff_str, pct_str = format_diff(diff, pct)

        return val1, val2, diff_str, pct_str

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

    tables_html = ""

    for conc in concurrencies:
        header = f"| 指标 | RUN-{run_id1} | RUN-{run_id2} | 差异 | 百分比 |"
        separator = "|------|----------|---------|---------|---------|"

        serving_table = "\n".join(
            [
                f"| {name} | {' | '.join(make_table_for_conc(conc, key))} |"
                for name, key in serving_metrics
            ]
        )

        ttft_table = "\n".join(
            [
                f"| {name} | {' | '.join(make_table_for_conc(conc, key))} |"
                for name, key in ttft_metrics
            ]
        )

        tpot_table = "\n".join(
            [
                f"| {name} | {' | '.join(make_table_for_conc(conc, key))} |"
                for name, key in tpot_metrics
            ]
        )

        itl_table = "\n".join(
            [
                f"| {name} | {' | '.join(make_table_for_conc(conc, key))} |"
                for name, key in itl_metrics
            ]
        )

        tables_html += f"""
### 并发级别: {conc}

#### 服务基准结果

{header}
{separator}
{serving_table}

#### 首Token延迟 (TTFT)

{header}
{separator}
{ttft_table}

#### 每Token生成时间 (TPOT)

{header}
{separator}
{tpot_table}

#### Token间延迟 (ITL)

{header}
{separator}
{itl_table}

"""

    def calc_avg_improvement(key_name):
        improvements = []
        for conc in concurrencies:
            val1 = (
                runid_data.get(run_id1, {})
                .get(chip_name, {})
                .get(conc, {})
                .get(key_name, "")
            )
            val2 = (
                runid_data.get(run_id2, {})
                .get(chip_name, {})
                .get(conc, {})
                .get(key_name, "")
            )
            try:
                v1 = float(val1)
                v2 = float(val2)
                if v1 > 0:
                    pct = ((v2 - v1) / v1) * 100
                    improvements.append(pct)
            except:
                pass
        return sum(improvements) / len(improvements) if improvements else 0

    tp_improvement = calc_avg_improvement("Request throughput (req/s)")
    output_tp_improvement = calc_avg_improvement("Output token throughput (tok/s)")
    ttft_change = calc_avg_improvement("P99 TTFT (ms)")
    tpot_change = calc_avg_improvement("P99 TPOT (ms)")
    itl_change = calc_avg_improvement("P99 ITL (ms)")

    analysis_lines = []
    analysis_lines.append("### 吞吐量对比\n")
    if tp_improvement > 0:
        analysis_lines.append(
            f"**请求吞吐量**: RUN-{run_id2} 相比 RUN-{run_id1} 平均提升 **{tp_improvement:.1f}%**\n"
        )
    else:
        analysis_lines.append(
            f"**请求吞吐量**: RUN-{run_id2} 相比 RUN-{run_id1} 平均变化 **{tp_improvement:.1f}%**\n"
        )

    if output_tp_improvement > 0:
        analysis_lines.append(
            f"**输出Token吞吐量**: RUN-{run_id2} 相比 RUN-{run_id1} 平均提升 **{output_tp_improvement:.1f}%**\n"
        )
    else:
        analysis_lines.append(
            f"**输出Token吞吐量**: RUN-{run_id2} 相比 RUN-{run_id1} 平均变化 **{output_tp_improvement:.1f}%**\n"
        )

    analysis_lines.append("### 延迟对比\n")
    if ttft_change > 0:
        analysis_lines.append(
            f"**TTFT P99**: RUN-{run_id2} 相比 RUN-{run_id1} 平均增加 **{ttft_change:.1f}%** (延迟增加)"
        )
    else:
        analysis_lines.append(
            f"**TTFT P99**: RUN-{run_id2} 相比 RUN-{run_id1} 平均改善 **{abs(ttft_change):.1f}%** (延迟降低)"
        )

    if tpot_change > 0:
        analysis_lines.append(
            f"**TPOT P99**: RUN-{run_id2} 相比 RUN-{run_id1} 平均增加 **{tpot_change:.1f}%** (延迟增加)"
        )
    else:
        analysis_lines.append(
            f"**TPOT P99**: RUN-{run_id2} 相比 RUN-{run_id1} 平均改善 **{abs(tpot_change):.1f}%** (延迟降低)"
        )

    if itl_change > 0:
        analysis_lines.append(
            f"**ITL P99**: RUN-{run_id2} 相比 RUN-{run_id1} 平均增加 **{itl_change:.1f}%** (延迟增加)"
        )
    else:
        analysis_lines.append(
            f"**ITL P99**: RUN-{run_id2} 相比 RUN-{run_id1} 平均改善 **{abs(itl_change):.1f}%** (延迟降低)"
        )

    conclusion = "\n".join(analysis_lines)

    concurrency_comparison_img = '<img src="./runid_comparison.png" width="1000" />'

    dataset = test_overview.get("dataset", "random")
    concurrency_list = test_overview.get("concurrency", concurrencies)
    concurrency = (
        str(concurrency_list) if concurrency_list else ", ".join(concurrencies)
    )
    total_requests_list = test_overview.get("total_requests", [])
    total_requests = str(total_requests_list) if total_requests_list else "N/A"
    input_len_list = test_overview.get("input_context_length", [])
    input_ctx = str(input_len_list) if input_len_list else "N/A"
    output_len_list = test_overview.get("output_context_length", [])
    output_ctx = str(output_len_list) if output_len_list else "N/A"
    model = test_overview.get("model", MODEL_NAME)
    chip = test_overview.get("chip", chip_name)

    chip_info = load_chip_config_by_model(chip_name, MODEL_NAME)
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

    vllm_info = load_vllm_config_by_model(chip_name, MODEL_NAME)
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
        val = vllm_info.get(param, "N/A")
        vllm_table_rows.append(f"| {param} | {val} |")
    vllm_table = "\n".join(vllm_table_rows)

    md_content = f"""# {MODEL_NAME}模型在{chip_name}上的RUN-ID对比报告

<div align="center">
**测试日期：** {current_date}

**对比RUN-ID：** {run_id1} vs {run_id2}

</div>

---

## 测试场景
对比同一芯片、同一测试套件下,同一模型优化前后测试结果比对，分析性能差异。

**测试模型** <br>
第一轮测试（RUN-{run_id1}）: {model} <br>
第二轮测试（RUN-{run_id2}）: {model}


## 🤖 芯片和模型配置信息

| 参数名称                    | {chip_name} |
|------------------------|-------------|
{chip_table}

---

## 🤖 vLLM启动配置信息

| 参数名称                    | {chip_name} |
|------------------------|-------------|
{vllm_table}

---

## 📊 测试概览

| 项目            | 配置                                    | 备注  |
|---------------|---------------------------------------|-----|
| **数据集**       | {dataset}                                |     |
| **并发数**       | {concurrency} |     |
| **总请求数**      | {total_requests}                                 |     |
| **请求输入上下文长度** | {input_ctx}                               |     |
| **请求输出上下文长度** | {output_ctx}                               |     |
| **模型**        | {model}                          |     |
| **被测芯片**      | {chip}                          |     |


**主要采集指标**：

| 指标                  | 单位         | 含义                                 |
|---------------------|------------|------------------------------------|
| TTFT                | ms         | Time To First Token，首 token 延迟     |
| TPOT                | ms/token   | Time Per Output Token，每 token 生成时间 |
| Throughput          | tokens/s   | 系统总吞吐                              |
| QPS                 | requests/s | 请求吞吐                               |
| P50/P95/P99 Latency | ms         | 延迟分位数                              |

---

## 各并发级别详细对比

{tables_html}

---

## 📊 RUN-ID对比柱状图

{concurrency_comparison_img}

---

## 📝 分析总结

{conclusion}

---

<div align="center">
*报告生成时间: {current_date}*
</div>
"""

    md_file = os.path.join(
        output_dir,
        f"{MODEL_NAME}_{chip_name}_{test_suite}_runid_compare_{run_id1}vs{run_id2}.md",
    )
    with open(md_file, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"Generated: {md_file}")
    return md_file


def main():
    parser = argparse.ArgumentParser(
        description="Compare run-id performance for a chip/model"
    )
    parser.add_argument(
        "--chip",
        type=str,
        default=None,
        help="Chip name (e.g., Hygon_BW1000, Kunlun_P800)",
    )
    parser.add_argument("--model", type=str, default=None, help="Model name")
    parser.add_argument(
        "--test-suite", type=str, default=None, help="Test suite name (e.g., test_01)"
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Run IDs to compare, can be '01,02' or '01' '02' or '01','02'",
    )
    args = parser.parse_args()

    global TEST_SUITES, RUN_IDS, CHIP_BASE_PATHS, MODEL_NAME

    if args.test_suite:
        TEST_SUITES = [args.test_suite]

    if args.run_id:
        run_ids_list = []
        for part in args.run_id.replace("'", "").split(","):
            part = part.strip()
            if part:
                run_ids_list.append(part)
        RUN_IDS = run_ids_list

    if args.model:
        MODEL_NAME = args.model

    if args.chip:
        chip_key = args.chip.lower()
        chip_key_map = {
            "hygon_bw1000": "Hygon_BW1000",
            "kunlun_p800": "Kunlun_P800",
            "nvidia_h100": "NVIDIA_H100",
        }
        chip_name = chip_key_map.get(chip_key, args.chip)

        chip_base_path_map = {
            "Hygon_BW1000": "reports/hygon_bw1000/benchmark/MiniMax-M2.5-bf16",
            "Kunlun_P800": "reports/kunlun_p800/benchmark/MiniMax-M2.5-W8A8-INT8-Dynamic",
            "NVIDIA_H100": "reports/nvidia_h100/benchmark/MiniMax-M2.5",
        }

        if chip_name in chip_base_path_map:
            CHIP_BASE_PATHS = {chip_name: chip_base_path_map[chip_name]}
        else:
            print(f"Unknown chip: {chip_name}")
            return

    run_id1 = RUN_IDS[0]
    run_id2 = RUN_IDS[1]
    runid_folder = f"run_{run_id1}_{run_id2}"

    for test_suite in TEST_SUITES:
        print(f"\n{'#' * 60}")
        print(f"Processing test suite: {test_suite}")
        print(f"Comparing RUN-ID: {run_id1} vs {run_id2}")
        print(f"{'#' * 60}\n")

        chip_configs = get_chip_configs(test_suite)

        for chip in chip_configs:
            chip_name = chip["name"]
            base_path = chip["base_path"]

            output_base = f"analysis/single_chip/{chip_name}/{MODEL_NAME}/compare_run/{test_suite}/{runid_folder}"
            Path(output_base).mkdir(parents=True, exist_ok=True)

            all_concurrencies = set()
            run_id_concurrencies = {}

            for run_id in RUN_IDS:
                concs = get_all_concurrencies(base_path, run_id)
                run_id_concurrencies[run_id] = set(concs)
                all_concurrencies.update(concs)

            common_concurrencies = set()
            for run_id in RUN_IDS:
                if not common_concurrencies:
                    common_concurrencies = run_id_concurrencies[run_id].copy()
                else:
                    common_concurrencies = common_concurrencies.intersection(
                        run_id_concurrencies[run_id]
                    )

            if not all_concurrencies:
                print(
                    f"No concurrency configurations found for {chip_name} / {test_suite}!"
                )
                continue

            concurrencies = sorted(common_concurrencies, key=lambda x: int(x))
            print(
                f"Found {len(concurrencies)} common concurrency levels: {', '.join(concurrencies)}"
            )
            print(f"All levels per run_id: {run_id_concurrencies}")

            runid_data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

            print(f"\nProcessing chip: {chip_name}")

            for run_id in RUN_IDS:
                print(f"\n  Processing RUN-ID: {run_id}")
                for conc in concurrencies:
                    metrics = get_chip_metrics(base_path, run_id, conc)
                    if metrics:
                        runid_data[run_id][chip_name][conc] = metrics
                        print(f"    - {conc}并发: OK")
                    else:
                        print(f"    - {conc}并发: No data")

            test_overview = get_test_overview(test_suite)

            print("\nGenerating comparison reports...")

            generate_comparison_csv(runid_data, concurrencies, output_base, chip_name)

            if HAS_MATPLOTLIB:
                generate_comparison_charts(
                    runid_data, concurrencies, output_base, chip_name
                )

            generate_markdown_report(
                runid_data,
                concurrencies,
                output_base,
                test_suite,
                chip_name,
                vllm_config=VLLM_CONFIG,
                test_overview=test_overview,
            )

            print(f"\n{'=' * 50}")
            print(
                f"Run ID comparison for {chip_name} - {test_suite} generated successfully!"
            )
            print(f"Output directory: {output_base}")
            print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
