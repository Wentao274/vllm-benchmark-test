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


RUN_IDS = ["01", "02"]

MODEL_NAME = "MiniMax-M2.5-W8A8"

CHIP_BASE_PATHS = {}


def load_yaml_config(config_path="config/models_scenarios.yaml"):
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


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

    chip_key_map = {
        "hygon_bw1000": "Hygon_BW1000",
        "kunlun_p800": "Kunlun_P800",
        "nvidia_h100": "NVIDIA_H100",
    }
    chip_key = chip_key_map.get(chip_name.lower(), chip_name)

    chip_configs = chips_raw.get(chip_key, [])
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

    chip_key_map = {
        "hygon_bw1000": "Hygon_BW1000",
        "kunlun_p800": "Kunlun_P800",
        "nvidia_h100": "NVIDIA_H100",
    }
    chip_key = chip_key_map.get(chip_name.lower(), chip_name)

    config_list = vllm_configs_raw.get(chip_key, [])
    if isinstance(config_list, list):
        for cfg in config_list:
            if cfg.get("model_name") == model_name:
                return cfg
        return config_list[0] if config_list else {}
    elif isinstance(config_list, dict):
        return config_list
    return {}


def get_test_overview(test_suite):
    config = load_yaml_config()
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

    section = None
    section_patterns = {
        "Serving Benchmark Result": "=========== Serving Benchmark Result",
        "End-to-End Latency": "----------------End-to-End Latency",
        "Time to First Token": "---------------Time to First Token",
        "Time per Output Token": "-----Time per Output Token",
        "Inter-Token Latency": "---------------Inter-Token Latency",
    }

    for line in lines:
        found_section = None
        for sec_name, sec_pattern in section_patterns.items():
            if sec_pattern in line:
                found_section = sec_name
                break

        if found_section:
            section = found_section
            continue

        if section and line.strip().startswith("==========="):
            section = None
            continue

        if section:
            match = re.match(r"(.+?):\s+(.+)$", line.strip())
            if match:
                key = match.group(1).strip()
                value = match.group(2).strip()
                metrics[key] = value

    if "Failed requests" not in metrics:
        metrics["Failed requests"] = "0"

    return metrics


def extract_concurrency_from_dir(dir_name):
    match = re.match(r"^(\d+)-", dir_name)
    if match:
        return match.group(1)
    return None


def extract_io_pair_from_dir(dir_name):
    match = re.search(r"-i(\d+)-o(\d+)", dir_name)
    if match:
        return match.group(1), match.group(2)
    return None, None


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


def get_all_io_pairs(base_path, run_id, concurrency):
    io_pairs = set()
    full_path = os.path.join(base_path, run_id)

    if not os.path.exists(full_path):
        return []

    for item in os.listdir(full_path):
        item_path = os.path.join(full_path, item)
        if os.path.isdir(item_path):
            dir_concurrency = extract_concurrency_from_dir(item)
            if dir_concurrency == concurrency:
                input_len, output_len = extract_io_pair_from_dir(item)
                if input_len and output_len:
                    io_pairs.add((input_len, output_len))

    return sorted(io_pairs, key=lambda x: (int(x[0]), int(x[1])))


def get_chip_metrics(base_path, run_id, concurrency, io_pair=None):
    full_path = os.path.join(base_path, run_id)

    if io_pair:
        dir_pattern = os.path.join(
            full_path, f"{concurrency}-*-i{io_pair[0]}-o{io_pair[1]}"
        )
    else:
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


def generate_comparison_csv(
    runid_data, concurrencies, output_dir, chip_name, run_ids=None
):
    if run_ids is None:
        run_ids = RUN_IDS

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

    # 支持多个 RUN-ID: 最多显示前两个的对比，其他的在报告中显示
    num_run_ids = len(run_ids)

    csv_lines = []

    # 根据 RUN-ID 数量决定列格式
    if num_run_ids == 2:
        # 两个 RUN-ID: 显示值、差异、百分比
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
                    .get(key_name.lower(), "")
                )
                val2 = (
                    runid_data.get(run_id2, {})
                    .get(chip_name, {})
                    .get(conc, {})
                    .get(key_name.lower(), "")
                )

                row.append(val1)
                row.append(val2)

                diff, pct = calculate_diff(val1, val2)
                diff_str, pct_str = format_diff(diff, pct)
                row.append(diff_str)
                row.append(pct_str)

            csv_lines.append(",".join(row))
    else:
        # 多个 RUN-ID: 以第一个为基准，计算其他与第一个的差异
        baseline_id = run_ids[0]
        other_ids = run_ids[1:]

        # 列格式: 值(baseline), 值(other1), Diff(other1), %(other1), 值(other2), Diff(other2), %(other2), ...
        header_parts = ["Metric"]
        for conc in concurrencies:
            header_parts.append(f"{conc}-{baseline_id} (基准)")
            for other_id in other_ids:
                header_parts.append(f"{conc}-{other_id}")
                header_parts.append(f"{conc}-{other_id}-Diff")
                header_parts.append(f"{conc}-{other_id}-%")
        csv_lines.append(",".join(header_parts))

        num_other = len(other_ids)
        for display_name, key_name in metric_names:
            if not key_name:
                csv_lines.append(
                    f"[{display_name}]"
                    + ",," * (len(concurrencies) * (1 + num_other * 3))
                )
                continue

            row = [display_name]

            for conc in concurrencies:
                # 获取基准值
                baseline_val = (
                    runid_data.get(baseline_id, {})
                    .get(chip_name, {})
                    .get(conc, {})
                    .get(key_name, "")
                )
                row.append(baseline_val)

                # 计算其他每个与基准的差异
                for other_id in other_ids:
                    other_val = (
                        runid_data.get(other_id, {})
                        .get(chip_name, {})
                        .get(conc, {})
                        .get(key_name, "")
                    )

                    row.append(other_val)

                    diff, pct = calculate_diff(baseline_val, other_val)
                    diff_str, pct_str = format_diff(diff, pct)
                    row.append(diff_str)
                    row.append(pct_str)

            csv_lines.append(",".join(row))

    csv_file = os.path.join(output_dir, "runid_comparison.csv")
    with open(csv_file, "w", encoding="utf-8") as f:
        f.write("\n".join(csv_lines))

    print(f"Generated: {csv_file}")
    return [csv_file]


def generate_comparison_charts(
    runid_data, concurrencies, output_dir, chip_name, model_name=None, run_ids=None
):
    if not HAS_MATPLOTLIB:
        return None

    if run_ids is None:
        run_ids = RUN_IDS

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
    plt.rcParams["axes.unicode_minus"] = False

    actual_model_name = model_name if model_name else MODEL_NAME
    num_run_ids = len(run_ids)

    # 生成 RUN-ID 列表字符串用于标题
    run_id_str = (
        " vs ".join(run_ids)
        if num_run_ids <= 3
        else " vs ".join(run_ids[:3])
        + (" (+{} more)".format(num_run_ids - 3) if num_run_ids > 3 else "")
    )

    x = range(len(concurrencies))

    # 根据 RUN-ID 数量调整宽度
    # 每个 RUN-ID 需要一个柱子位置，相邻柱子之间有一定间距
    total_width = 0.8
    bar_width = total_width / num_run_ids if num_run_ids > 0 else 0.35

    # 为每个 RUN-ID 分配颜色
    colors = [
        "#3498db",
        "#2ecc71",
        "#e74c3c",
        "#f39c12",
        "#9b59b6",
        "#1abc9c",
        "#e67e22",
        "#34495e",
    ]

    fig, axes = plt.subplots(2, 3, figsize=(22, 14))
    fig.suptitle(
        f"{actual_model_name} on {chip_name} - Run ID Comparison ({run_id_str})",
        fontsize=16,
        fontweight="bold",
    )

    def get_values(run_id, key):
        values = []
        for conc in concurrencies:
            val = (
                runid_data.get(run_id, {})
                .get(chip_name, {})
                .get(conc, {})
                .get(key.lower(), "0")
            )
            try:
                values.append(float(val))
            except:
                values.append(0)
        return values

    # 获取所有 RUN-ID 的数据
    all_run_ids_data = {}
    for run_id in run_ids:
        all_run_ids_data[run_id] = {
            "req_tp": get_values(run_id, "Request throughput (req/s)"),
            "output_tp": get_values(run_id, "Output token throughput (tok/s)"),
            "total_tp": get_values(run_id, "Total token throughput (tok/s)"),
            "ttft": get_values(run_id, "P99 TTFT (ms)"),
            "tpot": get_values(run_id, "P99 TPOT (ms)"),
            "itl": get_values(run_id, "P99 ITL (ms)"),
        }

    metrics = [
        ("Request Throughput (req/s)", "req_tp", axes[0, 0]),
        ("Output Token Throughput (tok/s)", "output_tp", axes[0, 1]),
        ("Total Token Throughput (tok/s)", "total_tp", axes[0, 2]),
        ("TTFT P99 (ms)", "ttft", axes[1, 0]),
        ("TPOT P99 (ms)", "tpot", axes[1, 1]),
        ("ITL P99 (ms)", "itl", axes[1, 2]),
    ]

    for title, data_key, ax in metrics:
        for i, run_id in enumerate(run_ids):
            offset = (i - (num_run_ids - 1) / 2) * bar_width
            values = all_run_ids_data[run_id][data_key]
            bars = ax.bar(
                [xi + offset for xi in x],
                values,
                bar_width * 0.7,
                label=run_id,
                color=colors[i % len(colors)],
                alpha=0.8,
            )

            max_val = max(values) if values else 1
            for bar, val in zip(bars, values):
                height = bar.get_height()
                if height > 0:
                    if "Throughput" in title and "tok/s" in title:
                        label_text = f"{val:.0f}"
                    elif "Throughput" in title:
                        label_text = f"{val:.2f}"
                    else:
                        label_text = f"{val:.1f}"
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        height + 0.02 * max_val,
                        label_text,
                        ha="center",
                        va="bottom",
                        fontsize=8,
                        fontweight="bold",
                    )

        ax.set_title(title, fontsize=13)
        ax.set_xlabel("Concurrency", fontsize=11)
        ax.set_ylabel(
            title.split("(")[-1].replace(")", "") if "(" in title else "", fontsize=11
        )
        ax.set_xticks(x)
        ax.set_xticklabels(concurrencies, rotation=0, fontsize=11)
        ax.legend(fontsize=11)
        ax.grid(axis="y", alpha=0.3)

        max_all = 0
        for run_id in run_ids:
            vals = all_run_ids_data[run_id][data_key]
            max_all = max(max_all, max(vals)) if vals else max_all
        if max_all > 0:
            ax.set_ylim(0, max_all * 1.15)

    for ax in axes.flat:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.set_facecolor("#f0f0f0")
    for ax in axes.flat:
        ax.set_facecolor("white")

    plt.tight_layout()

    chart_file = os.path.join(output_dir, "runid_comparison.png")
    plt.savefig(chart_file, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Generated chart: {chart_file}")
    return [chart_file]


def generate_markdown_report(
    runid_data,
    concurrencies,
    output_dir,
    test_suite,
    chip_name,
    model_name=None,
    run_ids=None,
    test_overview=None,
    io_pair=None,
):
    current_date = datetime.now().strftime("%Y-%m-%d")

    if run_ids is None:
        run_ids = RUN_IDS

    run_id1 = run_ids[0]
    run_id2 = run_ids[1] if len(run_ids) > 1 else run_ids[0]

    actual_model_name = model_name if model_name else MODEL_NAME

    if test_overview is None:
        test_overview = {}

    yaml_config = load_yaml_config()
    vllm_version = yaml_config.get("vllm_version", "N/A")

    def make_table_for_conc(conc, key_name):
        if len(run_ids) == 2:
            val1 = (
                runid_data.get(run_id1, {})
                .get(chip_name, {})
                .get(conc, {})
                .get(key_name.lower(), "")
            )
            val2 = (
                runid_data.get(run_id2, {})
                .get(chip_name, {})
                .get(conc, {})
                .get(key_name.lower(), "")
            )

            diff, pct = calculate_diff(val1, val2)
            diff_str, pct_str = format_diff(diff, pct)

            return val1, val2, diff_str, pct_str
        else:
            baseline_id = run_ids[0]
            baseline_val = (
                runid_data.get(baseline_id, {})
                .get(chip_name, {})
                .get(conc, {})
                .get(key_name.lower(), "")
            )

            result = [baseline_val]
            for other_id in run_ids[1:]:
                other_val = (
                    runid_data.get(other_id, {})
                    .get(chip_name, {})
                    .get(conc, {})
                    .get(key_name.lower(), "")
                )
                diff, pct = calculate_diff(baseline_val, other_val)
                diff_str, pct_str = format_diff(diff, pct)
                result.extend([other_val, diff_str, pct_str])

            return result

    serving_metrics = [
        ("成功请求数", "Successful requests"),
        ("失败请求数", "Failed requests"),
        ("测试持续时间 (s)", "Benchmark duration (s)"),
        ("总输入 tokens", "Total input tokens"),
        ("总生成 tokens", "Total generated tokens"),
        ("峰值并发请求数", "Peak concurrent requests"),
        ("**请求吞吐量 (req/s)**", "Request throughput (req/s)"),
        ("**输出 token 吞吐量 (tok/s)**", "Output token throughput (tok/s)"),
        ("峰值输出 token 吞吐量 (tok/s)", "Peak output token throughput (tok/s)"),
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
        if len(run_ids) == 2:
            header = f"| 指标 | RUN-{run_id1} | RUN-{run_id2} | 差异 | 百分比 |"
            separator = "|------|----------|---------|---------|---------|"
        else:
            baseline_id = run_ids[0]
            other_ids = run_ids[1:]
            header_parts = ["| 指标", f"RUN-{baseline_id} (基准)"]
            for other_id in other_ids:
                header_parts.extend([f"RUN-{other_id}", "差异", "%"])
            header = " | ".join(header_parts) + " |"

            num_cols = 1 + 3 * len(other_ids)
            sep_parts = ["|------"]
            sep_parts.append("---------------")
            for _ in other_ids:
                sep_parts.extend(["---------", "-------", "-------"])
            separator = " | ".join(sep_parts) + " |"

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

    def calc_avg_improvement(key_name, other_id):
        improvements = []
        for conc in concurrencies:
            baseline_val = (
                runid_data.get(run_id1, {})
                .get(chip_name, {})
                .get(conc, {})
                .get(key_name.lower(), "")
            )
            other_val = (
                runid_data.get(other_id, {})
                .get(chip_name, {})
                .get(conc, {})
                .get(key_name.lower(), "")
            )
            try:
                v1 = float(baseline_val)
                v2 = float(other_val)
                if v1 > 0:
                    pct = ((v2 - v1) / v1) * 100
                    improvements.append(pct)
            except:
                pass
        return sum(improvements) / len(improvements) if improvements else 0

    other_run_ids = run_ids[1:]
    improvements_data = {}

    for other_id in other_run_ids:
        improvements_data[other_id] = {
            "tp": calc_avg_improvement("Request throughput (req/s)", other_id),
            "output_tp": calc_avg_improvement(
                "Output token throughput (tok/s)", other_id
            ),
            "total_tp": calc_avg_improvement(
                "Total token throughput (tok/s)", other_id
            ),
            "ttft": calc_avg_improvement("P99 TTFT (ms)", other_id),
            "tpot": calc_avg_improvement("P99 TPOT (ms)", other_id),
            "itl": calc_avg_improvement("P99 ITL (ms)", other_id),
        }

    analysis_lines = []

    analysis_lines.append("### 吞吐量对比\n")
    for other_id in other_run_ids:
        imp = improvements_data[other_id]
        if imp["tp"] > 0:
            analysis_lines.append(
                f"**请求吞吐量**: RUN-{other_id} 相比 RUN-{run_id1} 平均提升 **{imp['tp']:.1f}%**\n"
            )
        else:
            analysis_lines.append(
                f"**请求吞吐量**: RUN-{other_id} 相比 RUN-{run_id1} 平均变化 **{imp['tp']:.1f}%**\n"
            )

        if imp["output_tp"] > 0:
            analysis_lines.append(
                f"**输出Token吞吐量**: RUN-{other_id} 相比 RUN-{run_id1} 平均提升 **{imp['output_tp']:.1f}%**\n"
            )
        else:
            analysis_lines.append(
                f"**输出Token吞吐量**: RUN-{other_id} 相比 RUN-{run_id1} 平均变化 **{imp['output_tp']:.1f}%**\n"
            )

        if imp["total_tp"] > 0:
            analysis_lines.append(
                f"**总Token吞吐量**: RUN-{other_id} 相比 RUN-{run_id1} 平均提升 **{imp['total_tp']:.1f}%**\n"
            )
        else:
            analysis_lines.append(
                f"**总Token吞吐量**: RUN-{other_id} 相比 RUN-{run_id1} 平均变化 **{imp['total_tp']:.1f}%**\n"
            )

    analysis_lines.append("### 延迟对比\n")
    for other_id in other_run_ids:
        imp = improvements_data[other_id]
        if imp["ttft"] > 0:
            analysis_lines.append(
                f"**TTFT P99**: RUN-{other_id} 相比 RUN-{run_id1} 平均增加 **{imp['ttft']:.1f}%** (延迟增加)\n"
            )
        else:
            analysis_lines.append(
                f"**TTFT P99**: RUN-{other_id} 相比 RUN-{run_id1} 平均改善 **{abs(imp['ttft']):.1f}%** (延迟降低)\n"
            )

        if imp["tpot"] > 0:
            analysis_lines.append(
                f"**TPOT P99**: RUN-{other_id} 相比 RUN-{run_id1} 平均增加 **{imp['tpot']:.1f}%** (延迟增加)\n"
            )
        else:
            analysis_lines.append(
                f"**TPOT P99**: RUN-{other_id} 相比 RUN-{run_id1} 平均改善 **{abs(imp['tpot']):.1f}%** (延迟降低)\n"
            )

        if imp["itl"] > 0:
            analysis_lines.append(
                f"**ITL P99**: RUN-{other_id} 相比 RUN-{run_id1} 平均增加 **{imp['itl']:.1f}%** (延迟增加)\n"
            )
        else:
            analysis_lines.append(
                f"**ITL P99**: RUN-{other_id} 相比 RUN-{run_id1} 平均改善 **{abs(imp['itl']):.1f}%** (延迟降低)\n"
            )

    conclusion = "\n".join(analysis_lines)

    concurrency_comparison_img = '<img src="./runid_comparison.png" width="1000" />'

    dataset = test_overview.get("dataset", "random")
    concurrency_list = test_overview.get("concurrency", concurrencies)
    concurrency_str = (
        str(concurrency_list) if concurrency_list else ", ".join(concurrencies)
    )
    total_requests_list = test_overview.get("total_requests", [])
    total_requests = str(total_requests_list) if total_requests_list else "N/A"
    input_len_list = test_overview.get("input_context_length", [])
    input_ctx = str(input_len_list) if input_len_list else "N/A"
    output_len_list = test_overview.get("output_context_length", [])
    output_ctx = str(output_len_list) if output_len_list else "N/A"
    model = test_overview.get("model", actual_model_name)
    chip = test_overview.get("chip", chip_name)

    chip_info = load_chip_config_by_model(chip_name, actual_model_name)
    chip_table_rows = []
    for param, val in chip_info.items():
        if param == "remark":
            continue
        if val is None:
            val = "N/A"
        chip_table_rows.append(f"| **{param}** | {val} |")
    chip_table = "\n".join(chip_table_rows)

    vllm_info = load_vllm_config_by_model(chip_name, actual_model_name)
    vllm_table_rows = []
    for param, val in vllm_info.items():
        if param == "remarks":
            continue
        if val is None:
            val = "N/A"
        display_name = param.replace("-", " ").replace("_", " ").title()
        vllm_table_rows.append(f"| **{display_name}** | {val} |")
    vllm_table = "\n".join(vllm_table_rows)

    if len(run_ids) == 2:
        run_ids_display = f"{run_ids[0]} vs {run_ids[1]}"
    else:
        run_ids_display = ", ".join(run_ids)

    if io_pair:
        title_suffix = f"多I/O测试报告 ({io_pair})"
    else:
        title_suffix = "多次运行结果对比报告"

    md_content = f"""# {actual_model_name}模型在{chip_name}上{title_suffix}

<div align="center">
**测试日期：** {current_date}

**对比RUN-ID：** {run_ids_display}

</div>

---

## 测试场景
对比同一芯片、同一测试套件下,同一模型优化前后测试结果比对，分析性能差异。

**测试模型** <br>
{"".join([f"第{i + 1}轮测试（RUN-{rid}）: {model} <br>" for i, rid in enumerate(run_ids)])}

## 🤖 芯片和模型配置信息

| 参数名称                    | {chip_name} |
|------------------------|-------------|
{chip_table}

---

## ⚙️ vLLM启动配置信息

| 参数名称                    | {chip_name} |
|------------------------|-------------|
{vllm_table}

---

## 📊 测试概览

| 项目            | 配置                                    | 备注  |
|---------------|---------------------------------------|-----|
| **测试套件**     | {test_suite}                           |     |
| **数据集**       | {dataset}                                |     |
| **并发数**       | {concurrency_str} |     |
| **总请求数**      | {total_requests}                                 |     |
| **请求输入上下文长度** | {input_ctx}                               |     |
| **请求输出上下文长度** | {output_ctx}                               |     |
| **模型**        | {model}                          |     |
| **被测芯片**      | {chip}                          |     |
| **测试场景**      | {"多I/O测试" if io_pair else "单I/O测试"}                          |     |


**主要采集指标**：

| 指标                  | 单位         | 含义                                 |
|---------------------|------------|------------------------------------|
| TTFT                | ms         | Time To First Token，首 token 延迟     |
| TPOT                | ms/token   | Time Per Output Token，每 token 生成时间 |
| Throughput          | tokens/s   | 系统总吞吐                              |
| QPS                 | requests/s | 请求吞吐                               |
| P50/P95/P99 Latency | ms         | 延迟分位数                              |

---

## 📊 RUN-ID对比柱状图

{concurrency_comparison_img}

---

## 各并发级别详细对比

{tables_html}

---

## 📝 分析总结

{conclusion}

---

<div align="center">
*报告生成时间: {current_date}*
</div>
"""

    run_ids_str = "_".join(run_ids)

    md_file = os.path.join(
        output_dir,
        f"{actual_model_name}_{chip_name}_{test_suite}_runid_compare_{run_ids_str}.md",
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
        required=True,
        help="Chip name (e.g., Hygon_BW1000, Kunlun_P800)",
    )
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument(
        "--test-suite",
        type=str,
        default="test_01",
        help="Test suite name (e.g., test_01)",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        required=True,
        help="Run IDs to compare, separated by comma (e.g., 01,02 or 01,02,03)",
    )
    parser.add_argument(
        "--concurrency",
        type=str,
        default=None,
        help="Specific concurrency levels to compare, comma-separated (e.g., 1,2,4,8,10)",
    )
    args = parser.parse_args()

    global RUN_IDS, CHIP_BASE_PATHS, MODEL_NAME

    args.chip = args.chip.lower()
    args.test_suite = args.test_suite.lower()

    concurrency_filter = None
    if args.concurrency:
        concurrency_filter = [s.strip() for s in args.concurrency.split(",")]

    run_ids_list = []
    for part in args.run_id.replace("'", "").split(","):
        part = part.strip()
        if part:
            run_ids_list.append(part)
    RUN_IDS = run_ids_list

    if len(RUN_IDS) < 2:
        print(f"\nError: At least 2 RUN-IDs are required for comparison")
        print(f"Provided: {len(RUN_IDS)} ({', '.join(RUN_IDS) if RUN_IDS else 'none'})")
        print(f'Usage: --run-id 01,02 or --run-id "01, 02"')
        return

    chip_name = args.chip
    model_input = args.model
    MODEL_NAME = model_input

    print(f"\n{'=' * 60}")
    print(f"Run ID Comparison Configuration")
    print(f"{'=' * 60}")
    print(f"Chip: {chip_name}")
    print(f"Model: {MODEL_NAME}")
    print(f"Test Suite: {args.test_suite}")
    print(f"RUN IDs: {', '.join(RUN_IDS)}")
    print(f"{'=' * 60}\n")

    benchmark_path = f"reports/benchmark/{chip_name}"

    if not os.path.exists(benchmark_path):
        print(f"\nError: Benchmark path not found: {benchmark_path}")
        return

    available_models = [
        d
        for d in os.listdir(benchmark_path)
        if os.path.isdir(os.path.join(benchmark_path, d))
    ]

    model_path = os.path.join(benchmark_path, model_input)
    if not os.path.exists(model_path):
        print(f"\nError: Model directory not found: {model_path}")
        print(f"Expected model: {model_input}")
        print(f"\nAvailable model directories for {chip_name}:")
        for m in available_models:
            print(f"  - {m}")
        return

    CHIP_BASE_PATHS = {chip_name: f"reports/benchmark/{chip_name}/{MODEL_NAME}"}

    test_suite = args.test_suite

    print(f"\n{'#' * 60}")
    print(f"Processing test suite: {test_suite}")
    print(f"Comparing RUN-IDs: {', '.join(RUN_IDS)}")
    print(f"Chip: {chip_name}, Model: {MODEL_NAME}")
    print(f"{'#' * 60}\n")

    chip_configs = get_chip_configs(test_suite)

    for chip in chip_configs:
        chip_name = chip["name"]
        base_path = chip["base_path"]

        missing_run_ids = []
        for run_id in RUN_IDS:
            run_id_path = os.path.join(base_path, run_id)
            if not os.path.exists(run_id_path):
                missing_run_ids.append(run_id)

        if missing_run_ids:
            print(
                f"\nError: Missing RUN-ID directories for {chip_name} / {MODEL_NAME} / {test_suite}"
            )
            print(f"Expected RUN-IDs: {', '.join(RUN_IDS)}")
            print(f"Missing: {', '.join(missing_run_ids)}")

            available_run_ids = [
                d
                for d in os.listdir(base_path)
                if os.path.isdir(os.path.join(base_path, d))
            ]
            print(
                f"Available RUN-IDs: {', '.join(available_run_ids) if available_run_ids else 'None'}"
            )
            continue

        runid_folder = "run_" + "_".join(RUN_IDS)
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
            f"Found {len(concurrencies)} common concurrency levels: {', '.join(concurrencies)}"
        )
        print(f"All levels per run_id: {run_id_concurrencies}")

        runid_data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

        is_multi_io = test_suite == "test_05"
        all_io_pairs = set()

        if is_multi_io:
            for run_id in RUN_IDS:
                for conc in concurrencies:
                    io_pairs = get_all_io_pairs(base_path, run_id, conc)
                    all_io_pairs.update(io_pairs)

            all_io_pairs = sorted(all_io_pairs, key=lambda x: (int(x[0]), int(x[1])))
            print(f"Detected multi-I/O scenario: {len(all_io_pairs)} I/O pairs")
            print(f"I/O pairs: {all_io_pairs}")

            if not all_io_pairs:
                print(f"No I/O pairs found for {chip_name} / {test_suite}!")
                return

        print(f"\nProcessing chip: {chip_name}")

        for run_id in RUN_IDS:
            print(f"\n  Processing RUN-ID: {run_id}")
            for conc in concurrencies:
                if is_multi_io:
                    for io_pair in all_io_pairs:
                        io_key = f"i{io_pair[0]}-o{io_pair[1]}"
                        metrics = get_chip_metrics(base_path, run_id, conc, io_pair)
                        if metrics:
                            normalized_metrics = {}
                            for key, value in metrics.items():
                                normalized_metrics[key.lower()] = value
                            runid_data[run_id][chip_name][(conc, io_key)] = (
                                normalized_metrics
                            )
                            print(f"    - {conc}并发/{io_key}: OK")
                        else:
                            print(f"    - {conc}并发/{io_key}: No data")
                else:
                    metrics = get_chip_metrics(base_path, run_id, conc)
                    if metrics:
                        normalized_metrics = {}
                        for key, value in metrics.items():
                            normalized_metrics[key.lower()] = value
                        runid_data[run_id][chip_name][conc] = normalized_metrics
                        print(f"    - {conc}并发: OK")
                    else:
                        print(f"    - {conc}并发: No data")

        test_overview = get_test_overview(test_suite)

        if is_multi_io:
            for io_pair in all_io_pairs:
                io_key = f"i{io_pair[0]}-o{io_pair[1]}"
                io_output_base = f"{output_base}/{io_key}"
                Path(io_output_base).mkdir(parents=True, exist_ok=True)

                io_runid_data = defaultdict(
                    lambda: defaultdict(lambda: defaultdict(dict))
                )
                for run_id in RUN_IDS:
                    for conc in concurrencies:
                        if (conc, io_key) in runid_data[run_id][chip_name]:
                            io_runid_data[run_id][chip_name][conc] = runid_data[run_id][
                                chip_name
                            ][(conc, io_key)]

                io_test_overview = test_overview.copy()
                io_test_overview["input_context_length"] = [int(io_pair[0])]
                io_test_overview["output_context_length"] = [int(io_pair[1])]

                print(f"\nGenerating comparison reports for I/O pair: {io_key}...")

                generate_comparison_csv(
                    io_runid_data,
                    concurrencies,
                    io_output_base,
                    chip_name,
                    run_ids=RUN_IDS,
                )

                if HAS_MATPLOTLIB:
                    generate_comparison_charts(
                        io_runid_data,
                        concurrencies,
                        io_output_base,
                        chip_name,
                        MODEL_NAME,
                        run_ids=RUN_IDS,
                    )

                generate_markdown_report(
                    io_runid_data,
                    concurrencies,
                    io_output_base,
                    test_suite,
                    chip_name,
                    model_name=MODEL_NAME,
                    run_ids=RUN_IDS,
                    test_overview=io_test_overview,
                    io_pair=io_key,
                )

            print(f"\nMulti-I/O reports generated for {chip_name} - {test_suite}")
        else:
            print("\nGenerating comparison reports...")

            generate_comparison_csv(
                runid_data, concurrencies, output_base, chip_name, run_ids=RUN_IDS
            )

            if HAS_MATPLOTLIB:
                generate_comparison_charts(
                    runid_data,
                    concurrencies,
                    output_base,
                    chip_name,
                    MODEL_NAME,
                    run_ids=RUN_IDS,
                )

            generate_markdown_report(
                runid_data,
                concurrencies,
                output_base,
                test_suite,
                chip_name,
                model_name=MODEL_NAME,
                run_ids=RUN_IDS,
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
