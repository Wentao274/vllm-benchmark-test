import os
import re
import glob
import yaml
import argparse
from pathlib import Path
from datetime import datetime

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("matplotlib not available, skipping chart generation")


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

    chip_key = chip_name
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

    chip_key = chip_name
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


def parse_benchmark_log(log_file):
    with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    lines = content.split("\n")
    metrics = {}

    in_results = False
    for i, line in enumerate(lines):
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
                metrics[key] = value

    return metrics


def extract_test_config_from_path(path):
    parts = path.split(os.sep)
    for part in parts:
        if re.match(r"^\d+-\d+-i\d+-o\d+$", part):
            return part
    return None


def extract_concurrency_from_config(config):
    match = re.match(r"^(\d+)-\d+-i\d+-o\d+$", config)
    if match:
        return int(match.group(1))
    return None


def extract_input_len_from_config(config):
    match = re.match(r"^\d+-i(\d+)-o\d+$", config)
    if match:
        return int(match.group(1))
    return 0


def parse_run_ids(run_ids_str, num_models):
    if not run_ids_str:
        return None

    parts = [p.strip() for p in run_ids_str.split(",")]

    if len(parts) == 1 and num_models > 1:
        return [parts[0]] * num_models

    if len(parts) != num_models:
        return None

    return parts


def get_test_params_from_yaml(test_suite):
    yaml_config = load_yaml_config()
    base_config = yaml_config.get("base_config", {})
    params = base_config.get("params", {})

    test_suite_params = params.get(test_suite, {})

    num_prompts = test_suite_params.get("num-prompts", [320])
    max_concurrency = test_suite_params.get("max-concurrency", [1])

    input_output_lens = test_suite_params.get("random-input-output-len", [])

    test_configs = set()
    for np in num_prompts:
        for io in input_output_lens:
            if isinstance(io, list) and len(io) >= 2:
                ni, no = io[0], io[1]
            else:
                continue
            config = f"{np}-i{ni}-o{no}"
            test_configs.add(config)

    return {
        "test_configs": sorted(
            test_configs, key=lambda x: extract_input_len_from_config(x)
        ),
        "concurrency_list": sorted([int(c) for c in max_concurrency], key=lambda x: x),
    }


def get_concurrency_from_log_path(path):
    parts = path.split(os.sep)
    for part in parts:
        if "bench-" in part:
            match = re.match(r"bench-(\d+)-", part)
            if match:
                return int(match.group(1))
    return None


def get_model_data(chip, model_name, test_suite, run_id, config_with_concurrency):
    reports_base = "reports"
    model_dir = os.path.join(
        reports_base, "benchmark", chip, model_name, test_suite, run_id
    )

    print(f"      get_model_data: checking {model_dir}")

    if not os.path.isdir(model_dir):
        print(f"      Warning: Directory not found: {model_dir}")
        return None

    config_path = os.path.join(model_dir, config_with_concurrency)
    print(
        f"      Looking for config_path: {config_path}, exists: {os.path.isdir(config_path)}"
    )

    if not os.path.isdir(config_path):
        return None
        return None

    log_files = glob.glob(os.path.join(config_path, "bench-*.log"))
    if not log_files:
        return None

    log_file = log_files[0]
    metrics = parse_benchmark_log(log_file)
    return metrics


def get_all_test_configs(chip, model_name, test_suite, run_id):
    reports_base = "reports"
    model_dir = os.path.join(
        reports_base, "benchmark", chip, model_name, test_suite, run_id
    )

    if not os.path.isdir(model_dir):
        return []

    test_configs = set()
    for item in os.listdir(model_dir):
        item_path = os.path.join(model_dir, item)
        if os.path.isdir(item_path):
            config = extract_test_config_from_path(item_path)
            if config:
                test_configs.add(config)

    return sorted(
        test_configs, key=lambda x: int(extract_concurrency_from_config(x) or 0)
    )


def generate_comparison_charts(models_data, test_config, output_dir, concurrency):
    if not HAS_MATPLOTLIB:
        return None

    sorted_models = sorted(models_data.keys())

    req_throughput = [
        float(models_data[m].get("request throughput (req/s)", 0) or 0)
        for m in sorted_models
    ]
    total_tput = [
        float(models_data[m].get("total token throughput (tok/s)", 0) or 0)
        for m in sorted_models
    ]
    ttft_p99 = [
        float(models_data[m].get("p99 ttft (ms)", 0) or 0) for m in sorted_models
    ]
    tpot_p99 = [
        float(models_data[m].get("p99 tpot (ms)", 0) or 0) for m in sorted_models
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"Model Comparison @ {test_config}", fontsize=14, fontweight="bold")

    colors = ["#3498db", "#2ecc71", "#e74c3c", "#9b59b6", "#f39c12"][
        : len(sorted_models)
    ]

    axes[0, 0].bar(sorted_models, req_throughput, color=colors, alpha=0.8)
    axes[0, 0].set_title("Request Throughput (req/s)", fontsize=11)
    axes[0, 0].set_ylabel("req/s")
    axes[0, 0].tick_params(axis="x", rotation=15)
    for i, v in enumerate(req_throughput):
        axes[0, 0].text(
            i,
            v + 0.05 * max(req_throughput) if req_throughput else 0.1,
            f"{v:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    axes[0, 1].bar(sorted_models, total_tput, color=colors, alpha=0.8)
    axes[0, 1].set_title("Total Token Throughput (tok/s)", fontsize=11)
    axes[0, 1].set_ylabel("tok/s")
    axes[0, 1].tick_params(axis="x", rotation=15)
    for i, v in enumerate(total_tput):
        axes[0, 1].text(
            i,
            v + 0.02 * max(total_tput) if total_tput else 1000,
            f"{v:.0f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    axes[1, 0].bar(sorted_models, ttft_p99, color=colors, alpha=0.8)
    axes[1, 0].set_title("TTFT P99 (ms)", fontsize=11)
    axes[1, 0].set_ylabel("ms")
    axes[1, 0].tick_params(axis="x", rotation=15)
    for i, v in enumerate(ttft_p99):
        axes[1, 0].text(
            i,
            v + 0.02 * max(ttft_p99) if ttft_p99 else 100,
            f"{v:.0f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    axes[1, 1].bar(sorted_models, tpot_p99, color=colors, alpha=0.8)
    axes[1, 1].set_title("TPOT P99 (ms)", fontsize=11)
    axes[1, 1].set_ylabel("ms")
    axes[1, 1].tick_params(axis="x", rotation=15)
    for i, v in enumerate(tpot_p99):
        axes[1, 1].text(
            i,
            v + 0.02 * max(tpot_p99) if tpot_p99 else 5,
            f"{v:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    for ax in axes.flat:
        ax.grid(axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()

    chart_file = os.path.join(output_dir, f"concurrency{concurrency}_comparison.png")
    plt.savefig(chart_file, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Generated chart: {chart_file}")
    return chart_file


def generate_comparison_csv(models_data, test_config, output_dir, concurrency):
    metric_names = [
        ("[Serving Benchmark Result]", ""),
        ("Successful requests", "successful requests"),
        ("Failed requests", "failed requests"),
        ("Benchmark duration (s)", "benchmark duration (s)"),
        ("Total input tokens", "total input tokens"),
        ("Total generated tokens", "total generated tokens"),
        ("Request throughput (req/s)", "request throughput (req/s)"),
        ("Output token throughput (tok/s)", "output token throughput (tok/s)"),
        (
            "Peak output token throughput (tok/s)",
            "peak output token throughput (tok/s)",
        ),
        ("Peak concurrent requests", "peak concurrent requests"),
        ("Total token throughput (tok/s)", "total token throughput (tok/s)"),
        ("[Time to First Token]", ""),
        ("Mean TTFT (ms)", "mean ttft (ms)"),
        ("Median TTFT (ms)", "median ttft (ms)"),
        ("P95 TTFT (ms)", "p95 ttft (ms)"),
        ("P99 TTFT (ms)", "p99 ttft (ms)"),
        ("[Time per Output Token]", ""),
        ("Mean TPOT (ms)", "mean tpot (ms)"),
        ("Median TPOT (ms)", "median tpot (ms)"),
        ("P95 TPOT (ms)", "p95 tpot (ms)"),
        ("P99 TPOT (ms)", "p99 tpot (ms)"),
        ("[Inter-token Latency]", ""),
        ("Mean ITL (ms)", "mean itl (ms)"),
        ("Median ITL (ms)", "median itl (ms)"),
        ("P95 ITL (ms)", "p95 itl (ms)"),
        ("P99 ITL (ms)", "p99 itl (ms)"),
    ]

    csv_lines = []
    header = ["Metric"] + sorted(models_data.keys())
    csv_lines.append(",".join(header))

    for display_name, key_name in metric_names:
        if not key_name:
            csv_lines.append(f"[{display_name}]" + ",," * (len(models_data) - 1))
            continue

        row = [display_name]
        for model in sorted(models_data.keys()):
            value = models_data[model].get(key_name, "")
            row.append(value)
        csv_lines.append(",".join(row))

    csv_file = os.path.join(output_dir, f"concurrency{concurrency}_comparison.csv")
    with open(csv_file, "w", encoding="utf-8") as f:
        f.write("\n".join(csv_lines))

    print(f"Generated: {csv_file}")
    return csv_file


def generate_comparison_markdown(
    models_data,
    test_config,
    output_dir,
    chip,
    test_suite,
    run_ids,
    concurrency_list,
    concurrency,
    chip_name,
    model_names,
):
    current_date = datetime.now().strftime("%Y-%m-%d")

    # 使用用户输入的模型顺序，第一个为基准
    # 只保留models_data中存在的模型，并保持用户输入的顺序
    ordered_models = [m for m in model_names if m in models_data.keys()]

    # 如果ordered_models为空fallback到排序
    if not ordered_models:
        ordered_models = sorted(models_data.keys())

    sorted_models = ordered_models
    headers = " | ".join(sorted_models)
    separator = " | ".join(["-----------"] * len(sorted_models))

    chip_table_rows = []
    all_params = set()
    for model in sorted_models:
        cfg = load_chip_config_by_model(chip_name, model)
        all_params.update(cfg.keys())
    all_params = sorted([p for p in all_params if p != "remark"])

    for param in all_params:
        row = f"| **{param}** |"
        for model in sorted_models:
            cfg = load_chip_config_by_model(chip_name, model)
            val = cfg.get(param, "N/A")
            row += f" {val} |"
        chip_table_rows.append(row)
    chip_table = "\n".join(chip_table_rows)

    vllm_table_rows = []
    all_vllm_params = set()
    for model in sorted_models:
        cfg = load_vllm_config_by_model(chip_name, model)
        all_vllm_params.update(cfg.keys())
    all_vllm_params = sorted([p for p in all_vllm_params if p != "remarks"])

    for param in all_vllm_params:
        display_name = param.replace("-", " ").replace("_", " ").title()
        row = f"| **{display_name}** |"
        for model in sorted_models:
            cfg = load_vllm_config_by_model(chip_name, model)
            val = cfg.get(param, "N/A")
            row += f" {val} |"
        vllm_table_rows.append(row)
    vllm_table = "\n".join(vllm_table_rows)

    def calculate_diff(baseline_val, other_val):
        """计算差异和百分比"""
        try:
            v1 = float(baseline_val)
            v2 = float(other_val)
            diff = v2 - v1
            if v1 != 0:
                pct = (diff / v1) * 100
                return diff, pct
            return diff, 0
        except:
            return None, None

    def format_diff(diff, pct):
        if diff is None:
            return "N/A", "N/A"
        sign = "+" if diff > 0 else ""
        diff_str = f"{sign}{diff:.2f}"
        pct_str = f"{sign}{pct:.1f}%"
        return diff_str, pct_str

    def make_row_with_diff(key_name):
        """生成包含差异和百分比的行，以第一个模型为基准"""
        baseline_model = sorted_models[0]
        baseline_val = models_data[baseline_model].get(key_name.lower(), "") or "0"

        cells = [baseline_val]

        for model in sorted_models[1:]:
            other_val = models_data[model].get(key_name.lower(), "") or "0"
            diff, pct = calculate_diff(baseline_val, other_val)
            diff_str, pct_str = format_diff(diff, pct)
            cells.append(other_val if other_val else "0")
            cells.append(diff_str)
            cells.append(pct_str)

        return " | ".join(cells)

    # 生成表头和分隔行（根据模型数量）
    num_models = len(sorted_models)
    if num_models == 2:
        # 两个模型：基准值 | 其他值 | 差异 | %
        headers = f"{sorted_models[0]} (基准) | {sorted_models[1]} | 差异 | %"
        separator = "--------------- | --------- | ------- | -------"
    else:
        # 多个模型：基准 | (值, 差异, %) * (n-1)
        header_parts = [f"{sorted_models[0]} (基准)"]
        for model in sorted_models[1:]:
            header_parts.extend([model, "差异", "%"])
        headers = " | ".join(header_parts)

        sep_parts = ["---------------"]
        for _ in sorted_models[1:]:
            sep_parts.extend(["---------", "-------", "-------"])
        separator = " | ".join(sep_parts)

    serving_rows = f"""| 成功请求数 | {make_row_with_diff("Successful requests")} |
| 失败请求数 | {make_row_with_diff("Failed requests")} |
| 测试持续时间 (s) | {make_row_with_diff("Benchmark duration (s)")} |
| 总输入 tokens | {make_row_with_diff("Total input tokens")} |
| 总生成 tokens | {make_row_with_diff("Total generated tokens")} |
| **请求吞吐量 (req/s)** | {make_row_with_diff("Request throughput (req/s)")} |
| **输出 token 吞吐量 (tok/s)** | {make_row_with_diff("Output token throughput (tok/s)")} |
| 峰值输出 token 吞吐量 (tok/s) | {make_row_with_diff("Peak output token throughput (tok/s)")} |
| 峰值并发请求数 | {make_row_with_diff("Peak concurrent requests")} |
| **总 token 吞吐量 (tok/s)** | {make_row_with_diff("Total token throughput (tok/s)")} |"""

    ttft_rows = f"""| 平均 TTFT (ms) | {make_row_with_diff("Mean TTFT (ms)")} |
| 中位 TTFT (ms) | {make_row_with_diff("Median TTFT (ms)")} |
| P95 TTFT (ms) | {make_row_with_diff("P95 TTFT (ms)")} |
| P99 TTFT (ms) | {make_row_with_diff("P99 TTFT (ms)")} |"""

    tpot_rows = f"""| 平均 TPOT (ms) | {make_row_with_diff("Mean TPOT (ms)")} |
| 中位 TPOT (ms) | {make_row_with_diff("Median TPOT (ms)")} |
| P95 TPOT (ms) | {make_row_with_diff("P95 TPOT (ms)")} |
| P99 TPOT (ms) | {make_row_with_diff("P99 TPOT (ms)")} |"""

    itl_rows = f"""| 平均 ITL (ms) | {make_row_with_diff("Mean ITL (ms)")} |
| 中位 ITL (ms) | {make_row_with_diff("Median ITL (ms)")} |
| P95 ITL (ms) | {make_row_with_diff("P95 ITL (ms)")} |
| P99 ITL (ms) | {make_row_with_diff("P99 ITL (ms)")} |"""

    analysis_lines = []

    try:
        throughputs = [
            (m, float(models_data[m].get("request throughput (req/s)", 0) or 0))
            for m in sorted_models
        ]
        max_tp = max(throughputs, key=lambda x: x[1])
        analysis_lines.append(
            f"- **请求吞吐量**: {max_tp[0]} 最高，达 {max_tp[1]:.2f} req/s"
        )
    except:
        pass

    try:
        total_tputs = [
            (m, float(models_data[m].get("total token throughput (tok/s)", 0) or 0))
            for m in sorted_models
        ]
        max_total = max(total_tputs, key=lambda x: x[1])
        analysis_lines.append(
            f"- **总token吞吐量**: {max_total[0]} 最高，达 {max_total[1]:.0f} tok/s"
        )
    except:
        pass

    try:
        ttft_p99 = [
            (
                m,
                float(
                    models_data[m].get("p99 ttft (ms)", float("inf")) or float("inf")
                ),
            )
            for m in sorted_models
        ]
        min_ttft = min(ttft_p99, key=lambda x: x[1])
        analysis_lines.append(
            f"- **TTFT P99**: {min_ttft[0]} 最优，为 {min_ttft[1]:.2f}ms"
        )
    except:
        pass

    try:
        tpot_p99 = [
            (
                m,
                float(
                    models_data[m].get("p99 tpot (ms)", float("inf")) or float("inf")
                ),
            )
            for m in sorted_models
        ]
        min_tpot = min(tpot_p99, key=lambda x: x[1])
        analysis_lines.append(
            f"- **TPOT P99**: {min_tpot[0]} 最优，为 {min_tpot[1]:.2f}ms"
        )
    except:
        pass

    analysis_content = (
        "\n".join(analysis_lines) if analysis_lines else "- 各模型性能表现待分析"
    )

    run_id_display = ", ".join(run_ids) if run_ids else "N/A"

    model_list_section = ""
    for i, model in enumerate(sorted_models):
        rid = run_ids[i] if i < len(run_ids) else "N/A"
        model_list_section += f"| {model} | {rid} | [OK] |\n"

    yaml_config = load_yaml_config()
    base_config = yaml_config.get("base_config", {})
    params = base_config.get("params", {})
    test_cfg = params.get(test_suite, {})

    dataset = test_cfg.get("dataset-name", "random")
    num_prompts = test_cfg.get("num-prompts", [])
    input_output_lens = test_cfg.get("random-input-output-len", [])

    if (
        input_output_lens
        and isinstance(input_output_lens[0], list)
        and len(input_output_lens[0]) >= 2
    ):
        io_str = ", ".join([f"({p[0]}, {p[1]})" for p in input_output_lens])
    else:
        input_len = test_cfg.get("random-input-len", [])
        output_len = test_cfg.get("random-output-len", [])
        if input_len and output_len:
            io_str = f"({input_len[0] if input_len else 'N/A'}, {output_len[0] if output_len else 'N/A'})"
        else:
            io_str = "N/A"

    config_concurrencies = test_cfg.get("max-concurrency", [])
    conc_str = (
        ", ".join([str(c) for c in config_concurrencies])
        if config_concurrencies
        else "N/A"
    )
    num_prompts_str = str(num_prompts[0]) if num_prompts else "N/A"

    vllm_version = yaml_config.get("vllm_version", "N/A")

    def format_tokens(val):
        try:
            v = int(val)
            if v >= 1024:
                return f"{v // 1024}k"
            else:
                return f"{v / 1024:.2f}k"
        except:
            return str(val)

    md_content = f"""# 多模型性能对比报告

<div>

**测试日期：** {current_date}

**芯片平台：** {chip}

**测试套件：** {test_suite}

**Run ID：** {run_id_display}

**并发级别：** {concurrency}并发

**测试配置：** {test_config}

</div>

---

## 🤖 芯片和模型配置信息

| 参数名称 | **{"** | **".join(sorted_models)}** |
|----------|{"----------|" * len(sorted_models)}
{chip_table}

---

## ⚙️ vLLM 启动配置信息

| 参数名称 | **{"** | **".join(sorted_models)}** |
|----------|{"----------|" * len(sorted_models)}
{vllm_table}

---

## 📊 模型列表

| 模型名称 | Run ID | 状态 |
|----------|--------|------|
{model_list_section}

---

## 📊 测试概览

| 项目            | 配置                                     | 备注  |
|---------------|----------------------------------------|-----|
| **数据集**       | {dataset}                                 |     |
| **并发数**       | {conc_str}    |     |
| **总请求数**      | {num_prompts_str}                                    |     |
| **输入输出长度** | {io_str} |     |
| **测试套件**     | {test_suite}                           |     |
| **被测芯片**      | {chip} |     |
| **vLLM版本**   | {vllm_version}                           |     |

---

## 📈 服务基准结果对比

| 指标 | {headers} |
|------|{separator}|
{serving_rows}

---

## ⏱️ 首 Token 延迟 (TTFT) 对比

| 指标 | {headers} |
|------|{separator}|
{ttft_rows}

---

## ⚡ 每 Token 生成时间 (TPOT) 对比

| 指标 | {headers} |
|------|{separator}|
{tpot_rows}

---

## 🔄 Token 间延迟 (ITL) 对比

| 指标 | {headers} |
|------|{separator}|
{itl_rows}

---

## 📊 模型性能对比

<img src="concurrency{concurrency}_comparison.png" width="1000" />

---

## 📝 分析小结

{analysis_content}

---

<div align="center">
*报告生成时间: {current_date}*
</div>
"""

    md_file = os.path.join(output_dir, f"concurrency{concurrency}_comparison.md")
    with open(md_file, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"Generated: {md_file}")
    return md_file


def main():
    parser = argparse.ArgumentParser(
        description="Compare model performance across different models"
    )
    parser.add_argument(
        "--chip",
        type=str,
        required=True,
        help="Chip platform (e.g., hygon_bw1000, nvidia_h100)",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model names to compare, separated by comma (e.g., MiniMax-M2.5-bf16,Qwen3.5-397B-A17B)",
    )
    parser.add_argument(
        "--test-suite",
        type=str,
        default="test_01",
        help="Test suite name (e.g., test_01)",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default="01",
        help="Run IDs for each model, separated by comma (e.g., 01 or 01,02)",
    )
    parser.add_argument(
        "-c",
        "--concurrency",
        type=str,
        default=None,
        help="Specific concurrency levels to compare, comma-separated (e.g., 1,2,4,8,10)",
    )

    args = parser.parse_args()

    args.chip = args.chip.lower()
    args.test_suite = args.test_suite.lower()
    args.run_id = args.run_id.lower()

    models = [m.strip() for m in args.model.split(",")]
    num_models = len(models)

    # 检查模型数量是否足够
    if num_models < 2:
        print(f"\nError: At least 2 models are required for comparison")
        print(f"Provided: {num_models} ({', '.join(models)})")
        print(f"Usage: --model model1,model2 or --model model1,model2,model3")
        return

    run_ids = parse_run_ids(args.run_id, num_models)
    if run_ids is None:
        print(
            f"Error: Invalid run-id format. Please provide either a single value or comma-separated values for {num_models} models"
        )
        return

    if len(run_ids) != num_models:
        print(
            f"Error: Number of run-ids ({len(run_ids)}) does not match number of models ({num_models})"
        )
        return

    chip = args.chip
    test_suite = args.test_suite

    print(f"\n{'=' * 60}")
    print(f"Model Comparison Configuration")
    print(f"{'=' * 60}")
    print(f"Chip: {chip}")
    print(f"Models: {', '.join(models)}")
    print(f"Test Suite: {test_suite}")
    print(f"Run IDs: {', '.join(run_ids)}")
    print(f"{'=' * 60}\n")

    # 验证模型目录是否存在（优先检查）
    benchmark_path = os.path.join("reports", "benchmark", chip)
    if not os.path.exists(benchmark_path):
        print(f"\nError: Benchmark path not found: {benchmark_path}")
        return

    available_models = [
        d
        for d in os.listdir(benchmark_path)
        if os.path.isdir(os.path.join(benchmark_path, d))
    ]

    # 检查每个模型目录是否存在
    missing_models = []
    for model_name in models:
        model_path = os.path.join(benchmark_path, model_name)
        if not os.path.exists(model_path):
            missing_models.append(model_name)

    if missing_models:
        print(f"\nError: Model directory not found for chip: {chip}")
        print(f"Expected models: {', '.join(models)}")
        print(f"Missing models: {', '.join(missing_models)}")
        print(f"\nAvailable model directories:")
        for m in available_models:
            print(f"  - {m}")
        return

    # 获取测试参数（只有在模型目录存在后才获取）
    test_params = get_test_params_from_yaml(test_suite)
    test_configs = test_params["test_configs"]
    concurrency_list = test_params["concurrency_list"]

    if not test_configs:
        print(f"No test configurations found in config file!")
        return

    print(f"Found {len(test_configs)} test configs from YAML")
    print(f"Concurrency list: {concurrency_list}")

    # 如果指定了并发级别，则过滤
    if args.concurrency:
        conc_list = [s.strip() for s in args.concurrency.split(",")]
        filtered_concs = [c for c in concurrency_list if str(c) in conc_list]
        if filtered_concs:
            concurrency_list = filtered_concs
            print(f"Using specified concurrency levels: {concurrency_list}")
        else:
            print(
                f"Warning: None of the specified concurrency levels {conc_list} found, using all"
            )

    print(f"Processing concurrency levels: {concurrency_list}")

    output_base = f"analysis/{chip}_comparison"
    Path(output_base).mkdir(parents=True, exist_ok=True)

    generated_reports = []

    for test_config in test_configs:
        print(f"\n=== Processing test config: {test_config} ===")

        for concurrency in concurrency_list:
            config_with_concurrency = f"{concurrency}-{test_config}"
            print(f"\n--- Processing concurrency: {concurrency} ---")
            print(
                f"    Looking for: benchmark/{chip}/{models[0]}/{test_suite}/{run_ids[0]}/{config_with_concurrency}"
            )

            output_dir = os.path.join(output_base, test_suite, config_with_concurrency)
            Path(output_dir).mkdir(parents=True, exist_ok=True)

            models_data = {}
            for i, model_name in enumerate(models):
                rid = run_ids[i]
                model_dir = os.path.join(
                    "reports", "benchmark", chip, model_name, test_suite, rid
                )
                config_path = os.path.join(model_dir, config_with_concurrency)
                print(f"    Checking path: {config_path}")

                metrics = get_model_data(
                    chip, model_name, test_suite, rid, config_with_concurrency
                )
                if metrics:
                    normalized_metrics = {}
                    for key, value in metrics.items():
                        normalized_metrics[key.lower()] = value
                    models_data[model_name] = normalized_metrics
                    print(f"    - {model_name} (run-id: {rid}): [OK]")
                else:
                    print(f"    - {model_name} (run-id: {rid}): [NOT FOUND]")

            if not models_data:
                print(f"    No data found for concurrency {concurrency}")
                continue

            print(f"    Comparing {len(models_data)} models, generating reports...")

            generate_comparison_csv(
                models_data, config_with_concurrency, output_dir, concurrency
            )
            generate_comparison_charts(
                models_data, config_with_concurrency, output_dir, concurrency
            )
            generate_comparison_markdown(
                models_data,
                config_with_concurrency,
                output_dir,
                chip,
                test_suite,
                run_ids,
                concurrency_list,
                concurrency,
                chip,
                models,
            )

            generated_reports.append((config_with_concurrency, concurrency))

    generate_summary_report(output_base, test_suite, generated_reports)

    print(f"\n{'=' * 60}")
    print("Model comparison reports generated successfully!")
    print(f"Output directory: {output_base}")
    print(f"{'=' * 60}")


def generate_summary_report(output_base, test_suite, generated_reports):
    current_date = datetime.now().strftime("%Y-%m-%d")

    sorted_reports = sorted(generated_reports, key=lambda x: x[1])

    summary_lines = ["# 多模型性能对比汇总报告\n"]
    summary_lines.append(f"\n**测试日期：** {current_date}\n")
    summary_lines.append(f"**测试套件：** {test_suite}\n")
    summary_lines.append("---\n")

    summary_lines.append("## 报告列表\n")
    summary_lines.append("| 并发级别 | 配置文件 | 报告链接 |")
    summary_lines.append("|----------|----------|----------|")

    for config_key, concurrency in sorted_reports:
        md_link = f"[详细报告](./{config_key}/concurrency{concurrency}_comparison.md)"
        summary_lines.append(f"| {concurrency} | {config_key} | {md_link} |")

    summary_lines.append("\n---\n")
    summary_lines.append(f"*共生成 {len(generated_reports)} 个并发级别的对比报告*\n")

    summary_file = os.path.join(output_base, test_suite, "summary.md")
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    print(f"Generated summary: {summary_file}")


if __name__ == "__main__":
    main()
