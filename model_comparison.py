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
            test_configs, key=lambda x: int(extract_concurrency_from_config(x) or 0)
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
        reports_base, chip, "benchmark", model_name, test_suite, run_id
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
        reports_base, chip, "benchmark", model_name, test_suite, run_id
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

    sorted_models = sorted(models_data.keys())
    headers = " | ".join(sorted_models)
    separator = " | ".join(["-----------"] * len(sorted_models))

    chip_table_rows = []
    chip_info_map = {}
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
    for param in chip_param_names:
        row = f"| **{param}** |"
        for model in sorted_models:
            cfg = load_chip_config_by_model(chip_name, model)
            val = cfg.get(param, "N/A")
            row += f" {val} |"
        chip_table_rows.append(row)
    chip_table = "\n".join(chip_table_rows)

    vllm_table_rows = []
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
    for param in vllm_param_names:
        row = f"| {param} |"
        for model in sorted_models:
            cfg = load_vllm_config_by_model(chip_name, model)
            val = cfg.get(param, "N/A")
            row += f" {val} |"
        vllm_table_rows.append(row)
    vllm_table = "\n".join(vllm_table_rows)

    def make_row(key_name):
        cells = []
        for model in sorted_models:
            value = models_data[model].get(key_name.lower(), "")
            cells.append(value if value else "0")
        return " | ".join(cells)

    def make_throughput_row(key_name, highlight_max=True):
        cells = []
        values = []
        for model in sorted_models:
            value = models_data[model].get(key_name.lower(), "")
            values.append((model, value))
        cells = []
        if highlight_max:
            try:
                numeric = [(m, float(v)) for m, v in values if v]
                if numeric:
                    max_val = max(numeric, key=lambda x: x[1])
                    for m, v in values:
                        if v and float(v) == max_val[1]:
                            cells.append(f"**{v}** ⭐")
                        else:
                            cells.append(v)
                else:
                    cells = [v for _, v in values]
            except:
                cells = [v for _, v in values]
        else:
            cells = [v for _, v in values]
        return " | ".join(cells)

    def make_latency_row(key_name, highlight_min=True):
        cells = []
        values = []
        for model in sorted_models:
            value = models_data[model].get(key_name.lower(), "")
            values.append((model, value))
        cells = []
        if highlight_min:
            try:
                numeric = [(m, float(v)) for m, v in values if v]
                if numeric:
                    min_val = min(numeric, key=lambda x: x[1])
                    for m, v in values:
                        if v and float(v) == min_val[1]:
                            cells.append(f"**{v}** ⭐")
                        else:
                            cells.append(v)
                else:
                    cells = [v for _, v in values]
            except:
                cells = [v for _, v in values]
        else:
            cells = [v for _, v in values]
        return " | ".join(cells)

    serving_rows = f"""| 成功请求数 | {make_row("Successful requests")} |
| 失败请求数 | {make_row("Failed requests")} |
| 测试持续时间 (s) | {make_row("Benchmark duration (s)")} |
| 总输入 tokens | {make_row("Total input tokens")} |
| 总生成 tokens | {make_row("Total generated tokens")} |
| **请求吞吐量 (req/s)** | {make_throughput_row("Request throughput (req/s)")} |
| **输出 token 吞吐量 (tok/s)** | {make_throughput_row("Output token throughput (tok/s)")} |
| 峰值输出 token 吞吐量 (tok/s) | {make_throughput_row("Peak output token throughput (tok/s)")} |
| 峰值并发请求数 | {make_row("Peak concurrent requests")} |
| **总 token 吞吐量 (tok/s)** | {make_throughput_row("Total token throughput (tok/s)")} |"""

    ttft_rows = f"""| 平均 TTFT (ms) | {make_latency_row("Mean TTFT (ms)")} |
| 中位 TTFT (ms) | {make_latency_row("Median TTFT (ms)")} |
| P95 TTFT (ms) | {make_latency_row("P95 TTFT (ms)")} |
| P99 TTFT (ms) | {make_latency_row("P99 TTFT (ms)")} |"""

    tpot_rows = f"""| 平均 TPOT (ms) | {make_latency_row("Mean TPOT (ms)")} |
| 中位 TPOT (ms) | {make_latency_row("Median TPOT (ms)")} |
| P95 TPOT (ms) | {make_latency_row("P95 TPOT (ms)")} |
| P99 TPOT (ms) | {make_latency_row("P99 TPOT (ms)")} |"""

    itl_rows = f"""| 平均 ITL (ms) | {make_latency_row("Mean ITL (ms)")} |
| 中位 ITL (ms) | {make_latency_row("Median ITL (ms)")} |
| P95 ITL (ms) | {make_latency_row("P95 ITL (ms)")} |
| P99 ITL (ms) | {make_latency_row("P99 ITL (ms)")} |"""

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

| 芯片名称                        | **{"** | **".join(sorted_models)}** |
|-----------------------------|{("-------------------------------|") * len(sorted_models)}
{chip_table}

---

## 🤖 vLLM启动配置信息

| 参数名称                    | **{"** | **".join(sorted_models)}** |
|-------------------------|{("-------------------|") * len(sorted_models)}
{vllm_table}

---

## 📊 模型列表

| 模型名称 | Run ID | 状态 |
|----------|--------|------|
"""

    for i, model in enumerate(sorted_models):
        rid = run_ids[i] if i < len(run_ids) else "N/A"
        md_content += f"| {model} | {rid} | [OK] |\n"

    md_content += f"""
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

![Model Performance Comparison](./concurrency{concurrency}_comparison.png)

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

    args = parser.parse_args()

    args.chip = args.chip.lower()
    args.test_suite = args.test_suite.lower()
    args.run_id = args.run_id.lower()

    models = [m.strip() for m in args.model.split(",")]
    num_models = len(models)

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

    test_params = get_test_params_from_yaml(test_suite)
    test_configs = test_params["test_configs"]
    concurrency_list = test_params["concurrency_list"]

    if not test_configs:
        print(f"No test configurations found in config file!")
        return

    print(f"Found {len(test_configs)} test configs from YAML")
    print(f"Concurrency list: {concurrency_list}")

    output_base = f"analysis/{chip}_comparison"
    Path(output_base).mkdir(parents=True, exist_ok=True)

    generated_reports = []

    for test_config in test_configs:
        print(f"\n=== Processing test config: {test_config} ===")

        for concurrency in concurrency_list:
            config_with_concurrency = f"{concurrency}-{test_config}"
            print(f"\n--- Processing concurrency: {concurrency} ---")
            print(
                f"    Looking for: {chip}/benchmark/{models[0]}/{test_suite}/{run_ids[0]}/{config_with_concurrency}"
            )

            output_dir = os.path.join(output_base, test_suite, config_with_concurrency)
            Path(output_dir).mkdir(parents=True, exist_ok=True)

            models_data = {}
            for i, model_name in enumerate(models):
                rid = run_ids[i]
                model_dir = os.path.join(
                    "reports", chip, "benchmark", model_name, test_suite, rid
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
    summary_lines = []
    summary_lines.append("# 多模型性能汇总对比报告\n")
    summary_lines.append(f"**生成时间:** {datetime.now().strftime('%Y-%m-%d')}\n")
    summary_lines.append("---\n")
    summary_lines.append("## 各并发级别报告链接\n")

    sorted_reports = sorted(generated_reports, key=lambda x: x[1])
    for config_key, concurrency in sorted_reports:
        csv_link = f"./{config_key}/concurrency{concurrency}_comparison.csv"
        md_link = f"./{config_key}/concurrency{concurrency}_comparison.md"
        summary_lines.append(
            f"- [{config_key} (CSV)]({csv_link}) | [Markdown]({md_link})"
        )

    summary_file = os.path.join(output_base, test_suite, "summary.md")
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    print(f"Generated summary: {summary_file}")


if __name__ == "__main__":
    main()
