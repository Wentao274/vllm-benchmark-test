import os
import re
import glob
import yaml
from pathlib import Path
from datetime import datetime

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("matplotlib not available, skipping chart generation")


def load_model_deployment_config(config_path="config/model_deployment.yaml"):
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            return config.get('models', {})
    return {}


def get_deployment_detail(model_name, deployment_config):
    config = deployment_config.get(model_name, {})
    if not config:
        return "未知"
    
    nodes = config.get('nodes', 'N/A')
    tp = config.get('tp', 'N/A')
    dp = config.get('dp', 'N/A')
    pp = config.get('pp', 'N/A')
    ep = "yes" if config.get('ep_enabled', False) else "no"
    kv_offload = "yes" if config.get('kv_cache_offload', False) else "no"
    lmcache = "yes" if config.get('lmcache', False) else "no"
    
    detail = f"节点数: {nodes}<br>TP={tp}, DP={dp}, PP={pp}, EP enabled: {ep}<br>KVCache Offload: {kv_offload}<br>LMcache: {lmcache}"
    return detail


def parse_benchmark_log(log_file):
    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    lines = content.split('\n')
    metrics = {}
    
    in_results = False
    for i, line in enumerate(lines):
        if '=========== Serving Benchmark Result' in line:
            in_results = True
            continue
        if in_results and line.strip().startswith('==========='):
            break
        if in_results:
            match = re.match(r'(.+?):\s+(.+)$', line.strip())
            if match:
                key = match.group(1).strip()
                value = match.group(2).strip()
                metrics[key] = value
    
    return metrics


def extract_test_config_from_path(path):
    parts = path.split(os.sep)
    for part in parts:
        if re.match(r'^\d+-\d+-i\d+-o\d+$', part):
            return part
    return None


def extract_concurrency_from_config(config):
    match = re.match(r'^(\d+)-\d+-i\d+-o\d+$', config)
    if match:
        return int(match.group(1))
    return None


def get_all_test_configs(reports_dir):
    test_configs = set()
    model_dirs = glob.glob(os.path.join(reports_dir, "*"))
    
    for model_dir in model_dirs:
        if os.path.isdir(model_dir):
            timestamp_dirs = glob.glob(os.path.join(model_dir, "*"))
            for ts_dir in timestamp_dirs:
                if os.path.isdir(ts_dir):
                    config_dirs = glob.glob(os.path.join(ts_dir, "*"))
                    for config_dir in config_dirs:
                        if os.path.isdir(config_dir):
                            config = extract_test_config_from_path(config_dir)
                            if config:
                                test_configs.add(config)
    
    return sorted(test_configs, key=lambda x: int(extract_concurrency_from_config(x) or 0))


def get_models_with_config(reports_dir, test_config):
    models_data = {}
    
    model_dirs = glob.glob(os.path.join(reports_dir, "*"))
    for model_dir in model_dirs:
        if not os.path.isdir(model_dir):
            continue
        
        model_name = os.path.basename(model_dir)
        ts_dirs = glob.glob(os.path.join(model_dir, "*"))
        
        for ts_dir in ts_dirs:
            if not os.path.isdir(ts_dir):
                continue
            
            config_dir = os.path.join(ts_dir, test_config)
            if os.path.isdir(config_dir):
                log_files = glob.glob(os.path.join(config_dir, "*.log"))
                if log_files:
                    log_file = log_files[0]
                    metrics = parse_benchmark_log(log_file)
                    models_data[model_name] = metrics
                    break
    
    return models_data


def generate_comparison_charts(models_data, test_config, output_dir):
    if not HAS_MATPLOTLIB:
        return None
    
    concurrency = extract_concurrency_from_config(test_config)
    sorted_models = sorted(models_data.keys())
    
    req_throughput = [float(models_data[m].get('Request throughput (req/s)', 0) or 0) for m in sorted_models]
    total_tput = [float(models_data[m].get('Total token throughput (tok/s)', 0) or 0) for m in sorted_models]
    ttft_p99 = [float(models_data[m].get('P99 TTFT (ms)', 0) or 0) for m in sorted_models]
    tpot_p99 = [float(models_data[m].get('P99 TPOT (ms)', 0) or 0) for m in sorted_models]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Model Comparison @ {concurrency} Concurrency', fontsize=14, fontweight='bold')
    
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6', '#f39c12'][:len(sorted_models)]
    
    axes[0, 0].bar(sorted_models, req_throughput, color=colors, alpha=0.8)
    axes[0, 0].set_title('Request Throughput (req/s)', fontsize=11)
    axes[0, 0].set_ylabel('req/s')
    axes[0, 0].tick_params(axis='x', rotation=15)
    for i, v in enumerate(req_throughput):
        axes[0, 0].text(i, v + 0.05 * max(req_throughput) if req_throughput else 0.1, f'{v:.2f}', 
                        ha='center', va='bottom', fontsize=9)
    
    axes[0, 1].bar(sorted_models, total_tput, color=colors, alpha=0.8)
    axes[0, 1].set_title('Total Token Throughput (tok/s)', fontsize=11)
    axes[0, 1].set_ylabel('tok/s')
    axes[0, 1].tick_params(axis='x', rotation=15)
    for i, v in enumerate(total_tput):
        axes[0, 1].text(i, v + 0.02 * max(total_tput) if total_tput else 1000, f'{v:.0f}', 
                        ha='center', va='bottom', fontsize=9)
    
    axes[1, 0].bar(sorted_models, ttft_p99, color=colors, alpha=0.8)
    axes[1, 0].set_title('TTFT P99 (ms)', fontsize=11)
    axes[1, 0].set_ylabel('ms')
    axes[1, 0].tick_params(axis='x', rotation=15)
    for i, v in enumerate(ttft_p99):
        axes[1, 0].text(i, v + 0.02 * max(ttft_p99) if ttft_p99 else 100, f'{v:.0f}', 
                        ha='center', va='bottom', fontsize=9)
    
    axes[1, 1].bar(sorted_models, tpot_p99, color=colors, alpha=0.8)
    axes[1, 1].set_title('TPOT P99 (ms)', fontsize=11)
    axes[1, 1].set_ylabel('ms')
    axes[1, 1].tick_params(axis='x', rotation=15)
    for i, v in enumerate(tpot_p99):
        axes[1, 1].text(i, v + 0.02 * max(tpot_p99) if tpot_p99 else 5, f'{v:.2f}', 
                        ha='center', va='bottom', fontsize=9)
    
    for ax in axes.flat:
        ax.grid(axis='y', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    chart_file = os.path.join(output_dir, f'concurrency{concurrency}_comparison.png')
    plt.savefig(chart_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Generated chart: {chart_file}")
    return chart_file


def generate_comparison_csv(model_name, models_data, test_config, output_dir):
    concurrency = extract_concurrency_from_config(test_config)
    
    metric_names = [
        ("[Serving Benchmark Result]", ""),
        ("Successful requests", "Successful requests"),
        ("Failed requests", "Failed requests"),
        ("Benchmark duration (s)", "Benchmark duration (s)"),
        ("Total input tokens", "Total input tokens"),
        ("Total generated tokens", "Total generated tokens"),
        ("Request throughput (req/s)", "Request throughput (req/s)"),
        ("Output token throughput (tok/s)", "Output token throughput (tok/s)"),
        ("Peak output token throughput (tok/s)", "Peak output token throughput (tok/s)"),
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
    with open(csv_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(csv_lines))
    
    print(f"Generated: {csv_file}")
    return csv_file


def generate_comparison_markdown(model_name, models_data, test_config, output_dir, deployment_config=None):
    concurrency = extract_concurrency_from_config(test_config)
    current_date = datetime.now().strftime("%Y-%m-%d")
    
    sorted_models = sorted(models_data.keys())
    headers = " | ".join(sorted_models)
    separator = " | ".join(["-----------"] * len(sorted_models))
    
    def make_row(key_name):
        cells = []
        for model in sorted_models:
            value = models_data[model].get(key_name, '')
            cells.append(value)
        return " | ".join(cells)
    
    def make_throughput_row(key_name, highlight_max=True):
        cells = []
        values = []
        for model in sorted_models:
            value = models_data[model].get(key_name, '')
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
            value = models_data[model].get(key_name, '')
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
        throughputs = [(m, float(models_data[m].get('Request throughput (req/s)', 0) or 0)) for m in sorted_models]
        max_tp = max(throughputs, key=lambda x: x[1])
        analysis_lines.append(f"- **请求吞吐量**: {max_tp[0]} 最高，达 {max_tp[1]:.2f} req/s")
    except:
        pass
    
    try:
        total_tputs = [(m, float(models_data[m].get('Total token throughput (tok/s)', 0) or 0)) for m in sorted_models]
        max_total = max(total_tputs, key=lambda x: x[1])
        analysis_lines.append(f"- **总token吞吐量**: {max_total[0]} 最高，达 {max_total[1]:.0f} tok/s")
    except:
        pass
    
    try:
        ttft_p99 = [(m, float(models_data[m].get('P99 TTFT (ms)', float('inf')) or float('inf'))) for m in sorted_models]
        min_ttft = min(ttft_p99, key=lambda x: x[1])
        analysis_lines.append(f"- **TTFT P99**: {min_ttft[0]} 最优，为 {min_ttft[1]:.2f}ms")
    except:
        pass
    
    try:
        tpot_p99 = [(m, float(models_data[m].get('P99 TPOT (ms)', float('inf')) or float('inf'))) for m in sorted_models]
        min_tpot = min(tpot_p99, key=lambda x: x[1])
        analysis_lines.append(f"- **TPOT P99**: {min_tpot[0]} 最优，为 {min_tpot[1]:.2f}ms")
    except:
        pass
    
    analysis_content = "\n".join(analysis_lines) if analysis_lines else "- 各模型性能表现待分析"
    
    md_content = f"""# 多模型性能对比报告

<div align="center">
**测试日期：** {current_date}
**并发级别：** {concurrency}并发
**测试配置：** {test_config}
</div>

---

## 📊 模型列表

| 模型名称 | 部署详情 | 状态 |
|----------|----------|------|
"""
    
    for model in sorted_models:
        detail = get_deployment_detail(model, deployment_config or {})
        md_content += f"| {model} | {detail} | ✅ 已加载 |\n"
    
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
    with open(md_file, 'w', encoding='utf-8') as f:
        f.write(md_content)
    
    print(f"Generated: {md_file}")
    return md_file


def main():
    reports_dir = "reports/benchmark"
    output_base = "analysis/model_comparison"
    
    Path(output_base).mkdir(parents=True, exist_ok=True)
    
    test_configs = get_all_test_configs(reports_dir)
    
    if not test_configs:
        print("No test configurations found!")
        return
    
    print(f"Found {len(test_configs)} test configurations: {', '.join(test_configs)}")
    
    deployment_config = load_model_deployment_config()
    
    for test_config in test_configs:
        print(f"\nProcessing config: {test_config}")
        
        output_dir = os.path.join(output_base, test_config)
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        models_data = get_models_with_config(reports_dir, test_config)
        
        if not models_data:
            print(f"  No data found for config {test_config}")
            continue
        
        print(f"  Found {len(models_data)} models: {', '.join(models_data.keys())}")
        
        generate_comparison_csv("comparison", models_data, test_config, output_dir)
        generate_comparison_charts(models_data, test_config, output_dir)
        generate_comparison_markdown("comparison", models_data, test_config, output_dir, deployment_config)
    
    generate_summary_report(output_base, test_configs, reports_dir)
    
    print(f"\n{'='*50}")
    print("Model comparison reports generated successfully!")
    print(f"Output directory: {output_base}")
    print(f"{'='*50}")


def generate_summary_report(output_base, test_configs, reports_dir):
    summary_lines = []
    summary_lines.append("# 多模型性能汇总对比报告\n")
    summary_lines.append(f"**生成时间:** {datetime.now().strftime('%Y-%m-%d')}\n")
    summary_lines.append("---\n")
    summary_lines.append("## 各并发级别报告链接\n")
    
    for config in test_configs:
        concurrency = extract_concurrency_from_config(config)
        md_link = f"./{config}/concurrency{concurrency}_comparison.md"
        csv_link = f"./{config}/concurrency{concurrency}_comparison.csv"
        summary_lines.append(f"- [{config} (CSV)]({csv_link}) | [Markdown]({md_link})")
    
    summary_file = os.path.join(output_base, "summary.md")
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(summary_lines))
    
    print(f"Generated summary: {summary_file}")


if __name__ == "__main__":
    main()
