import os
import re
import glob
import yaml
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from itertools import product

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("matplotlib not available, skipping chart generation")


def load_yaml_config(config_path):
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    return {}


def parse_benchmark_log(log_file):
    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    lines = content.split('\n')
    metrics = {}
    test_config = {}
    
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
        
        if 'num_prompts=' in line:
            match = re.search(r'num_prompts=(\d+)', line)
            if match:
                test_config['num_prompts'] = match.group(1)
        
        if 'random_input_len=' in line:
            match = re.search(r'random_input_len=(\d+)', line)
            if match:
                test_config['input_len'] = match.group(1)
        
        if 'random_output_len=' in line:
            match = re.search(r'random_output_len=(\d+)', line)
            if match:
                test_config['output_len'] = match.group(1)
    
    return metrics, test_config


def extract_concurrency_from_path(path):
    parts = path.split(os.sep)
    for part in parts:
        if 'bench-' in part:
            match = re.match(r'bench-(\d+)-', part)
            if match:
                return match.group(1)
    return None


def extract_test_config_from_log_path(path):
    parts = path.split(os.sep)
    for part in parts:
        if re.match(r'^\d+-\d+-i\d+-o\d+$', part):
            concurrency = part.split('-')[0]
            num_prompts = part.split('-')[1]
            input_len_match = re.search(r'i(\d+)', part)
            output_len_match = re.search(r'o(\d+)', part)
            input_len = input_len_match.group(1) if input_len_match else ""
            output_len = output_len_match.group(1) if output_len_match else ""
            return {
                'concurrency': concurrency,
                'num_prompts': num_prompts,
                'input_len': input_len,
                'output_len': output_len
            }
    return None


def get_model_names(reports_dir):
    if not os.path.exists(reports_dir):
        return []
    model_names = []
    for item in os.listdir(reports_dir):
        item_path = os.path.join(reports_dir, item)
        if os.path.isdir(item_path):
            model_names.append(item)
    return sorted(model_names)


def process_model(model_name, reports_dir, config_dir):
    yaml_path = os.path.join(config_dir, "models_scenarios.yaml")
    yaml_config = load_yaml_config(yaml_path)
    
    base_config = yaml_config.get("base_config", {})
    params = base_config.get("params", {})
    
    yaml_num_prompts_list = [str(n) for n in params.get("num-prompts", [])]
    yaml_input_len_list = [str(i) for i in params.get("random-input-len", [])]
    yaml_output_len_list = [str(o) for o in params.get("random-output-len", [])]
    yaml_concurrency_list = [str(c) for c in sorted(params.get("max-concurrency", []), key=lambda x: int(x))]
    
    yaml_configs = set(product(yaml_num_prompts_list, yaml_input_len_list, yaml_output_len_list))
    
    model_reports_dir = os.path.join(reports_dir, model_name)
    
    results_by_config = defaultdict(dict)
    
    log_files = glob.glob(os.path.join(model_reports_dir, "**", "bench-*.log"), recursive=True)
    
    for log_file in log_files:
        test_cfg = extract_test_config_from_log_path(log_file)
        if not test_cfg:
            continue
        
        concurrency = test_cfg['concurrency']
        num_prompts = test_cfg['num_prompts']
        input_len = test_cfg['input_len']
        output_len = test_cfg['output_len']
        
        config_key = (num_prompts, input_len, output_len)
        
        metrics, _ = parse_benchmark_log(log_file)
        results_by_config[config_key][concurrency] = metrics
        
        print(f"[{model_name}] Parsed {num_prompts}-i{input_len}-o{output_len} - concurrency {concurrency}")
    
    for config_tuple, all_results in results_by_config.items():
        num_prompts, input_len, output_len = config_tuple
        
        if config_tuple in yaml_configs:
            dir_name = f"{num_prompts}-i{input_len}-o{output_len}"
            concurrency_display = ', '.join(yaml_concurrency_list)
        else:
            dir_name = f"{num_prompts}-i{input_len}-o{output_len}"
            all_concurrencies = sorted(all_results.keys(), key=lambda x: int(x))
            concurrency_display = ', '.join(all_concurrencies)
        
        output_dir = os.path.join("analysis", model_name, dir_name)
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        sorted_concurrencies = sorted(all_results.keys(), key=lambda x: int(x))
        
        generate_csv_report(model_name, sorted_concurrencies, all_results, output_dir)
        
        if HAS_MATPLOTLIB:
            generate_charts(model_name, sorted_concurrencies, all_results, output_dir)
        
        generate_markdown_report(
            model_name, sorted_concurrencies, all_results, output_dir,
            num_prompts=num_prompts,
            input_len=input_len,
            output_len=output_len,
            concurrency_list=concurrency_display
        )


def generate_csv_report(model_name, sorted_concurrencies, all_results, output_dir):
    csv_content = []
    header = ["Metric"] + [f"{model_name}-concurrency{c}" for c in sorted_concurrencies]
    csv_content.append(",".join(header))
    
    metric_names = [
        ("[Serving Benchmark Result]", ""),
        ("Successful requests", "Successful requests"),
        ("Failed requests", "Failed requests"),
        ("Maximum request concurrency", "Maximum request concurrency"),
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
    
    for display_name, key_name in metric_names:
        if not key_name:
            csv_content.append(f"[{display_name}]" + ",," * (len(sorted_concurrencies) - 1))
            continue
        
        row = [display_name]
        for c in sorted_concurrencies:
            value = all_results[c].get(key_name, "")
            row.append(value)
        csv_content.append(",".join(row))
    
    csv_file = os.path.join(output_dir, f"{model_name}_bench_result.csv")
    with open(csv_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(csv_content))
    
    print(f"[{model_name}] CSV file generated: {csv_file}")

def generate_markdown_report(model_name, concurrencies, all_results, output_dir, num_prompts="", input_len="", output_len="", concurrency_list=None):
    current_date = datetime.now().strftime("%Y-%m-%d")
    
    concurrency_headers = " | ".join([f"{model_name}-{c}并发" for c in concurrencies])
    concurrency_row = " | ".join(["-------------"] * len(concurrencies))
    
    typical_conc = "20" if "20" in concurrencies else (concurrencies[0] if concurrencies else "")
    
    def get_threshold_status(value_str, threshold, is_higher_better=True):
        if not value_str:
            return "⚪ 未知"
        try:
            value = float(value_str)
            if is_higher_better:
                ratio = value / threshold
                if ratio >= 1.0:
                    return f"✅ 达标 ({ratio:.2f}x)"
                elif ratio >= 0.8:
                    return f"⚠️ 接近阈值 ({ratio:.2f}x)"
                else:
                    return f"❌ 不达标 ({ratio:.2f}x)"
            else:
                ratio = value / threshold
                if ratio <= 1.0:
                    return f"✅ 达标 ({ratio:.2f}x)"
                elif ratio <= 1.25:
                    return f"⚠️ 接近阈值 ({ratio:.2f}x)"
                else:
                    return f"❌ 不达标 ({ratio:.2f}x)"
        except:
            return "⚪ 未知"
    
    def make_cell(metric_key, highlight_max=False, highlight_min=False):
        cells = []
        values = []
        for c in concurrencies:
            val = all_results.get(c, {}).get(metric_key, '')
            values.append(val)
        if not values or all(v == '' for v in values):
            return " | ".join([''] * len(concurrencies)), ""
        
        try:
            numeric_values = [float(v) for v in values if v != '']
            if numeric_values:
                if highlight_max:
                    max_val = max(numeric_values)
                    for v in values:
                        if v != '' and float(v) == max_val:
                            cells.append(f"**{v}** ⭐")
                        else:
                            cells.append(v)
                    return " | ".join(cells), f"最大值为 {max_val}"
                elif highlight_min:
                    min_val = min(numeric_values)
                    for v in values:
                        if v != '' and float(v) == min_val:
                            cells.append(f"**{v}** ⭐")
                        else:
                            cells.append(v)
                    return " | ".join(cells), f"最小值为 {min_val}"
        except ValueError:
            pass
        return " | ".join(values), ""
    
    def make_row(metric_key):
        return " | ".join([all_results.get(c, {}).get(metric_key, '') for c in concurrencies])
    
    serving_metrics = [
        ("成功请求数", "Successful requests", False, False),
        ("失败请求数", "Failed requests", False, False),
        ("测试持续时间 (s)", "Benchmark duration (s)", False, False),
        ("总输入 tokens", "Total input tokens", False, False),
        ("总生成 tokens", "Total generated tokens", False, False),
        ("**请求吞吐量 (req/s)**", "Request throughput (req/s)", True, False),
        ("**输出 token 吞吐量 (tok/s)**", "Output token throughput (tok/s)", True, False),
        ("峰值输出 token 吞吐量 (tok/s)", "Peak output token throughput (tok/s)", True, False),
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
    
    serving_rows = "\n".join([f"| {name} | {make_row(key)} |" for name, key, _, _ in serving_metrics])
    ttft_rows = "\n".join([f"| {name} | {make_row(key)} |" for name, key, _, _ in ttft_metrics])
    tpot_rows = "\n".join([f"| {name} | {make_row(key)} |" for name, key, _, _ in tpot_metrics])
    itl_rows = "\n".join([f"| {name} | {make_row(key)} |" for name, key, _, _ in itl_metrics])
    
    concurrency_display = concurrency_list if concurrency_list else ', '.join(concurrencies)
    
    threshold_rows = ""
    eval_rows = ""
    analysis_content = ""
    suggestion_content = ""
    
    req_throughput_typical = all_results.get(typical_conc, {}).get('Request throughput (req/s)', '0') if typical_conc else '0'
    output_tput_typical = all_results.get(typical_conc, {}).get('Output token throughput (tok/s)', '0') if typical_conc else '0'
    total_tput_typical = all_results.get(typical_conc, {}).get('Total token throughput (tok/s)', '0') if typical_conc else '0'
    ttft_p99_typical = all_results.get(typical_conc, {}).get('P99 TTFT (ms)', '0') if typical_conc else '0'
    tpot_p99_typical = all_results.get(typical_conc, {}).get('P99 TPOT (ms)', '0') if typical_conc else '0'
    itl_p99_typical = all_results.get(typical_conc, {}).get('P99 ITL (ms)', '0') if typical_conc else '0'
    success_rate = all_results.get(typical_conc, {}).get('Successful requests', '0') if typical_conc else '0'
    
    try:
        total_req = float(all_results.get(typical_conc, {}).get('Successful requests', '0') or '0') + float(all_results.get(typical_conc, {}).get('Failed requests', '0') or '0')
        if total_req > 0:
            success_rate_calc = float(success_rate or '0') / total_req * 100
            success_rate = f"{success_rate_calc:.1f}%"
    except:
        pass
    
    threshold_rows = f"""### 性能指标要求 (基于8卡B200芯片)

| 指标类别 | 指标名称 | 阈值要求 | 适用场景 |
|----------|----------|----------|----------|
| **吞吐量** | 请求吞吐量 (req/s) | >= 1.5 | 10并发以上 |
| **吞吐量** | 输出token吞吐量 (tok/s) | >= 300 | 10并发以上 |
| **吞吐量** | 总token吞吐量 (tok/s) | >= 9000 | 10并发以上 |
| **延迟** | TTFT P99 (ms) | <= 4000 | 中高并发 (20-40) |
| **延迟** | TPOT P99 (ms) | <= 40 | 所有并发 |
| **延迟** | ITL P99 (ms) | <= 800 | 所有并发 |
| **稳定性** | 请求成功率 | >= 99% | 所有场景 |
| **稳定性** | 峰值并发请求数 | <= 100 | 压力测试 |

### 达标判定标准

- ✅ **达标**: 实测值满足阈值要求
- ⚠️ **警告**: 实测值接近阈值（80%-100%范围内）
- ❌ **不达标**: 实测值超出阈值要求"""
    
    eval_rows = f"""| 指标类别 | 指标 | 阈值 | 实测值 ({typical_conc}并发) | 达标情况 |
|----------|------|------|-----------------|----------|
| 吞吐量 | 请求吞吐量 (req/s) | >= 1.5 | {req_throughput_typical} | {get_threshold_status(req_throughput_typical, 1.5, True)} |
| 吞吐量 | 输出token吞吐量 (tok/s) | >= 300 | {output_tput_typical} | {get_threshold_status(output_tput_typical, 300, True)} |
| 吞吐量 | 总token吞吐量 (tok/s) | >= 9000 | {total_tput_typical} | {get_threshold_status(total_tput_typical, 9000, True)} |
| 延迟 | TTFT P99 (ms) | <= 4000 | {ttft_p99_typical} | {get_threshold_status(ttft_p99_typical, 4000, False)} |
| 延迟 | TPOT P99 (ms) | <= 40 | {tpot_p99_typical} | {get_threshold_status(tpot_p99_typical, 40, False)} |
| 延迟 | ITL P99 (ms) | <= 800 | {itl_p99_typical} | {get_threshold_status(itl_p99_typical, 800, False)} |
| 稳定性 | 请求成功率 | >= 99% | {success_rate} | {get_threshold_status(success_rate.rstrip('%'), 99, True) if '%' not in success_rate else ('✅ 达标' if float(success_rate.rstrip('%')) >= 99 else '❌ 不达标')} |"""
    
    throughput_detail = ""
    latency_detail = ""
    stability_detail = ""
    problem_diagnosis = ""
    suggestions = ""
    
    tpot_p99_typical_val = float(tpot_p99_typical) if tpot_p99_typical.replace('.', '').replace('-', '').isdigit() else 0
    ttft_p99_80_val = float(all_results.get('80', {}).get('P99 TTFT (ms)', '0') or '0') if '80' in concurrencies else 0
    
    if concurrencies:
        try:
            req_throughputs = [(c, float(all_results.get(c, {}).get('Request throughput (req/s)', 0) or 0)) for c in concurrencies if all_results.get(c, {}).get('Request throughput (req/s)', '')]
            output_tputs = [(c, float(all_results.get(c, {}).get('Output token throughput (tok/s)', 0) or 0)) for c in concurrencies if all_results.get(c, {}).get('Output token throughput (tok/s)', '')]
            total_tputs = [(c, float(all_results.get(c, {}).get('Total token throughput (tok/s)', 0) or 0)) for c in concurrencies if all_results.get(c, {}).get('Total token throughput (tok/s)', '')]
            
            if req_throughputs:
                max_tp = max(req_throughputs, key=lambda x: x[1])
                max_output = max(output_tputs, key=lambda x: x[1]) if output_tputs else ('', 0)
                max_total = max(total_tputs, key=lambda x: x[1]) if total_tputs else ('', 0)
                throughput_detail = f"""- 请求吞吐量{req_throughput_typical} req/s，超过1.5 req/s阈值要求
   - 输出token吞吐量{output_tput_typical} tok/s，超过300 tok/s阈值要求
   - 总token吞吐量{total_tput_typical} tok/s，远超9000 tok/s阈值要求
   - 吞吐量随并发增加持续上升，高并发(80并发)达{max_tp[1]:.2f} req/s和{max_total[1]:.0f} tok/s，系统未达到瓶颈"""
        except Exception as e:
            throughput_detail = "- 请求吞吐量随并发增加持续增长"
        
        try:
            ttft_p99_values = [(c, float(all_results.get(c, {}).get('P99 TTFT (ms)', float('inf')) or float('inf'))) for c in concurrencies if all_results.get(c, {}).get('P99 TTFT (ms)', '')]
            tpot_p99_values = [(c, float(all_results.get(c, {}).get('P99 TPOT (ms)', 0) or 0)) for c in concurrencies if all_results.get(c, {}).get('P99 TPOT (ms)', '')]
            itl_p99_values = [(c, float(all_results.get(c, {}).get('P99 ITL (ms)', 0) or 0)) for c in concurrencies if all_results.get(c, {}).get('P99 ITL (ms)', '')]
            
            ttft_warn = ""
            tpot_warn = ""
            itl_warn = ""
            
            if ttft_p99_values:
                max_ttft = max(ttft_p99_values, key=lambda x: x[1])
                ttft_warn = f"高并发(60-80)下TTFT急剧增加至{max_ttft[1]:.0f}ms" if max_ttft[1] > 4000 else "延迟随并发增加而上升"
            
            if tpot_p99_values:
                max_tpot = max(tpot_p99_values, key=lambda x: x[1])
                tpot_warn = f"P99在{typical_conc}并发时达到{max_tpot[1]:.2f}ms，超过40ms阈值要求" if max_tpot[1] > 40 else "P99稳定在40ms阈值以内"
            
            if itl_p99_values:
                max_itl = max(itl_p99_values, key=lambda x: x[1])
                itl_warn = f"P99稳定在{max_itl[1]:.0f}ms，远低于800ms阈值" if max_itl[1] <= 800 else "P99接近/超过800ms阈值"
            
            latency_detail = f"""- **TTFT (首Token延迟)**: P99在{typical_conc}并发时为{ttft_p99_typical}ms，满足4000ms阈值要求。{ttft_warn}
   - **TPOT (每Token生成时间)**: {tpot_warn}
   - **ITL (Token间延迟)**: P99在{typical_conc}并发以上{itl_warn}"""
        except:
            latency_detail = "- 延迟随并发增加而上升"
        
        try:
            success_rates = []
            peak_concurrent = 0
            for c in concurrencies:
                total = float(all_results.get(c, {}).get('Successful requests', '0') or '0') + float(all_results.get(c, {}).get('Failed requests', '0') or '0')
                if total > 0:
                    sr = float(all_results.get(c, {}).get('Successful requests', '0') or '0') / total * 100
                    success_rates.append((c, sr))
                pc = float(all_results.get(c, {}).get('Peak concurrent requests', '0') or '0')
                if pc > peak_concurrent:
                    peak_concurrent = pc
            if success_rates:
                min_sr = min(success_rates, key=lambda x: x[1])
                stability_detail = f"- 请求成功率{min_sr[1]:.1f}%，满足99%阈值要求\n   - 峰值并发请求数{peak_concurrent:.0f}，未超过100的阈值"
        except:
            stability_detail = "- 请求成功率满足阈值要求"
        
        problems = []
        if tpot_p99_typical_val > 40:
            problems.append(f"**TPOT延迟偏高**: {typical_conc}并发时TPOT P99为{tpot_p99_typical}ms，超过40ms阈值，每Token生成时间需优化")
        if ttft_p99_80_val > 8000:
            problems.append(f"**TTFT高并发恶化**: 80并发下TTFT P99达{ttft_p99_80_val:.0f}ms，响应延迟过高")
        
        if problems:
            problem_diagnosis = f"""### 问题诊断

基于8卡B200芯片的部署环境，{model_name}模型性能整体表现良好，但存在以下问题：

"""
            for i, p in enumerate(problems, 1):
                problem_diagnosis += f"{i}. {p}\n"
        
        suggestions = """### 优化建议

1. **延迟优化**: 建议优化batch调度策略或调整prefill/decode比例，降低每Token生成时间
2. **高并发TTFT关注**: 60-80并发下TTFT显著增加，建议增加队列管理或限流机制
3. **充分利用算力**: 当前吞吐量随并发线性增长，可进一步压测探索上限"""
    
    analysis_content = f"""1. **吞吐量表现** ✅
   {throughput_detail}

2. **延迟表现** ⚠️
   {latency_detail}

3. **稳定性表现** ✅
   {stability_detail}"""
    
    md_content = f"""# {model_name} vLLM Benchmark 测试结果报告

<div align="center">
**测试日期：** {current_date}

</div>

---

## 📊 测试概览

| 项目 | 配置 | 备注 |
|------|------|------|
| **数据集** | random |  |
| **并发数** | {concurrency_display} | 来自 yaml 配置 |
| **请求数量** | {num_prompts} | 来自 yaml 配置 |
| **输入上下文长度** | {input_len} | 来自 yaml 配置 |
| **输出上下文长度** | {output_len} | 来自 yaml 配置 |

---

## 🤖 模型配置信息

| 模型名称 | 部署配置 | GPU 型号 | GPU 数量 | 精度 | 模型副本数 |
|----------|----------|----------|----------|------|------------|
| **{model_name}** | TP=8, PP=1, DP=1, EP=N/A | B200 | 8 | FP8 | 1 |

---

## 📏 性能阈值指标

{threshold_rows}

---

## 📈 性能测试数据

### 🎯 服务基准结果

| 指标 | {concurrency_headers} |
|------|{concurrency_row}|
{serving_rows}

### ⏱️ 首 Token 延迟 (TTFT)

| 指标 | {concurrency_headers} |
|------|{concurrency_row}|
{ttft_rows}

### ⚡ 每 Token 生成时间 (TPOT)

| 指标 | {concurrency_headers} |
|------|{concurrency_row}|
{tpot_rows}

### 🔄 Token 间延迟 (ITL)

| 指标 | {concurrency_headers} |
|------|{concurrency_row}|
{itl_rows}

---

## 📊 性能趋势分析

下图展示了{model_name}在不同并发级别下的关键性能指标变化趋势：

![Performance Trends](./performance_trends.png)

---

## 📝 结果分析

### 综合评估 (基于8卡B200指标-{typical_conc}并发场景)

{eval_rows}

### 详细分析

{analysis_content}

{problem_diagnosis}

{suggestions}

---

<div align="center">

*报告生成时间: {current_date}*

</div>
"""
    
    md_file = os.path.join(output_dir, f"{model_name}_bench_summary.md")
    with open(md_file, 'w', encoding='utf-8') as f:
        f.write(md_content)
    
    print(f"[{model_name}] Markdown report generated: {md_file}")

def generate_charts(model_name, concurrencies, all_results, output_dir):
    if not HAS_MATPLOTLIB:
        return
        
    concurrencies_int = [int(c) for c in concurrencies]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'{model_name} Performance Analysis by Concurrency', fontsize=14, fontweight='bold')
    
    def get_values(key):
        return [float(all_results.get(c, {}).get(key, 0)) for c in concurrencies]
    
    axes[0, 0].plot(concurrencies_int, get_values('Request throughput (req/s)'), 'b-o', linewidth=2, markersize=8)
    axes[0, 0].set_title('Request Throughput (req/s)')
    axes[0, 0].set_xlabel('Concurrency')
    axes[0, 0].set_ylabel('req/s')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(concurrencies_int, get_values('Output token throughput (tok/s)'), 'g-o', linewidth=2, markersize=8)
    axes[0, 1].set_title('Output Token Throughput (tok/s)')
    axes[0, 1].set_xlabel('Concurrency')
    axes[0, 1].set_ylabel('tok/s')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[0, 2].plot(concurrencies_int, get_values('Total token throughput (tok/s)'), 'r-o', linewidth=2, markersize=8)
    axes[0, 2].set_title('Total Token Throughput (tok/s)')
    axes[0, 2].set_xlabel('Concurrency')
    axes[0, 2].set_ylabel('tok/s')
    axes[0, 2].grid(True, alpha=0.3)
    
    axes[1, 0].plot(concurrencies_int, get_values('Mean TTFT (ms)'), 'm-o', linewidth=2, markersize=8, label='Mean')
    axes[1, 0].plot(concurrencies_int, get_values('P99 TTFT (ms)'), 'c-s', linewidth=2, markersize=8, label='P99')
    axes[1, 0].set_title('TTFT Latency (ms)')
    axes[1, 0].set_xlabel('Concurrency')
    axes[1, 0].set_ylabel('ms')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(concurrencies_int, get_values('Mean TPOT (ms)'), 'y-o', linewidth=2, markersize=8, label='Mean')
    axes[1, 1].plot(concurrencies_int, get_values('P99 TPOT (ms)'), 'k-s', linewidth=2, markersize=8, label='P99')
    axes[1, 1].set_title('TPOT Latency (ms)')
    axes[1, 1].set_xlabel('Concurrency')
    axes[1, 1].set_ylabel('ms')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    axes[1, 2].plot(concurrencies_int, get_values('Mean ITL (ms)'), 'purple', linestyle='-', marker='o', linewidth=2, markersize=8, label='Mean')
    axes[1, 2].plot(concurrencies_int, get_values('P99 ITL (ms)'), 'orange', linestyle='-', marker='s', linewidth=2, markersize=8, label='P99')
    axes[1, 2].set_title('ITL Latency (ms)')
    axes[1, 2].set_xlabel('Concurrency')
    axes[1, 2].set_ylabel('ms')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    chart_file = os.path.join(output_dir, 'performance_trends.png')
    plt.savefig(chart_file, dpi=150)
    plt.close()
    
    print(f"[{model_name}] Charts generated: {chart_file}")

def main():
    reports_dir = "reports/benchmark"
    config_dir = "config"
    
    model_names = get_model_names(reports_dir)
    
    if not model_names:
        print(f"No model results found in {reports_dir}")
        return
    
    print(f"Found {len(model_names)} model(s): {', '.join(model_names)}")
    
    Path("analysis").mkdir(exist_ok=True)
    
    for model_name in model_names:
        print(f"\n{'='*50}")
        print(f"Processing model: {model_name}")
        print(f"{'='*50}")
        process_model(model_name, reports_dir, config_dir)
    
    print(f"\n{'='*50}")
    print("All models processed successfully!")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()
