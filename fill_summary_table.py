import re
import os
import glob
import argparse
import json


CHIP_MODEL_CONFIG = [
    {
        "chip_dir": "nvidia_h100",
        "model": "MiniMax-M2.5",
        "col_name": "NVIDIA_H100_MiniMax-M2.5",
    },
    {
        "chip_dir": "hygon_bw1000",
        "model": "MiniMax-M2.5-bf16",
        "col_name": "Hygon_BW1000_MiniMax-M2.5-bf16",
    },
    {
        "chip_dir": "hygon_bw1000",
        "model": "MiniMax-M2.5-W8A8",
        "col_name": "Hygon_BW1000_MiniMax-M2.5-W8A8",
    },
    {
        "chip_dir": "hygon_bw1000",
        "model": "GLM-5-W8A8",
        "col_name": "Hygon_BW1000_GLM-5-W8A8",
    },
    {
        "chip_dir": "kunlun_p800",
        "model": "MiniMax-M2.5-W8A8-INT8-Dynamic",
        "col_name": "Kunlun_P800_MiniMax-M2.5-W8A8-INT8-Dynamic",
    },
    {
        "chip_dir": "inspur_MetaX_C550",
        "model": "MiniMax-M2.5-W8A8",
        "col_name": "inspur_MetaX_C550_MiniMax-M2.5-W8A8",
    },
    {
        "chip_dir": "inspur_MetaX_C550",
        "model": "GLM-5-W8A8",
        "col_name": "inspur_MetaX_C550_GLM-5-W8A8",
    },
    {
        "chip_dir": "inspur_MetaX_C550",
        "model": "Kimi-K2.5_int4_2",
        "col_name": "inspur_MetaX_C550_Kimi-K2.5_int4_2",
    },
]

TEST_SUITES = ["test_01", "test_02"]

SUMMARY_METRICS = [
    ("请求吞吐量（Request throughput (req/s)）", "Request throughput (req/s)"),
    (
        "输入token吞吐量（Input token throughput (tok/s)）",
        "Input token throughput (tok/s)",
    ),
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
    ("端到端延迟 （P99 E2E Latency (ms)）", "P99 E2E Latency (ms)"),
]

E2E_CHIPS = {"inspur_MetaX_C550", "hygon_bw1000"}

CACHE_FILE = "analysis/summary/.summary_cache.json"


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


def find_concurrency_dir(chip_dir, model, test_suite, concurrency, run_id="01"):
    base = f"reports/benchmark/{chip_dir}/{model}/{test_suite}/{run_id}"
    if not os.path.isdir(base):
        return None
    for d in os.listdir(base):
        if d.startswith(f"{concurrency}-"):
            full_path = os.path.join(base, d)
            if os.path.isdir(full_path):
                return d
    return None


def find_log_file(chip_dir, model, test_suite, concurrency, run_id="01"):
    conc_dir = find_concurrency_dir(chip_dir, model, test_suite, concurrency, run_id)
    if not conc_dir:
        return None
    base = f"reports/benchmark/{chip_dir}/{model}/{test_suite}/{run_id}/{conc_dir}"
    pattern = os.path.join(base, "bench-*.log")
    log_files = glob.glob(pattern)
    if log_files:
        return log_files[0]
    return None


def get_metric_value(metrics, display_name, key_name, chip_dir):
    value = metrics.get(key_name, None)

    if display_name.startswith("输入token吞吐量"):
        if value is None:
            output_tput = metrics.get("Output token throughput (tok/s)")
            total_tput = metrics.get("Total token throughput (tok/s)")
            if output_tput and total_tput:
                try:
                    result = float(total_tput) - float(output_tput)
                    return f"{result:.2f}"
                except:
                    return "N/A"
            return "N/A"

    if display_name.startswith("端到端延迟"):
        if chip_dir not in E2E_CHIPS:
            return "N/A"
        if value is None:
            return "N/A"

    if value is None:
        return "N/A"

    return value


def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cache(cache):
    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def read_current_data():
    cache = load_cache()
    if not cache:
        return {}
    return cache.get("rows", {})


def merge_data(existing, new_data):
    for test_suite, conc_data in new_data.items():
        if test_suite not in existing:
            existing[test_suite] = {}
        existing[test_suite].update(conc_data)
    return existing


def generate_markdown(all_rows_data, col_names, nvidia_col):
    header_parts = ["测试场景", "性能指标"]
    separator_parts = ["---------", "--------------------------------------------"]
    for col in col_names:
        if col == nvidia_col:
            header_parts.append(f"{col}(基准)")
        else:
            header_parts.append(col)
            header_parts.append("相对基准")
        separator_parts.append("-----------")
        if col != nvidia_col:
            separator_parts.append("-----------")

    all_rows = []
    for test_suite in TEST_SUITES:
        if test_suite not in all_rows_data:
            continue
        conc_keys = sorted(
            all_rows_data[test_suite].keys(), key=lambda x: int(x.replace("并发", ""))
        )
        for conc_key in conc_keys:
            conc_data = all_rows_data[test_suite][conc_key]
            base_value = None
            nvidia_val = conc_data.get(nvidia_col, {}).get(SUMMARY_METRICS[0][0], "N/A")

            for display_name, key_name in SUMMARY_METRICS:
                row_parts = [f"{test_suite}({conc_key})", display_name]

                if display_name == SUMMARY_METRICS[0][0]:
                    base_value = None
                    nv = conc_data.get(nvidia_col, {}).get(display_name, "N/A")
                    if nv != "N/A":
                        try:
                            base_value = float(nv)
                        except:
                            base_value = None

                current_base = None
                nv = conc_data.get(nvidia_col, {}).get(display_name, "N/A")
                if nv != "N/A":
                    try:
                        current_base = float(nv)
                    except:
                        current_base = None

                for col in col_names:
                    val = conc_data.get(col, {}).get(display_name, "N/A")
                    row_parts.append(val)
                    if col != nvidia_col:
                        if (
                            current_base is not None
                            and current_base != 0
                            and val != "N/A"
                        ):
                            try:
                                curr = float(val)
                                pct = ((curr - current_base) / current_base) * 100
                                row_parts.append(f"{pct:+.1f}%")
                            except:
                                row_parts.append("-")
                        else:
                            row_parts.append("-")

                all_rows.append("| " + " | ".join(row_parts) + " |")

        if test_suite != TEST_SUITES[-1]:
            empty_cols = [""] * (2 + len(col_names) + len(col_names) - 1)
            all_rows.append("| " + " | ".join(empty_cols) + " |")

    md_content = f"""# 各芯片不同模型benchmark测试结果汇总

---


## 📈 各芯片平台和模型性能指标对比详情


| {" | ".join(header_parts)} |
|{" | ".join(separator_parts)}|
{chr(10).join(all_rows)}
"""

    output_path = "analysis/summary/chip_test_summary.md"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"Updated: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Fill summary table with benchmark data"
    )
    parser.add_argument(
        "--concurrency",
        type=str,
        default="1,1",
        help="Concurrency for each test suite (test_01,test_02), e.g. '1,1' or '64,10'",
    )
    args = parser.parse_args()

    conc_list = [s.strip() for s in args.concurrency.split(",")]
    if len(conc_list) == 1:
        conc_map = {ts: conc_list[0] for ts in TEST_SUITES}
    elif len(conc_list) == len(TEST_SUITES):
        conc_map = dict(zip(TEST_SUITES, conc_list))
    else:
        print(
            f"Error: --concurrency expects {len(TEST_SUITES)} values (one per test suite), got {len(conc_list)}"
        )
        return

    print(f"Concurrency mapping: {conc_map}")

    col_names = [cfg["col_name"] for cfg in CHIP_MODEL_CONFIG]

    nvidia_col = None
    for cfg in CHIP_MODEL_CONFIG:
        if "nvidia" in cfg["chip_dir"].lower() and "h100" in cfg["chip_dir"].lower():
            nvidia_col = cfg["col_name"]
            break
    if nvidia_col is None:
        nvidia_col = col_names[0]

    new_data = {}
    for test_suite in TEST_SUITES:
        concurrency = conc_map[test_suite]
        conc_key = f"并发{concurrency}"
        new_data.setdefault(test_suite, {})[conc_key] = {}

        for cfg in CHIP_MODEL_CONFIG:
            col_name = cfg["col_name"]
            log_file = find_log_file(
                cfg["chip_dir"], cfg["model"], test_suite, concurrency
            )
            if log_file and os.path.exists(log_file):
                metrics = parse_benchmark_log(log_file)
                row_data = {}
                for display_name, key_name in SUMMARY_METRICS:
                    row_data[display_name] = get_metric_value(
                        metrics, display_name, key_name, cfg["chip_dir"]
                    )
                new_data[test_suite][conc_key][col_name] = row_data
            else:
                new_data[test_suite][conc_key][col_name] = {
                    dn: "N/A" for dn, _ in SUMMARY_METRICS
                }

            status = "OK" if log_file and os.path.exists(log_file) else "MISSING"
            print(f"  {col_name} / {test_suite} / {conc_key}: {status}")

    all_rows_data = read_current_data()
    merge_data(all_rows_data, new_data)

    cache = {"rows": all_rows_data}
    save_cache(cache)

    generate_markdown(all_rows_data, col_names, nvidia_col)


if __name__ == "__main__":
    main()
