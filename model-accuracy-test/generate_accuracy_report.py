import argparse
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate accuracy test comparison report"
    )
    parser.add_argument(
        "--chip",
        type=str,
        required=True,
        help="Chip names separated by comma, e.g., nvidia_h100,hygon_bw1000",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Task name(s) separated by comma, e.g., IFBench,lm-eval:gsm_plus. If not specified, compare all tasks.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name(s) separated by comma, e.g., GLM-5,MiniMax-M2.5. If not specified, compare all models.",
    )
    return parser.parse_args()


def find_log_files(
    base_dir: Path,
    chips: List[str],
    task: Optional[List[str]] = None,
    models: Optional[List[str]] = None,
) -> Dict[str, Dict[str, Dict[str, List[Path]]]]:
    result: Dict[str, Dict[str, Dict[str, List[Path]]]] = {}
    task_dirs = []
    not_found_tasks = []

    if task:
        for t in task:
            task_path = None
            task_display_name = t

            if t == "IFBench":
                task_path = base_dir / "IFBench"
                if task_path.exists():
                    task_dirs.append(("IFBench", task_path))
                else:
                    not_found_tasks.append(t)
            elif t.startswith("lm-eval:"):
                task_name = t.replace("lm-eval:", "")
                task_path = base_dir / "lm-eval" / task_name
                if task_path.exists():
                    task_dirs.append((t, task_path))
                else:
                    not_found_tasks.append(t)
            else:
                if (base_dir / t).exists():
                    task_dirs.append((t, base_dir / t))
                else:
                    task_path = base_dir / "lm-eval" / t
                    if task_path.exists():
                        task_dirs.append((f"lm-eval:{t}", task_path))
                    else:
                        not_found_tasks.append(t)
    else:
        for item in base_dir.iterdir():
            if item.is_dir() and item.name not in ["accuracy_report", "__pycache__"]:
                if item.name == "lm-eval":
                    for task_subdir in item.iterdir():
                        if task_subdir.is_dir():
                            task_dirs.append(
                                (f"lm-eval:{task_subdir.name}", task_subdir)
                            )
                else:
                    task_dirs.append((item.name, item))

    if not_found_tasks:
        print(f"Warning: Tasks not found: {', '.join(not_found_tasks)}")

    for task_name, task_path in task_dirs:
        for model_dir in task_path.iterdir():
            if not model_dir.is_dir():
                continue
            model_name = model_dir.name

            if models and model_name not in models:
                continue

            for chip in chips:
                chip_dir = None
                for subdir in model_dir.iterdir():
                    if subdir.is_dir() and subdir.name == chip:
                        chip_dir = subdir
                        break

                if chip_dir:
                    for log_file in chip_dir.glob("*.log"):
                        quant = extract_quantization_level(log_file.name)
                        if "append-think" in quant or "append" in quant.lower():
                            continue
                        if task_name not in result:
                            result[task_name] = {}
                        if model_name not in result[task_name]:
                            result[task_name][model_name] = {}
                        if chip not in result[task_name][model_name]:
                            result[task_name][model_name][chip] = []
                        result[task_name][model_name][chip].append(log_file)

    return result


def extract_quantization_level(filename: str) -> str:
    basename = os.path.basename(filename)
    match = re.match(r"^([^_]+)_", basename)
    if match:
        return match.group(1)
    return "unknown"


def parse_ifbench_log(log_path: Path) -> Tuple[Optional[float], Optional[float]]:
    strict_acc = None
    loose_acc = None

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        if "eval_results_strict" in line and i + 1 < len(lines):
            next_line = lines[i + 1]
            match = re.search(r"Accuracy:\s*([\d.]+)", next_line)
            if match:
                strict_acc = float(match.group(1))
        elif "eval_results_loose" in line and i + 1 < len(lines):
            next_line = lines[i + 1]
            match = re.search(r"Accuracy:\s*([\d.]+)", next_line)
            if match:
                loose_acc = float(match.group(1))

    return strict_acc, loose_acc


def parse_gsm_plus_log(log_path: Path) -> Tuple[Optional[float], Optional[float]]:
    flexible_extract = None
    strict_match = None

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    lines = content.split("\n")
    for line in lines:
        if "flexible-extract" in line:
            parts = [p.strip() for p in line.split("|")]
            for i, p in enumerate(parts):
                if "flexible-extract" in p and i + 4 < len(parts):
                    try:
                        val = float(parts[i + 4].strip())
                        if 0 < val <= 1:
                            flexible_extract = val
                    except:
                        pass
        if "strict-match" in line:
            parts = [p.strip() for p in line.split("|")]
            for i, p in enumerate(parts):
                if "strict-match" in p and i + 4 < len(parts):
                    try:
                        val = float(parts[i + 4].strip())
                        if 0 < val <= 1:
                            strict_match = val
                    except:
                        pass

    return flexible_extract, strict_match


def parse_mmlu_pro_log(log_path: Path) -> Tuple[Optional[float], Dict[str, float]]:
    overall_value = None
    detailed_values: Dict[str, float] = {}

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    lines = content.split("\n")

    for line in lines:
        if re.match(r"\|mmlu_pro\s*\|", line):
            parts = [p.strip() for p in line.split("|")]
            for i, p in enumerate(parts):
                try:
                    if p and p != "↑" and p != "mmlu_pro":
                        val = float(p.replace("±", "").strip())
                        if 0 <= val <= 1:
                            overall_value = val
                            break
                except:
                    continue

        match = re.match(r"\|\s*-\s+(\w+)\s*\|", line)
        if match:
            task_item = match.group(1)
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 8:
                try:
                    val = float(parts[7].replace("↑", "").strip())
                    if 0 <= val <= 1:
                        detailed_values[task_item] = val
                except:
                    continue

    return overall_value, detailed_values


def parse_ruler_log(log_path: Path) -> Tuple[Optional[float], Dict[str, float]]:
    overall_avg = None
    detailed_values: Dict[str, float] = {}

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    lines = content.split("\n")
    values = []

    for line in lines:
        if re.match(r"\|\s*-\s*(\w+)", line):
            match = re.match(r"\|\s*-\s*(\S+)\s*\|", line)
            if match:
                task_item = match.group(1).strip()
                parts = [p.strip() for p in line.split("|")]
                if len(parts) >= 8:
                    try:
                        val = float(parts[7].replace("↑", "").strip())
                        if 0 <= val <= 1:
                            detailed_values[task_item] = val
                            values.append(val)
                    except:
                        continue

    if values:
        overall_avg = sum(values) / len(values)

    return overall_avg, detailed_values


def parse_log_file(log_path: Path, task_name: str) -> Dict[str, Optional[float]]:
    result: Dict[str, Optional[float]] = {
        "strict": None,
        "loose": None,
        "flexible": None,
        "strict_match": None,
        "value": None,
        "average": None,
    }

    if task_name == "IFBench":
        strict_acc, loose_acc = parse_ifbench_log(log_path)
        if strict_acc is not None:
            result["strict"] = strict_acc
        if loose_acc is not None:
            result["loose"] = loose_acc
    elif task_name == "lm-eval:gsm_plus":
        flexible, strict_match = parse_gsm_plus_log(log_path)
        if flexible is not None:
            result["flexible"] = flexible
        if strict_match is not None:
            result["strict_match"] = strict_match
    elif task_name == "lm-eval:mmlu_pro":
        val, _ = parse_mmlu_pro_log(log_path)
        if val is not None:
            result["value"] = val
    elif task_name == "lm-eval:ruler":
        avg, _ = parse_ruler_log(log_path)
        if avg is not None:
            result["average"] = avg

    return result


def get_task_metrics(task_name: str) -> List[Tuple[str, str]]:
    if task_name == "IFBench":
        return [("Strict", "strict"), ("Loose", "loose")]
    elif task_name == "lm-eval:gsm_plus":
        return [("Flexible", "flexible"), ("Strict", "strict_match")]
    elif task_name == "lm-eval:mmlu_pro":
        return [("Value", "value")]
    elif task_name == "lm-eval:ruler":
        return [("Average", "average")]
    return []


def generate_detailed_table(
    log_files_data: Dict[Tuple[str, str], Path],
    task_name: str,
    chips: List[str],
    base_chip: str,
) -> str:
    lines = []

    all_detailed: Dict[str, Dict[Tuple[str, str], float]] = {}

    for (chip, quant), log_path in log_files_data.items():
        if task_name == "lm-eval:mmlu_pro":
            _, detailed = parse_mmlu_pro_log(log_path)
        elif task_name == "lm-eval:ruler":
            _, detailed = parse_ruler_log(log_path)
        else:
            continue

        for item, val in detailed.items():
            if item not in all_detailed:
                all_detailed[item] = {}
            all_detailed[item][(chip, quant)] = val

    if not all_detailed:
        return ""

    sorted_keys = sorted(log_files_data.keys())
    baseline_keys = [k for k in sorted_keys if k[0] == base_chip]
    compare_keys = [k for k in sorted_keys if k[0] != base_chip]

    header_parts = ["Item"]
    for key in baseline_keys:
        header_parts.append(f"{key[0]}({key[1]})")
    for key in compare_keys:
        header_parts.append(f"{key[0]}({key[1]})")
        header_parts.append("差值")
        header_parts.append("百分比")

    lines.append("| " + " | ".join(header_parts) + " |")

    separator_parts = ["------"]
    for _ in baseline_keys:
        separator_parts.append("------")
    for _ in compare_keys:
        separator_parts.append("------")
        separator_parts.append("------")
        separator_parts.append("------")

    lines.append("|" + "|".join(separator_parts) + "|")

    for item in sorted(all_detailed.keys()):
        row_parts = [item]

        baseline_val = None
        for key in baseline_keys:
            val = all_detailed[item].get(key)
            if val is not None:
                if baseline_val is None:
                    baseline_val = val
            val_str = f"{val:.4f}" if val is not None else "N/A"
            row_parts.append(val_str)

        for key in compare_keys:
            other_val = all_detailed[item].get(key)
            val_str = f"{other_val:.4f}" if other_val is not None else "N/A"
            row_parts.append(val_str)

            if baseline_val is not None and other_val is not None:
                diff = other_val - baseline_val
                diff_rate = (diff / baseline_val) * 100 if baseline_val != 0 else 0
                diff_str = f"{diff:.4f}"
                sign = "+" if diff_rate >= 0 else "-"
                diff_rate_str = f"{sign} {abs(diff_rate):.2f}%"
            else:
                diff_str = "N/A"
                diff_rate_str = "N/A"

            row_parts.append(diff_str)
            row_parts.append(diff_rate_str)

        lines.append("| " + " | ".join(row_parts) + " |")

    return "\n".join(lines)


def generate_report(
    log_files: Dict[str, Dict[str, Dict[str, List[Path]]]], chips: List[str]
) -> str:
    report_lines = []
    chips_str = "&".join(chips)
    report_lines.append(f"# Accuracy Test Comparison Report - Chips: {chips_str}")
    report_lines.append("")

    base_chip = chips[0]
    other_chips = chips[1:] if len(chips) > 1 else []

    all_models = set()
    for task_name, models in log_files.items():
        all_models.update(models.keys())

    detailed_tables: List[Tuple[str, str, str]] = []

    for model_name in sorted(all_models):
        report_lines.append(f"## {model_name}模型")
        report_lines.append("")

        model_tasks_data = {}
        for task_name, models in log_files.items():
            if model_name in models:
                model_tasks_data[task_name] = models[model_name]

        all_chip_quant_keys = set()
        for task_name, model_data in model_tasks_data.items():
            for chip in chips:
                if chip in model_data:
                    for log_file in model_data[chip]:
                        quant = extract_quantization_level(log_file.name)
                        all_chip_quant_keys.add((chip, quant))

        sorted_chip_quant_keys = sorted(all_chip_quant_keys)
        baseline_keys = [k for k in sorted_chip_quant_keys if k[0] == base_chip]
        compare_keys = [k for k in sorted_chip_quant_keys if k[0] in other_chips]

        header_parts = ["Task"]
        for key in baseline_keys:
            header_parts.append(f"{key[0]}({key[1]})")
        for key in compare_keys:
            header_parts.append(f"{key[0]}({key[1]})")
            header_parts.append("差值")
            header_parts.append("百分比")

        report_lines.append("| " + " | ".join(header_parts) + " |")

        separator_parts = ["------"]
        for _ in baseline_keys:
            separator_parts.append("------")
        for _ in compare_keys:
            separator_parts.append("------")
            separator_parts.append("------")
            separator_parts.append("------")

        report_lines.append("|" + "|".join(separator_parts) + "|")

        all_task_rows = []

        for task_name in sorted(model_tasks_data.keys()):
            model_data = model_tasks_data[task_name]

            chip_quant_results = {}
            chip_quant_log_files: Dict[Tuple[str, str], Path] = {}

            for chip in chips:
                if chip in model_data:
                    for log_file in model_data[chip]:
                        quant = extract_quantization_level(log_file.name)
                        key = (chip, quant)
                        if key not in chip_quant_results:
                            chip_quant_results[key] = {}
                        chip_quant_log_files[key] = log_file
                        parsed = parse_log_file(log_file, task_name)
                        for metric_key, val in parsed.items():
                            if (
                                val is not None
                                and metric_key not in chip_quant_results[key]
                            ):
                                chip_quant_results[key][metric_key] = val

            metrics = get_task_metrics(task_name)
            if not metrics:
                continue

            for metric_name, metric_key in metrics:
                task_label = (
                    f"{task_name} ({metric_name})" if len(metrics) > 1 else task_name
                )
                row_parts = [task_label]

                baseline_val = None
                for key in baseline_keys:
                    val = chip_quant_results.get(key, {}).get(metric_key)
                    if val is not None and baseline_val is None:
                        baseline_val = val
                    val_str = f"{val:.4f}" if val is not None else "N/A"
                    row_parts.append(val_str)

                for compare_key in compare_keys:
                    other_val = chip_quant_results.get(compare_key, {}).get(metric_key)
                    val_str = f"{other_val:.4f}" if other_val is not None else "N/A"
                    row_parts.append(val_str)

                    if baseline_val is not None and other_val is not None:
                        diff = other_val - baseline_val
                        diff_rate = (
                            (diff / baseline_val) * 100 if baseline_val != 0 else 0
                        )
                        diff_str = f"{diff:.4f}"
                        sign = "+" if diff_rate >= 0 else "-"
                        diff_rate_str = f"{sign} {abs(diff_rate):.2f}%"
                    else:
                        diff_str = "N/A"
                        diff_rate_str = "N/A"

                    row_parts.append(diff_str)
                    row_parts.append(diff_rate_str)

                all_task_rows.append((task_name, row_parts))

            if task_name in ["lm-eval:mmlu_pro", "lm-eval:ruler"]:
                detailed_table = generate_detailed_table(
                    chip_quant_log_files, task_name, chips, base_chip
                )
                if detailed_table:
                    if task_name == "lm-eval:mmlu_pro":
                        title = "mmlu_pro任务子数据集详细比对结果"
                    else:
                        title = "ruler任务子数据集详细比对结果"
                    detailed_tables.append((model_name, title, detailed_table))

        for _, row_parts in all_task_rows:
            report_lines.append("| " + " | ".join(row_parts) + " |")

        report_lines.append("")

    if detailed_tables:
        report_lines.append("# 任务子数据集详细比对结果")
        report_lines.append("")

        for model_name, title, table in detailed_tables:
            report_lines.append(f"## {model_name}模型 - {title}")
            report_lines.append("")
            report_lines.append(table)
            report_lines.append("")

    return "\n".join(report_lines)


def main():
    args = parse_args()
    base_dir = Path(__file__).parent.resolve()

    chips = [c.strip() for c in args.chip.split(",")]

    models = None
    if args.model:
        models = [m.strip() for m in args.model.split(",")]

    tasks = None
    if args.task:
        tasks = [t.strip() for t in args.task.split(",")]

    log_files = find_log_files(base_dir, chips, tasks, models)

    if not log_files:
        print(
            f"No log files found for chips: {chips}, models: {models}, tasks: {tasks}"
        )
        return

    report = generate_report(log_files, chips)

    chips_str = "&".join(chips)
    models_str = "_".join(models) if models else ""

    if tasks:
        task_names = []
        for t in tasks:
            if t == "IFBench":
                task_names.append("ifbench")
            elif t.startswith("lm-eval:"):
                task_names.append(t.replace("lm-eval:", "").replace("_", ""))
            else:
                task_names.append(t.replace("_", ""))
        task_str = "_".join(task_names)

        if models_str:
            filename = f"{chips_str}_{models_str}_{task_str}.md"
        else:
            filename = f"{chips_str}_{task_str}.md"
    else:
        if models_str:
            filename = f"{chips_str}_{models_str}_all_task.md"
        else:
            filename = f"{chips_str}_all_task.md"

    output_dir = base_dir / "accuracy_report"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / filename

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"Report saved to: {output_path}")
    print()
    print(report)


if __name__ == "__main__":
    main()
