import os
import sys
import yaml
import subprocess
import requests
import time
import re
import glob
from datetime import datetime
from itertools import product
from pathlib import Path

API_KEY = os.environ.get("API_KEY", "abc123")

TEST_SUITES = ["test_01", "test_03", "test_05", "test_06", "test_07"]

RUN_ID = "01"

try:
    from gpu_monitor import GPUMonitor, generate_gpu_charts

    HAS_GPU_MONITOR = True
except ImportError:
    HAS_GPU_MONITOR = False
    print("Warning: GPU monitor module not available")


def get_model_info_from_api(base_url, api_key):
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        response = requests.get(f"{base_url}/v1/models", headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data.get("data") and len(data["data"]) > 0:
                model_info = data["data"][0]
                model_name = model_info.get("id")
                owned_by = model_info.get("owned_by")
                model_path = model_info.get("root")
                if owned_by == "vllm" and model_path:
                    return model_name, model_path
                else:
                    return model_name, None
    except Exception as e:
        print(f"Failed to get model info from API: {e}")
    return None, None


def run_benchmark(chip_name, base_config, model_config, test_suites, run_id):
    base_url = base_config.get("base_url", "http://127.0.0.1:8080")

    model_name_yaml = model_config.get("name")
    served_model_name = model_config.get("served-model-name")
    model_path_yaml = model_config.get("model_path")

    model_name, model_path = get_model_info_from_api(base_url, API_KEY)

    if not model_name:
        model_name = served_model_name
    if not model_path:
        model_path = model_path_yaml

    print(f"Model Name: {model_name}")
    print(f"Model Path: {model_path}")
    print(f"Running test suites: {', '.join(test_suites)}")

    temperature = base_config.get("temperature", 0.7)
    seed = base_config.get("seed", 123)
    ready_timeout = base_config.get("ready-check-timeout-sec", 30)

    M = model_name_yaml
    output_base = f"reports/benchmark/{chip_name}/{M}"

    params_config = base_config.get("params", {})

    for test_suite in test_suites:
        test_params = params_config.get(test_suite, {})
        max_concurrency = test_params.get("max-concurrency", [10])
        num_prompts = test_params.get("num-prompts", [300])
        random_input_output_len = test_params.get(
            "random-input-output-len", [[20000, 100]]
        )

        run_id_dir = os.path.join(output_base, test_suite, run_id)
        if os.path.exists(run_id_dir):
            print(
                f"Error: Run ID '{run_id}' already exists for test suite '{test_suite}' at path: {run_id_dir}"
            )
            print(f"Please either:")
            print(f"  1. Use a different RUN_ID (--run-id)")
            print(f"  2. Delete the existing directory: {run_id_dir}")
            continue

        print(f"\n=== Running test suite: {test_suite} ===")

        gpu_monitor = GPUMonitor(interval=10) if HAS_GPU_MONITOR else None

        for nc, np, io_len in product(
            max_concurrency, num_prompts, random_input_output_len
        ):
            ni = io_len[0]
            no = io_len[1]
            param_dir = f"{test_suite}/{run_id}/{nc}-{np}-i{ni}-o{no}"
            output_dir = os.path.join(output_base, param_dir)
            Path(output_dir).mkdir(parents=True, exist_ok=True)

            log_file = os.path.join(
                output_dir, f"bench-{test_suite}-{nc}-{np}-i{ni}-o{no}.log"
            )

            if gpu_monitor:
                monitor_param_dir = f"{test_suite}/{run_id}/{nc}-{np}-i{ni}-o{no}"
                gpu_monitor.start_monitoring(
                    "monitor", chip_name, model_name_yaml, monitor_param_dir
                )

            cmd = [
                "vllm",
                "bench",
                "serve",
                "--backend",
                "openai-chat",
                "--endpoint",
                "/v1/chat/completions",
                "--dataset-name",
                test_params.get("dataset-name", "random"),
                "--random-input-len",
                str(ni),
                "--random-output-len",
                str(no),
                "--model",
                str(model_path),
                "--trust-remote-code",
                "--base-url",
                base_url,
                "--num-prompts",
                str(np),
                "--max-concurrency",
                str(nc),
                "--temperature",
                str(temperature),
                "--seed",
                str(seed),
                "--metric_percentiles",
                "95,99",
                "--served-model-name",
                str(model_name),
                "--ready-check-timeout-sec",
                str(ready_timeout),
            ]

            print(f"Running: {' '.join(cmd)}")
            print(f"Log file: {log_file}")

            log_f = open(log_file, "w")
            process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
            )

            for line in process.stdout:
                print(line, end="")
                log_f.write(line)

            process.wait()
            log_f.close()

            if gpu_monitor:
                gpu_log = gpu_monitor.stop_monitoring()
                if gpu_log:
                    gpu_log_dir = os.path.dirname(gpu_log)
                    generate_gpu_charts(gpu_log, gpu_log_dir)

            print(f"Completed: {log_file}")
            time.sleep(30)


def get_available_run_ids(chip_name, model_name, test_suite):
    base_path = f"reports/benchmark/{chip_name}/{model_name}/{test_suite}"
    if not os.path.exists(base_path):
        return []

    run_ids = []
    for item in os.listdir(base_path):
        item_path = os.path.join(base_path, item)
        if os.path.isdir(item_path):
            run_ids.append(item)

    return sorted(run_ids)


def run_single_report(chip_name, model_name, test_suite, run_id, concurrency=None):
    print(f"\n{'#' * 60}")
    print(f"Generating single run-id report: {run_id}")
    print(f"{'#' * 60}")

    chip_key_map = {
        "Hygon_BW1000": "hygon_bw1000",
        "Kunlun_P800": "kunlun_p800",
        "NVIDIA_H100": "nvidia_h100",
    }
    chip_key = chip_key_map.get(chip_name, chip_name.lower())

    cmd = [
        sys.executable,
        "parse_single_chip_model.py",
        "--chip",
        chip_key,
        "--model",
        model_name,
        "--test-suite",
        test_suite,
        "--run-id",
        run_id,
    ]

    if concurrency:
        cmd.extend(["--concurrency", concurrency])

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False, text=True)
    return result.returncode == 0


def run_runid_comparison(chip_name, model_name, test_suite, run_ids, concurrency=None):
    print(f"\n{'#' * 60}")
    print(f"Generating run-id comparison report for: {', '.join(run_ids)}")
    print(f"{'#' * 60}")

    chip_key_map = {
        "Hygon_BW1000": "hygon_bw1000",
        "Kunlun_P800": "kunlun_p800",
        "NVIDIA_H100": "nvidia_h100",
    }
    chip_key = chip_key_map.get(chip_name, chip_name.lower())

    run_ids_str = ",".join(run_ids)

    cmd = [
        sys.executable,
        "parse_single_chip_model_runid.py",
        "--chip",
        chip_key,
        "--model",
        model_name,
        "--test-suite",
        test_suite,
        "--run-id",
        run_ids_str,
    ]

    if concurrency:
        cmd.extend(["--concurrency", concurrency])

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False, text=True)
    return result.returncode == 0


def generate_reports(chip_name, model_name, test_suites, concurrency=None):
    chip_key_map = {
        "Hygon_BW1000": "hygon_bw1000",
        "Kunlun_P800": "kunlun_p800",
        "NVIDIA_H100": "nvidia_h100",
    }
    chip_key = chip_key_map.get(chip_name, chip_name.lower())

    for test_suite in test_suites:
        print(f"\n{'=' * 60}")
        print(f"Checking test suite: {test_suite}")
        print(f"{'=' * 60}")

        run_ids = get_available_run_ids(chip_key, model_name, test_suite)

        if not run_ids:
            print(f"No run-id data found for test suite: {test_suite}")
            continue

        print(f"Found {len(run_ids)} run-id(s): {', '.join(run_ids)}")

        if len(run_ids) == 1:
            run_id = run_ids[0]
            print(f"Only one run-id found, generating single report...")
            run_single_report(chip_key, model_name, test_suite, run_id, concurrency)
        else:
            print(f"Multiple run-ids found, generating comparison report...")
            run_runid_comparison(chip_key, model_name, test_suite, run_ids, concurrency)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Run vLLM benchmark and generate reports"
    )
    parser.add_argument(
        "--chip",
        type=str,
        required=True,
        help="Chip name to test (e.g., hygon_bw1000, kunlun_p800, nvidia_h100)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name to test (e.g., MiniMax-M2.5, Qwen3.5-122B-A10B). If not specified, uses the first model in config.",
    )
    parser.add_argument(
        "--test-suite",
        type=str,
        default=None,
        help=f"Test suite to run (default: all). Available: {', '.join(TEST_SUITES)}",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=RUN_ID,
        help=f"Run ID to identify this test run (default: {RUN_ID})",
    )
    parser.add_argument(
        "--skip-benchmark",
        action="store_true",
        help="Skip running benchmark, only generate reports based on existing data",
    )
    parser.add_argument(
        "--only-report",
        action="store_true",
        help="Only generate reports without running benchmark",
    )
    parser.add_argument(
        "-c",
        "--concurrency",
        type=str,
        default=None,
        help="Specific concurrency levels to generate report for, comma-separated (e.g., 1,2,4,8,10)",
    )
    args = parser.parse_args()

    yaml_path = os.path.join(
        os.path.dirname(__file__), "config", "models_scenarios.yaml"
    )

    with open(yaml_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    base_config = config.get("base_config", {})
    params_config = base_config.get("params", {})
    models = config.get("models", {})

    chip_name = args.chip.lower()
    if chip_name not in models:
        print(
            f"Error: Chip '{chip_name}' not found in config. Available chips: {', '.join(models.keys())}"
        )
        return

    available_models = models[chip_name]

    if args.model:
        model_name_lower = args.model.lower()
        selected_model = None
        for m in available_models:
            if m.get("name", "").lower() == model_name_lower:
                selected_model = m
                break
        if not selected_model:
            print(
                f"Error: Model '{args.model}' not found for chip '{chip_name}'. Available models:"
            )
            for m in available_models:
                print(f"  - {m.get('name')} (served: {m.get('served-model-name')})")
            return
        model_configs = [selected_model]
    else:
        model_configs = [available_models[0]]
        print(f"No model specified, using default: {model_configs[0].get('name')}")

    test_suites_to_run = []
    if args.test_suite:
        test_suites_to_run = [s.strip() for s in args.test_suite.split(",")]
    else:
        test_suites_to_run = TEST_SUITES

    invalid_suites = [s for s in test_suites_to_run if s not in params_config]
    if invalid_suites:
        print(
            f"Error: Test suite(s) {invalid_suites} not found in config. Available: {', '.join(params_config.keys())}"
        )
        return

    run_id = args.run_id

    for model_config in model_configs:
        model_name = model_config.get("name")

        print(f"\n{'#' * 60}")
        print(f"Processing chip: {chip_name}, model: {model_name}")
        print(f"{'#' * 60}")

        if not args.only_report:
            if not args.skip_benchmark:
                run_benchmark(
                    chip_name, base_config, model_config, test_suites_to_run, run_id
                )
            else:
                print("Skipping benchmark execution (--skip-benchmark specified)")
                print("Loading config for report generation...")

        print(f"\nFinished benchmark for chip: {chip_name}, model: {model_name}")

        print(f"\n{'#' * 60}")
        print("Generating reports...")
        print(f"{'#' * 60}")

        generate_reports(chip_name, model_name, test_suites_to_run, args.concurrency)

        print(f"\n{'#' * 60}")
        print(f"All reports generated for chip: {chip_name}, model: {model_name}")
        print(f"{'#' * 60}")


if __name__ == "__main__":
    main()
