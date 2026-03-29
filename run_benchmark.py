import os
import yaml
import subprocess
import requests
import time
from datetime import datetime
from itertools import product
from pathlib import Path

API_KEY = os.environ.get("API_KEY", "abc123")

TEST_SUITES = ["test_01", "test_02", "test_03", "test_04", "test_05", "test_06"]

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
    
    model_name_yaml = model_config.get("served-model-name")
    model_path_yaml = model_config.get("model_path")
    
    model_name, model_path = get_model_info_from_api(base_url, API_KEY)
    
    if not model_name:
        model_name = model_name_yaml
    if not model_path:
        model_path = model_path_yaml
    
    print(f"Model Name: {model_name}")
    print(f"Model Path: {model_path}")
    print(f"Running test suites: {', '.join(test_suites)}")
    
    temperature = base_config.get("temperature", 0.7)
    seed = base_config.get("seed", 123)
    ready_timeout = base_config.get("ready-check-timeout-sec", 30)
    
    M = Path(model_path).name if model_path else "unknown"
    output_base = f"reports/{chip_name}/benchmark/{M}"
    
    params_config = base_config.get("params", {})
    
    for test_suite in test_suites:
        test_params = params_config.get(test_suite, {})
        max_concurrency = test_params.get("max-concurrency", [10])
        num_prompts = test_params.get("num-prompts", [300])
        random_input_len = test_params.get("random-input-len", [20000])
        random_output_len = test_params.get("random-output-len", [100])
        
        run_id_dir = os.path.join(output_base, test_suite, run_id)
        if os.path.exists(run_id_dir):
            print(f"Error: Run ID '{run_id}' already exists for test suite '{test_suite}' at path: {run_id_dir}")
            print(f"Please either:")
            print(f"  1. Use a different RUN_ID (--run-id)")
            print(f"  2. Delete the existing directory: {run_id_dir}")
            continue
        
        print(f"\n=== Running test suite: {test_suite} ===")
        
        gpu_monitor = GPUMonitor(interval=10) if HAS_GPU_MONITOR else None
        
        for nc, np, ni, no in product(max_concurrency, num_prompts, random_input_len, random_output_len):
            param_dir = f"{test_suite}/{run_id}/{nc}-{np}-i{ni}-o{no}"
            output_dir = os.path.join(output_base, param_dir)
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            log_file = os.path.join(output_dir, f"bench-{test_suite}-{nc}-{np}-i{ni}-o{no}.log")
            
            if gpu_monitor:
                gpu_monitor.start_monitoring("monitor/logs", model_name, param_dir)
            
            cmd = [
                "vllm", "bench", "serve",
                "--backend", "openai-chat",
                "--endpoint", "/v1/chat/completions",
                "--dataset-name", test_params.get("dataset-name", "random"),
                "--random-input-len", str(ni),
                "--random-output-len", str(no),
                "--model", str(model_path),
                "--trust-remote-code",
                "--base-url", base_url,
                "--num-prompts", str(np),
                "--max-concurrency", str(nc),
                "--temperature", str(temperature),
                "--seed", str(seed),
                "--metric_percentiles", "95,99",
                "--served-model-name", str(model_name),
                "--ready-check-timeout-sec", str(ready_timeout),
            ]
            
            print(f"Running: {' '.join(cmd)}")
            print(f"Log file: {log_file}")
            
            log_f = open(log_file, "w")
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            
            for line in process.stdout:
                print(line, end="")
                log_f.write(line)
            
            process.wait()
            log_f.close()
            
            if gpu_monitor:
                gpu_log = gpu_monitor.stop_monitoring()
                if gpu_log:
                    generate_gpu_charts(gpu_log, output_dir)
            
            print(f"Completed: {log_file}")
            time.sleep(30)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Run vLLM benchmark")
    parser.add_argument("--chip", type=str, required=True, 
                        help="Chip name to test (e.g., hygon_bw1000, kunlun_p800, nvidia_h100)")
    parser.add_argument("--test-suite", type=str, default=None,
                        help=f"Test suite to run (default: all). Available: {', '.join(TEST_SUITES)}")
    parser.add_argument("--run-id", type=str, default=RUN_ID,
                        help=f"Run ID to identify this test run (default: {RUN_ID})")
    args = parser.parse_args()
    
    yaml_path = os.path.join(os.path.dirname(__file__), "config", "models_scenarios.yaml")
    
    with open(yaml_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    base_config = config.get("base_config", {})
    params_config = base_config.get("params", {})
    models = config.get("models", {})
    
    chip_name = args.chip.lower()
    if chip_name not in models:
        print(f"Error: Chip '{chip_name}' not found in config. Available chips: {', '.join(models.keys())}")
        return
    
    test_suites_to_run = [args.test_suite] if args.test_suite else TEST_SUITES
    
    if args.test_suite and args.test_suite not in TEST_SUITES:
        print(f"Error: Test suite '{args.test_suite}' not found. Available: {', '.join(TEST_SUITES)}")
        return
    
    run_id = args.run_id
    
    model_configs = models[chip_name]
    for model_config in model_configs:
        print(f"Processing chip: {chip_name}, model: {model_config.get('name')}")
        run_benchmark(chip_name, base_config, model_config, test_suites_to_run, run_id)
        print(f"Finished chip: {chip_name}, model: {model_config.get('name')}")

if __name__ == "__main__":
    main()