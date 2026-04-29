#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys

import yaml


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_args_from_config(config: dict) -> list[str]:
    args = []

    if "model" in config:
        args.extend(["--model", str(config["model"])])

    if "served_model_name" in config:
        args.extend(["--served-model-name", str(config["served_model_name"])])

    if "url" in config:
        args.extend(["--url", str(config["url"])])

    if "input_file" in config:
        args.extend(["--input-file", str(config["input_file"])])

    if "output_file" in config:
        args.extend(["--output-file", str(config["output_file"])])

    if "num_clients" in config:
        args.extend(["--num-clients", str(config["num_clients"])])

    if "max_active_conversations" in config:
        args.extend(
            ["--max-active-conversations", str(config["max_active_conversations"])]
        )

    if config.get("warmup_step", False):
        args.append("--warmup-step")

    if "extra_body_json" in config:
        extra_body_json_str = json.dumps(config["extra_body_json"])
        args.extend(["--extra-body-json", extra_body_json_str])

    if "csv_output" in config:
        args.extend(["--csv-output", str(config["csv_output"])])

    if "seed" in config:
        args.extend(["--seed", str(config["seed"])])

    if "trust_remote_code" in config:
        if config["trust_remote_code"]:
            args.append("--trust-remote-code")
        else:
            args.append("--no-trust-remote-code")

    if "max_num_requests" in config:
        args.extend(["--max-num-requests", str(config["max_num_requests"])])

    if "max_turns" in config:
        args.extend(["--max-turns", str(config["max_turns"])])

    if config.get("no_early_stop", False):
        args.append("--no-early-stop")

    if "limit_max_tokens" in config:
        args.extend(["--limit-max-tokens", str(config["limit_max_tokens"])])

    if "limit_min_tokens" in config:
        args.extend(["--limit-min-tokens", str(config["limit_min_tokens"])])

    if "request_rate" in config:
        args.extend(["--request-rate", str(config["request_rate"])])

    if "max_retries" in config:
        args.extend(["--max-retries", str(config["max_retries"])])

    if "conversation_sampling" in config:
        args.extend(["--conversation-sampling", str(config["conversation_sampling"])])

    if config.get("verify_output", False):
        args.append("--verify-output")

    if "request_timeout_sec" in config:
        args.extend(["--request-timeout-sec", str(config["request_timeout_sec"])])

    if config.get("no_stream", False):
        args.append("--no-stream")

    if config.get("excel_output", False):
        args.append("--excel-output")

    if config.get("verbose", False):
        args.append("--verbose")

    if config.get("print_content", False):
        args.append("--print-content")

    if "warmup_percentages" in config:
        args.extend(["--warmup-percentages", str(config["warmup_percentages"])])

    return args


def main():
    parser = argparse.ArgumentParser(
        description="Run multi-turn benchmark from configuration file"
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file (default: config.yaml)",
    )
    parser.add_argument(
        "--benchmark-script",
        type=str,
        default="benchmark_serving_multi_turn.py",
        help="Path to benchmark script (default: benchmark_serving_multi_turn.py)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the command without executing",
    )

    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    config = load_config(args.config)
    benchmark_args = build_args_from_config(config)

    cmd = ["python", args.benchmark_script] + benchmark_args

    print(f"Configuration: {args.config}")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 50)

    if args.dry_run:
        print("Dry run mode - command not executed.")
        sys.exit(0)

    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
