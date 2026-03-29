# vllm-benchmark-test测试说明
vllm benchmark serve test for llm

## 运行所有测试套件
python run_benchmark.py --chip kunlun_p800
## 只运行 test_01
python run_benchmark.py --chip kunlun_p800 --test-suite test_01


## 指定run_id的使用方式：
### 使用命令行参数指定 run_id
python run_benchmark.py --chip kunlun_p800 --run-id 02
### 不指定则使用代码中定义的默认值 (01)
python run_benchmark.py --chip kunlun_p800
### 组合使用
python run_benchmark.py --chip kunlun_p800 --test-suite test_01 --run-id 03

**【帮助信息】**

usage: run_benchmark.py [-h] --chip CHIP [--test-suite TEST_SUITE]
                        [--run-id RUN_ID]

options: <br>
-- chip CHIP                Chip name to test <br>
-- test-suite TEST_SUITE    Test suite to run <br>
--run-id RUN_ID            Run ID to identify this test run (default: 01) <br>

# how to generate single models analysis based on diffierent concurrency
python parse_single_chip_model.py

# how to generate comparison report between chips
python chip_comparison.py

# how to generate comparison report between models
python model_comparison.py
