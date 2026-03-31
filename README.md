# vllm-benchmark-test测试说明
vllm benchmark serve test for llm

##使用方式：
### 测试 hygon_bw1000 芯片上的 minimax-m2.5 模型
python run_benchmark.py --chip hygon_bw1000 --model minimax-m2.5
### 测试 hygon_bw1000 芯片上的 Qwen3.5 模型
python run_benchmark.py --chip hygon_bw1000 --model Qwen3.5
### 不指定模型则测试该芯片下所有配置的模型
python run_benchmark.py --chip hygon_bw1000
### 组合使用: 执行hygon_bw1000平台下Qwen3.5模型的test_01测试场景的第2次测试
python run_benchmark.py --chip hygon_bw1000 --model Qwen3.5 --test-suite test_01 --run-id 02

### 帮助信息：
usage: run_benchmark.py [-h] --chip CHIP [--model MODEL]
                        [--test-suite TEST_SUITE] [--run-id RUN_ID]

**options**:<br>
--chip CHIP           Chip name to test <br>
--model MODEL         Model name to test (e.g., minimax-m2.5, Qwen3.5) <br>
--test-suite TEST_SUITE  Test suite to run <br>
--run-id RUN_ID       Run ID to identify this test run <br>

# how to generate single models analysis based on diffierent concurrency
python parse_single_chip_model.py

# how to generate comparison report between chips
python chip_comparison.py

# how to generate comparison report between models
python model_comparison.py
