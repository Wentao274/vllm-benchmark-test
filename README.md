# vllm-benchmark-test测试说明
vllm benchmark serve test for llm

## 1. run_benchmark基准测试使用
#### 帮助信息：
**usage**: <br> 
run_benchmark.py [-h] --chip CHIP [--model MODEL]
                        [--test-suite TEST_SUITE] [--run-id RUN_ID]

**options**:<br>
--chip CHIP           Chip name to test <br>
--model MODEL         Model name to test (e.g., minimax-m2.5, Qwen3.5) <br>
--test-suite TEST_SUITE  Test suite to run, use "," split multiple test suite; if not specified, use TEST_SUITES list defined in scripts to run <br>
--run-id RUN_ID       Run ID to identify this test run, if not specified, use RUN_ID defined in scripts <br>

### 1.1 测试 hygon_bw1000 芯片上的 minimax-m2.5 模型
python run_benchmark.py --chip hygon_bw1000 --model minimax-m2.5

### 1.2 测试 hygon_bw1000 芯片上的 Qwen3.5 模型
python run_benchmark.py --chip hygon_bw1000 --model Qwen3.5

### 1.3 不指定模型则测试该芯片下所有配置的模型
python run_benchmark.py --chip hygon_bw1000

### 1.4 组合使用: 执行hygon_bw1000平台下Qwen3.5模型的test_01测试场景的第2次测试
python run_benchmark.py --chip hygon_bw1000 --model Qwen3.5 --test-suite test_01 --run-id 02

### 1.5 指定多个测试套件
python run_benchmark.py --chip kunlun_p800 --model qwen3.5-plus --test-suite test_05,test_06 --run-id 02



## 2. 如何生成单个平台下的单个模型的单次测试的性能变化
**命令**<br>
python parse_single_chip_model.py

#### 帮助信息：
usage:<br> 
parse_single_chip_model.py [-h] [--chip CHIP] [--model MODEL]
                            [--test-suite TEST_SUITE] [--run-id RUN_ID]

**options**:<br>
--chip CHIP           Chip name to test (e.g., Hygon_BW1000, Kunlun_P800, NVIDIA_H100)<br>
--model MODEL         Model name to test (e.g., MiniMax-M2.5)<br>
--test-suite TEST_SUITE  Test suite name (e.g., test_01)<br>
--run-id RUN_ID       Run ID (e.g., 01)<br>

#### 示例：
#### 生成 Hygon_BW1000 芯片上 MiniMax-M2.5 模型 test_01 测试的第01次报告
python parse_single_chip_model.py --chip Hygon_BW1000 --model MiniMax-M2.5 --test-suite test_01 --run-id 01

#### 使用默认参数
python parse_single_chip_model.py <br>

如果不指定任何参数，则默认使用CHIP_BASE_PATHS的第一个Key和代码中定义的MODEL_NAME, TEST_SUITES和RUN_ID

## 3. 如何生成不同芯片平台对比的性能报告
**命令**<br>
python chip_comparison.py

#### 帮助信息：
usage:<br> 
chip_comparison.py [-h] [--chip CHIP] [--model MODEL]
                          [--test-suite TEST_SUITE] [--run-id RUN_ID]

**options**:<br>
--chip CHIP           Chip names to compare, comma-separated (e.g., hygon_bw1000,nvidia_h100)<br>
--model MODEL         Model name to test (e.g., MiniMax-M2.5)<br>
--test-suite TEST_SUITE  Test suite name (e.g., test_01)<br>
--run-id RUN_ID       Run ID (e.g., 01)<br>

#### 示例：
#### 生成 Hygon_BW1000 和 NVIDIA_H100 芯片上 MiniMax-M2.5 模型 test_01 测试的第01次对比报告
python chip_comparison.py --chip hygon_bw1000,nvidia_h100 --model MiniMax-M2.5 --test-suite test_01 --run-id 01

#### 使用默认参数（对比所有芯片）
python chip_comparison.py

所有参数值大小写不敏感


## 4. 如何生成同一芯片下不同模型之间的性能对比报告

**命令**<br>
python model_comparison.py

#### 帮助信息：
usage:<br> 
model_comparison.py [-h] --chip CHIP --model MODEL
                          [--test-suite TEST_SUITE] [--run-id RUN_ID]

**options**:<br>
--chip CHIP           Chip platform (e.g., hygon_bw1000, nvidia_h100)<br>
--model MODEL         Model names to compare, separated by comma (e.g., MiniMax-M2.5-bf16,Qwen3.5-397B-A17B)<br>
--test-suite TEST_SUITE  Test suite name (e.g., test_01)<br>
--run-id RUN_ID       Run IDs for each model, separated by comma (e.g., 01 or 01,02)<br>

#### 示例：
##### 4.1 对比同一run-id（所有模型使用相同的run-id）
python model_comparison.py --chip hygon_bw1000 --model "MiniMax-M2.5-bf16,Qwen3.5-397B-A17B" --test-suite test_01 --run-id 01

##### 4.2 对比不同run-id（第一个模型用01，第二个用02）
python model_comparison.py --chip hygon_bw1000 --model "MiniMax-M2.5-bf16,Qwen3.5-397B-A17B" --test-suite test_01 --run-id 01,02

##### 4.3 使用默认参数（test_01, run-id 01）
python model_comparison.py --chip hygon_bw1000 --model "MiniMax-M2.5-bf16,Qwen3.5-397B-A17B"

**注意**：所有参数值大小写不敏感

#### 输出说明：
- 输出目录：`analysis/<chip>_comparison/<test_suite>/<concurrency-np-iNi-oNo>/`
- 生成文件：
  - `concurrency<XXX>_comparison.csv` - CSV格式对比数据
  - `concurrency<XXX>_comparison.png` - 可视化图表
  - `concurrency<XXX>_comparison.md` - Markdown格式报告
- 汇总报告：`analysis/<chip>_comparison/<test_suite>/summary.md`
