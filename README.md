# vllm-benchmark-test测试说明
vllm benchmark serve test for llm

## 1. run_benchmark基准测试使用
#### 帮助信息：
**usage**: <br> 
run_benchmark.py [-h] --chip CHIP [--model MODEL]
                        [--test-suite TEST_SUITE] [--run-id RUN_ID]

**options**:<br>
--chip CHIP           指定测试的芯片的名称，该名称必须在models_scenarios.yaml配置文件中存在（在models下列出），比如hygon_bw1000 <br>
--model MODEL         指定要测试的模型名，该名称必须在models_scenarios.yaml中指定的芯片名称下存在（在models下的对应芯片名称下列出） (比如, MiniMax-M2.5-bf16, MiniMax-M2.5-W8A8, Qwen3.5-397B-A17B)，如果不指定，则使用配置文件中指定芯片的第一个模型名 <br>
--test-suite TEST_SUITE  指定要测试的测试套件，该名称必须在models_scenarios.yaml中存在（在params下）, 多个测试套件使用逗号分隔; 如果不指定，默认执行脚本中定义的 TEST_SUITES列表中的测试套件 <br>
--run-id RUN_ID       指定本次测试的测试序号, 如果不指定，使用脚本中定义的默认测试序号 <br>

### 1.1 测试 hygon_bw1000 芯片上的 MiniMax-M2.5-bf16 模型
python run_benchmark.py --chip hygon_bw1000 --model MiniMax-M2.5-bf16

### 1.2 测试 nvidia_h100 芯片上的 MiniMax-M2.5 模型
python run_benchmark.py --chip nvidia_h100 --model MiniMax-M2.5

### 1.3 不指定模型则默认测试该芯片下的第一个模型
python run_benchmark.py --chip hygon_bw1000

### 1.4 组合使用: 执行hygon_bw1000平台下MiniMax-M2.5-W8A8模型的test_01测试场景的第2次测试
python run_benchmark.py --chip hygon_bw1000 --model MiniMax-M2.5-W8A8 --test-suite test_01 --run-id 02

### 1.5 指定多个测试套件
python run_benchmark.py --chip kunlun_p800 --model MiniMax-M2.5-W8A8-INT8-Dynamic --test-suite test_05,test_06 --run-id 02


## 2. 配置文件说明

### 2.1 config/models_scenarios.yaml
vLLM基准测试的场景配置，包含：
- base_config: 基础配置（URL、端口、测试参数等）
- models: 各芯片平台下的模型配置

### 2.2 config/chip_conf.yaml
芯片配置信息，用于生成报告时的芯片详情

### 2.3 config/model_deployment.yaml
模型部署配置，用于生成报告时的vLLM启动配置信息


## 3. GPU监控
gpu_monitor.py 提供GPU监控功能，支持：
- nvidia-smi (NVIDIA GPU)
- hy-smi (海光GPU)
- rocm-smi (AMD GPU)
- xpu-smi (昆仑芯 GPU)
- mx-smi (沐曦 GPU)

监控数据会保存到 monitor/logs 目录，并生成GPU使用趋势图表（待实现）。


## 4. 数据目录结构

### 4.1 Benchmark 测试结果目录结构

测试结果保存在 `reports` 目录下。

#### 目录结构：
```
reports/benchmark/<chip_name>/<model_name>/<test_suite>/<run_id>/<concurrency>-<num_prompts>-i<input_len>-o<output_len>/
```

#### 示例：
```
reports/benchmark/hygon_bw1000/MiniMax-M2.5-bf16/test_01/01/1-320-i10240-o256/
reports/benchmark/nvidia_h100/MiniMax-M2.5/test_03/02/4-100-i194560-o1024/
```

#### 说明：
- `benchmark`: 固定目录名
- `{chip_name}`: 芯片平台（如 `hygon_bw1000`, `nvidia_h100`, `kunlun_p800`）
- `{model_name}`: 模型名称（如 `MiniMax-M2.5`, `MiniMax-M2.5-bf16`, `MiniMax-M2.5-W8A8-INT8-Dynamic`）
- `{test_suite}`: 测试套件（如 `test_01`, `test_03`, `test_05` 等）
- `{run_id}`: 测试运行 ID（如 `01`, `02`）
- `{concurrency}-{num_prompts}-i{input_len}-o{output_len}`: 并发数-提示数-输入长度-输出长度

#### 输出文件：
- `bench-<test_suite>-<conc>-<num_prompts>-i<input>-o<output>.log`: 测试日志
- `bench-<test_suite>-<conc>-<num_prompts>-i<input>-o<output>.jsonl`: 详细结果数据


### 4.2 GPU 监控日志目录结构

在运行 benchmark 测试时，会自动启动 GPU 监控，记录 GPU 使用情况。监控日志保存在 `monitor` 目录下。

#### 目录结构：
```
monitor/logs/<chip_name>/<model_name>/<test_suite>/<run_id>/<concurrency>-<num_prompts>-i<input_len>-o<output_len>/
```

#### 示例：
```
monitor/logs/nvidia_h100/MiniMax-M2.5/test_01/01/1-320-i10240-o256/gpu_monitor_20260430123341.log
monitor/logs/kunlun_p800/MiniMax-M2.5-W8A8-INT8-Dynamic/test_03/01/4-100-i194560-o1024/gpu_monitor_20260430143239.log
```

#### 说明：
- `logs`: 固定目录名
- `{chip_name}`: 芯片平台（如 `nvidia_h100`, `kunlun_p800`, `hygon_bw1000`）
- `{model_name}`: 模型名称（如 `MiniMax-M2.5`, `MiniMax-M2.5-W8A8-INT8-Dynamic`）
- `{test_suite}`: 测试套件（如 `test_01`, `test_03`, `test_05` 等）
- `{run_id}`: 测试运行 ID（如 `01`, `02`）
- `{concurrency}-{num_prompts}-i{input_len}-o{output_len}`: 并发数-提示数-输入长度-输出长度
- `gpu_monitor_{timestamp}.log`: GPU 监控日志文件

#### 日志格式：
日志文件为 CSV 格式，包含以下字段：
- Time: 时间戳
- GPU: GPU 索引
- Name: GPU 名称
- Used_MB: 已使用显存 (MB)
- Total_MB: 总显存 (MB)
- Utilization_%: GPU 利用率 (%)
- Memory_%: 显存利用率 (%)
- Temperature_C: 温度 (°C)


## 5. run_benchmark_gen_report 一键运行基准测试并自动生成报告
该脚本整合了 `run_benchmark.py` 的功能，在执行完 benchmark 测试后，会自动检测 test-suite 目录下的 run-id 数量：
- 如果只有 1 个 run-id，则调用 `parse_single_chip_model.py` 生成单次测试的性能报告
- 如果有多个 run-id，则调用 `compare_runids.py` 生成多个 run-id 的对比报告

**命令**<br>
python run_benchmark_gen_report.py

#### 帮助信息：
usage:<br> 
run_benchmark_gen_report.py [-h] --chip CHIP [--model MODEL]
                              [--test-suite TEST_SUITE] [--run-id RUN_ID]
                              [--skip-benchmark] [--only-report]
                              [-c CONCURRENCY]

**options**:<br>
--chip CHIP             指定要测试的芯片名称 (e.g., hygon_bw1000, kunlun_p800, nvidia_h100)<br>
--model MODEL           指定要测试的模型名称 (e.g., MiniMax-M2.5-bf16, MiniMax-M2.5-W8A8, Qwen3.5-397B-A17B). 如果不指定，使用指定芯片的配置中的第一个模型.<br>
--test-suite TEST_SUITE  指定要测试的套件 (default: all). Available: test_01, test_03, test_05, test_06, test_07<br>
--run-id RUN_ID         指定本次测试的测试序号 (default: 01)<br>
--skip-benchmark        跳过benchmark测试，只使用已有测试数据生成报告<br>
--only-report           类似--skip-benchmark参数，不执行benchmark测试，只生成报告<br>
-c CONCURRENCY          指定要生成哪个并发级别的报告, 多个使用逗号分隔 (e.g., 1,2,4,8,10)<br>

#### 示例：
##### 5.1 完整流程：运行 benchmark 并自动生成报告
python run_benchmark_gen_report.py --chip hygon_bw1000

##### 5.2 仅基于现有数据生成报告（不运行 benchmark）
python run_benchmark_gen_report.py --chip hygon_bw1000 --only-report
- 跳过所有 benchmark 相关操作，直接生成报告
- 适用于已有 benchmark 数据，只需生成报告的场景

##### 5.3 跳过 benchmark 执行，直接生成报告
python run_benchmark_gen_report.py --chip hygon_bw1000 --skip-benchmark
- 跳过 benchmark 执行但保留配置加载过程
- 与 `--only-report` 效果相同，提供参数别名以兼容不同使用习惯

##### 5.4 指定测试套件和模型
python run_benchmark_gen_report.py --chip hygon_bw1000 --model MiniMax-M2.5-bf16 --test-suite test_01

##### 5.5 指定并发数，只生成特定并发级别的报告
- python run_benchmark_gen_report.py --chip hygon_bw1000 -c 1
- python run_benchmark_gen_report.py --chip hygon_bw1000 -c 1,2,4,8,10

##### 5.6 组合使用：指定并发数并只生成报告
python run_benchmark_gen_report.py --chip hygon_bw1000 --only-report -c 1,2,4

**说明**：
- 该脚本会自动判断 test-suite 目录下有多少个 run-id，根据数量决定生成单次报告还是对比报告
- `-c` / `--concurrency` 参数用于指定只生成特定并发级别的报告，方便快速查看特定并发下的性能
- 如果不指定 `-c` 参数，默认会对所有并发级别生成报告


## 6. 如何生成单个平台下的单个模型的单次测试的性能变化
**命令**<br>
python parse_single_chip_model.py

#### 帮助信息：
usage:<br> 
parse_single_chip_model.py [-h] [--chip CHIP] [--model MODEL]
                            [--test-suite TEST_SUITE] [--run-id RUN_ID]
                            [--concurrency CONCURRENCY]

**options**:<br>
--chip CHIP               Chip name to test (e.g., Hygon_BW1000, Kunlun_P800, NVIDIA_H100)<br>
--model MODEL             Model name to test (e.g., MiniMax-M2.5-bf16, MiniMax-M2.5-W8A8-INT8-Dynamic, MiniMax-M2.5)<br>
--test-suite TEST_SUITE   Test suite name (e.g., test_01)<br>
--run-id RUN_ID           Run ID (e.g., 01)<br>
--concurrency CONCURRENCY Specific concurrency levels to include, comma-separated (e.g., 1,2,4,8,10)<br>

#### 示例：
##### 6.1 生成 Hygon_BW1000 芯片上 MiniMax-M2.5-bf16 模型 test_01 测试的第01次报告
python parse_single_chip_model.py --chip hygon_bw1000 --model MiniMax-M2.5-bf16 --test-suite test_01 --run-id 01

##### 6.2 生成 kunlun_p800 芯片上 MiniMax-M2.5-W8A8-INT8-Dynamic 模型 test_01 测试的第01次报告
python parse_single_chip_model.py --chip kunlun_p800 --model MiniMax-M2.5-W8A8-INT8-Dynamic --test-suite test_01 --run-id 01

##### 6.3 生成 nvidia_h100 芯片上 MiniMax-M2.5 模型 test_01 测试的第01次报告
python parse_single_chip_model.py --chip nvidia_h100 --model MiniMax-M2.5 --test-suite test_01 --run-id 01

##### 6.4 使用默认参数
python parse_single_chip_model.py

如果不指定任何参数，则默认使用CHIP_BASE_PATHS的第一个Key和代码中定义的MODEL_NAME, TEST_SUITES和RUN_ID

##### 6.5 指定特定并发数
python parse_single_chip_model.py --chip hygon_bw1000 --model MiniMax-M2.5-bf16 --test-suite test_01 --run-id 01 --concurrency 1,2,4

## 7. 如何生成同一芯片、同一模型下不同测试run-id的性能对比报告

**命令**<br>
python compare_runids.py

#### 帮助信息：
usage:<br> 
compare_runids.py [-h] --chip CHIP --model MODEL
                               [--test-suite TEST_SUITE] [--run-id RUN_ID]
                               [--concurrency CONCURRENCY]

**options**:<br>
--chip CHIP               Chip platform (e.g., Hygon_BW1000, Kunlun_P800, NVIDIA_H100)<br>
--model MODEL             Model name to test (e.g., MiniMax-M2.5-bf16, MiniMax-M2.5-W8A8-INT8-Dynamic, MiniMax-M2.5)<br>
--test-suite TEST_SUITE   Test suite name (e.g., test_01, test_05)<br>
--run-id RUN_ID           Run IDs to compare, comma-separated (e.g., 01,02)<br>
--concurrency CONCURRENCY Specific concurrency levels to compare, comma-separated (e.g., 1,2,4,8,10)<br>

#### 示例：
##### 7.1 对比同一芯片、同一模型下的两次测试run-id
python compare_runids.py --chip hygon_bw1000 --model MiniMax-M2.5-bf16 --test-suite test_03 --run-id '01','02'
<br>或者<br>
python compare_runids.py --chip hygon_bw1000 --model MiniMax-M2.5-bf16 --test-suite test_03 --run-id '01,02'

##### 7.2 对比三次测试run-id
python compare_runids.py --chip hygon_bw1000 --model MiniMax-M2.5-bf16 --test-suite test_03 --run-id '01','02','03'

##### 7.3 使用默认参数（对比01和02）
python compare_runids.py

##### 7.4 指定特定并发数进行对比
python compare_runids.py --chip hygon_bw1000 --model MiniMax-M2.5-bf16 --test-suite test_01 --run-id '01,02' --concurrency 1,2,4
python compare_runids.py -c 1,4,8 --chip hygon_bw1000 --model MiniMax-M2.5-bf16 --run-id '01,02'

##### 7.5 生成多I/O测试报告（test_05）
python compare_runids.py --chip hygon_bw1000 --model MiniMax-M2.5-W8A8 --test-suite test_05 --run-id 01,02

**说明**：
- 该脚本用于对比同一芯片、同一模型在不同测试运行（run-id）下的性能差异
- 方便分析不同配置、不同优化或不同版本间的性能变化
- run-id之间用逗号分隔，至少需要2个run-id才能进行对比
- 支持多I/O测试场景（test_05），会为每个I/O组合生成单独的对比报告
- 输出目录：`analysis/single_chip/<chip_name>/<model_name>/compare_run/<test_suite>/run_<run_id1>_<run_id2>/`

#### 输出说明：
- 输出目录：`analysis/single_chip/<chip_name>/<model_name>/compare_run/<test_suite>/run_<run_id1>_<run_id2>/`
- 生成文件：
  - `runid_comparison.csv` - CSV格式对比数据
  - `runid_comparison.png` - 可视化图表
  - `<model_name>_<chip_name>_<test_suite>_runid_compare_<run_ids>.md` - Markdown格式报告
- 多I/O测试（test_05）额外生成：
  - `<io_pair>/` 子目录，每个I/O组合一个目录
  - 包含各I/O组合的单独对比报告


## 8. 如何生成同一芯片下不同模型之间的性能对比报告

**命令**<br>
python model_comparison.py

#### 帮助信息：
usage:<br> 
model_comparison.py [-h] --chip CHIP --model MODEL
                          [--test-suite TEST_SUITE] [--run-id RUN_ID]
                          [-c CONCURRENCY]

**options**:<br>
--chip CHIP           Chip platform (e.g., hygon_bw1000, nvidia_h100)<br>
--model MODEL         Model names to compare, separated by comma (e.g., MiniMax-M2.5-bf16,Qwen3.5-397B-A17B)<br>
--test-suite TEST_SUITE  Test suite name (e.g., test_01)<br>
--run-id RUN_ID       Run IDs for each model, separated by comma (e.g., 01 or 01,02)<br>
-c CONCURRENCY        Specific concurrency levels to compare, comma-separated (e.g., 1,2,4,8,10)<br>

#### 示例：
##### 8.1 对比同一run-id（所有模型使用相同的run-id）
python model_comparison.py --chip hygon_bw1000 --model "MiniMax-M2.5-bf16,Qwen3.5-397B-A17B" --test-suite test_01 --run-id 01

##### 8.2 对比不同run-id（第一个模型用01，第二个用02）
python model_comparison.py --chip hygon_bw1000 --model "MiniMax-M2.5-bf16,Qwen3.5-397B-A17B" --test-suite test_01 --run-id '01,02'

##### 8.3 使用默认参数（test_01, run-id 01）
python model_comparison.py --chip hygon_bw1000 --model "MiniMax-M2.5-bf16,Qwen3.5-397B-A17B"

##### 8.4 指定特定并发级别
python model_comparison.py --chip hygon_bw1000 --model "MiniMax-M2.5-bf16,Qwen3.5-397B-A17B" -c 1,2,4,8

**注意**：所有参数值大小写不敏感

#### 输出说明：
**注：此脚本输出的比对报告是每个并发级别单独分开的**
- 输出目录：`analysis/<chip>_comparison/<test_suite>/<concurrency-np-iNi-oNo>/`
- 生成文件：
  - `concurrency<XXX>_comparison.csv` - CSV格式对比数据
  - `concurrency<XXX>_comparison.png` - 可视化图表
  - `concurrency<XXX>_comparison.md` - Markdown格式报告
- 汇总报告：`analysis/<chip>_comparison/<test_suite>/summary.md`


## 9. 如何生成全并发级别的多模型对比报告

**命令**<br>
python model_comparison_all_concurrency.py

与 `model_comparison.py` 不同的是，该脚本会将所有并发级别的对比数据合并到一个Markdown报告中，而不是按每个并发级别分别生成报告。

#### 帮助信息：
usage:<br>
model_comparison_all_concurrency.py [-h] --chip CHIP --model MODEL
                                       [--test-suite TEST_SUITE] [--run-id RUN_ID]

**options**:<br>
--chip CHIP           Chip platform (e.g., hygon_bw1000, nvidia_h100)<br>
--model MODEL         Model names to compare, separated by comma (e.g., MiniMax-M2.5-bf16,Qwen3.5-397B-A17B)<br>
--test-suite TEST_SUITE  Test suite name (e.g., test_01)<br>
--run-id RUN_ID       Run IDs for each model, separated by comma (e.g., 01 or 01,02)<br>

#### 示例：
##### 9.1 对比两个模型的所有并发级别
python model_comparison_all_concurrency.py --chip hygon_bw1000 --model "MiniMax-M2.5-bf16,MiniMax-M2.5-W8A8"

##### 9.2 指定测试套件和run-id
python model_comparison_all_concurrency.py --chip hygon_bw1000 --model "MiniMax-M2.5-bf16,MiniMax-M2.5-W8A8" --test-suite test_01 --run-id '01,02'

**注意**：所有参数值大小写不敏感

#### 输出说明：
- 输出目录：`analysis/<chip_name>_comparison_all_concurrency/`
- 生成文件：
  - `all_concurrency_comparison.csv` - CSV格式对比数据（包含所有并发级别）
  - `all_concurrency_comparison.png` - 可视化图表（跨所有并发级别）
  - `all_concurrency_comparison.md` - Markdown格式报告（合并所有并发级别）
- 报告特点：
  - 包含各并发级别的详细对比表格
  - 包含模型性能对比柱状图
  - 包含分析小结（以第一个模型为基准，显示其他模型的性能改进百分比）


## 10. 如何生成不同芯片平台对比的性能报告
**命令**<br>
python chip_comparison.py

#### 帮助信息：
usage:<br> 
chip_comparison.py [-h] [--chip CHIP] [--model MODEL]
                          [--test-suite TEST_SUITE] [--run-id RUN_ID] [--concurrency CONCURRENCY]

**options**:<br>
--chip CHIP           Chip names to compare, comma-separated (e.g., hygon_bw1000,nvidia_h100)<br>
--model MODEL         Model names for each chip, comma-separated and must match chip order (e.g., MiniMax-M2.5-bf16,MiniMax-M2.5)<br>
--test-suite TEST_SUITE  Test suite name (e.g., test_01)<br>
--run-id RUN_ID       Run IDs, can be '01' for all chips or '01,02,03' for each chip<br>
--concurrency CONCURRENCY  Specific concurrency levels to compare, comma-separated (e.g., 1,2,4,8,10)<br>

#### 示例：
##### 10.1 所有芯片使用相同 run-id，默认使用第一个模型
python chip_comparison.py --chip hygon_bw1000,kunlun_p800,nvidia_h100 --test-suite test_01 --run-id 01

##### 10.2 每个芯片使用不同的模型（一一对应）
python chip_comparison.py --chip hygon_bw1000,kunlun_p800,nvidia_h100 --model "MiniMax-M2.5-bf16,MiniMax-M2.5-W8A8-INT8-Dynamic,MiniMax-M2.5" --test-suite test_01 --run-id 01

##### 10.3 每个芯片使用不同的 run-id（按 --chip 参数顺序一一对应）
python chip_comparison.py --chip hygon_bw1000,kunlun_p800,nvidia_h100 --test-suite test_01 --run-id '01,02,01'
- Hygon_BW1000 使用 run-id 01
- Kunlun_P800 使用 run-id 02
- NVIDIA_H100 使用 run-id 01

##### 10.4 使用默认参数（对比所有芯片，run-id 01）
python chip_comparison.py

##### 10.5 指定特定并发数进行对比
python chip_comparison.py --chip hygon_bw1000,nvidia_h100 --test-suite test_01 --concurrency 1,2,4,8

##### 10.6 使用简写-c指定并发数
python chip_comparison.py -c 1,4,8,16 --chip hygon_bw1000,nvidia_h100

**注意**：
- model 参数如果只有一个值，所有芯片使用相同模型
- model 参数如果有多个值（逗号分隔），按 --chip 参数顺序一一对应，每个芯片比对不同的模型
- run-id 参数如果只有一个值，所有芯片使用相同 run-id
- run-id 参数如果有多个值（逗号分隔），按 --chip 参数顺序一一对应
- 所有参数值大小写不敏感
