# 精度测试比对报告生成工具

## 概述

`generate_accuracy_report.py` 是一个用于生成模型精度测试比对报告的工具。它支持多个芯片平台的测试结果比对，自动解析不同任务的日志文件，生成Markdown格式的比对报告。

## 功能特点

- 支持多芯片平台比对（以第一个芯片为基准）
- 支持指定模型比对
- 动态匹配测试任务（根据目录结构自动识别）
- 自动提取不同量化级别的测试结果
- 生成差值和百分比比对数据
- mmlu_pro和ruler任务支持详细子数据集比对

## 使用方式

### 基本命令

```bash
python generate_accuracy_report.py --chip <chip1,chip2,...> [--task <task1,task2,...>] [--model <model1,model2,...>]
```

### 参数说明

| 参数 | 必选 | 说明 |
|------|------|------|
| `--chip` | 是 | 芯片名称，多个芯片用逗号分隔。第一个芯片作为基准，如：`nvidia_h100,hygon_bw1000` |
| `--task` | 否 | 任务名称，多个任务用逗号分隔。不指定则比对所有任务。如：`IFBench,lm-eval:gsm_plus` |
| `--model` | 否 | 模型名称，多个模型用逗号分隔。不指定则比对所有模型。如：`GLM-5,MiniMax-M2.5` |

### 任务名称匹配规则

任务名称根据目录结构动态匹配：

1. **直接目录名**：如 `IFBench`，对应 `model-accuracy-test/IFBench/` 目录
2. **lm-eval子任务**：如 `lm-eval:gsm_plus`，对应 `model-accuracy-test/lm-eval/gsm_plus/` 目录
3. **简写形式**：`gsm_plus` 自动匹配为 `lm-eval:gsm_plus`

如果指定的任务名称未匹配到对应目录，将提示警告信息。

### 目录结构要求

```
model-accuracy-test/
├── generate_accuracy_report.py
├── accuracy_report/           # 生成的报告目录
├── IFBench/                   # IFBench测试结果
│   └── <模型名>/
│       └── <芯片名>/
│           └── <量化级别>_IFBench.log
├── <其他任务>/               # 其他任务测试结果
│   └── <模型名>/
│       └── <芯片名>/
│           └── <量化级别>_<任务名>.log
└── lm-eval/                   # lm-eval测试结果
    ├── gsm_plus/
    │   └── <模型名>/
    │       └── <芯片名>/
    │           └── <量化级别>_gsm_plus.log
    ├── mmlu_pro/
    │   └── ...
    └── ruler/
        └── ...
```

## 使用场景

### 场景1：单芯片测试报告

生成单个芯片的测试结果汇总：

```bash
python generate_accuracy_report.py --chip nvidia_h100
```

输出文件：`accuracy_report/nvidia_h100_all_task.md`

### 场景2：双芯片比对

比对两个芯片平台的测试结果：

```bash
python generate_accuracy_report.py --chip nvidia_h100,hygon_bw1000
```

输出文件：`accuracy_report/nvidia_h100&hygon_bw1000_all_task.md`

### 场景3：多芯片比对（3个及以上）

比对多个芯片平台，每个芯片的差值和百分比紧跟其数值列：

```bash
python generate_accuracy_report.py --chip nvidia_h100,hygon_bw1000,metax_c550
```

输出文件：`accuracy_report/nvidia_h100&hygon_bw1000&metax_c550_all_task.md`

### 场景4：指定任务比对

只比对特定任务的测试结果：

```bash
# 单个任务
python generate_accuracy_report.py --chip nvidia_h100,hygon_bw1000 --task IFBench

# 多个任务
python generate_accuracy_report.py --chip nvidia_h100,hygon_bw1000 --task IFBench,lm-eval:mmlu_pro

# 简写形式
python generate_accuracy_report.py --chip nvidia_h100,hygon_bw1000 --task gsm_plus,mmlu_pro
```

输出文件：
- 单任务：`accuracy_report/nvidia_h100&hygon_bw1000_ifbench.md`
- 多任务：`accuracy_report/nvidia_h100&hygon_bw1000_ifbench_mmlupro.md`

### 场景5：指定模型比对

只比对特定模型的测试结果：

```bash
# 单个模型
python generate_accuracy_report.py --chip nvidia_h100,hygon_bw1000 --model GLM-5

# 多个模型
python generate_accuracy_report.py --chip nvidia_h100,hygon_bw1000 --model GLM-5,MiniMax-M2.5
```

输出文件：
- 单模型：`accuracy_report/nvidia_h100&hygon_bw1000_GLM-5_all_task.md`
- 多模型：`accuracy_report/nvidia_h100&hygon_bw1000_GLM-5_MiniMax-M2.5_all_task.md`

### 场景6：组合筛选

同时指定芯片、任务和模型：

```bash
python generate_accuracy_report.py --chip nvidia_h100,metax_c550 --task IFBench,lm-eval:mmlu_pro --model GLM-5
```

输出文件：`accuracy_report/nvidia_h100&metax_c550_GLM-5_ifbench_mmlupro.md`

## 支持的日志解析格式

### 1. IFBench

从日志文件提取：
- `Generating eval_results_strict Accuracy: 0.376667` → Strict精度
- `Generating eval_results_loose Accuracy: 0.410000` → Loose精度

### 2. lm-eval:gsm_plus

从日志表格提取：
- `flexible-extract` 行的 Value 列值
- `strict-match` 行的 Value 列值

### 3. lm-eval:mmlu_pro

从日志表格提取：
- `|mmlu_pro|` 行的 Value 列值（总体精度）
- 各子任务（biology、business等）的详细数据

### 4. lm-eval:ruler

从日志表格提取：
- 所有子任务（niah_multikey、ruler_cwe等）的 Value 列值
- 计算平均值作为总体精度

### 5. 自定义任务

对于其他任务，脚本会尝试解析日志文件中的数值。如需支持特定格式，可扩展 `parse_log_file` 函数。

## 报告格式说明

### 汇总表格

每个模型的所有测试任务汇总在一个表格中：

```markdown
| Task | nvidia_h100(FP8) | hygon_bw1000(W8A8) | 差值 | 百分比 |
|------|------|------|------|------|
| IFBench (Strict) | 0.6667 | 0.3767 | -0.2900 | - 43.50% |
| IFBench (Loose) | 0.7133 | 0.4100 | -0.3033 | - 42.52% |
| lm-eval:mmlu_pro | 0.7858 | 0.7893 | 0.0035 | + 0.45% |
```

### 详细比对表格

mmlu_pro和ruler任务的子数据集详细比对放在报告末尾：

```markdown
## GLM-5模型 - mmlu_pro任务子数据集详细比对结果

| Item | nvidia_h100(FP8) | hygon_bw1000(W8A8) | 差值 | 百分比 |
|------|------|------|------|------|
| biology | 0.8968 | 0.8898 | -0.0070 | - 0.78% |
| business | 0.8390 | 0.8137 | -0.0253 | - 3.02% |
```

## 注意事项

1. **量化级别提取**：从日志文件名第一个下划线前提取，如 `INT4_IFBench.log` → `INT4`
2. **过滤规则**：自动过滤包含 `append-think` 或 `append` 的量化级别
3. **基准选择**：第一个芯片作为基准，其他芯片与其计算差值
4. **多量化级别**：同一芯片的多个量化级别分别列出，以第一个非空值作为基准
5. **任务匹配**：任务名称根据目录结构动态匹配，未匹配到的任务会提示警告

## 示例输出

### 完整命令示例

```bash
# 生成 nvidia_h100 和 metax_c550 的所有任务比对报告
python generate_accuracy_report.py --chip nvidia_h100,metax_c550

# 生成三个芯片的 IFBench 任务比对报告
python generate_accuracy_report.py --chip nvidia_h100,hygon_bw1000,metax_c550 --task IFBench

# 生成指定任务的比对报告
python generate_accuracy_report.py --chip nvidia_h100,hygon_bw1000 --task IFBench,lm-eval:mmlu_pro,lm-eval:ruler

# 生成指定模型的比对报告
python generate_accuracy_report.py --chip nvidia_h100,metax_c550 --model GLM-5,MiniMax-M2.5

# 组合筛选：指定芯片、任务和模型
python generate_accuracy_report.py --chip nvidia_h100,metax_c550 --task IFBench,lm-eval:mmlu_pro --model GLM-5
```

### 输出文件命名规则

| 参数组合 | 文件名格式 |
|---------|-----------|
| 仅芯片 | `{chips}_all_task.md` |
| 芯片 + 任务 | `{chips}_{tasks}.md` |
| 芯片 + 模型 | `{chips}_{models}_all_task.md` |
| 芯片 + 任务 + 模型 | `{chips}_{models}_{tasks}.md` |

示例：
- `nvidia_h100&metax_c550_all_task.md`
- `nvidia_h100&metax_c550_ifbench.md`
- `nvidia_h100&metax_c550_GLM-5_all_task.md`
- `nvidia_h100&metax_c550_GLM-5_ifbench_mmlupro.md`
