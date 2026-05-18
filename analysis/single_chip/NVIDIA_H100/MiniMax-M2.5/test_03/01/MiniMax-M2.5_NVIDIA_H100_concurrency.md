# MiniMax-M2.5模型在NVIDIA_H100上的Benchmark基准测试报告

<div align="center">
**测试日期：** 2026-05-18

</div>

---

## 测试场景
使用vllm bench serve基准测试工具对不同并发数，请求上下文长度下的性能变化趋势。

**主要采集指标**：

| 指标                  | 单位         | 含义                                 |
|---------------------|------------|------------------------------------|
| Request throughput  | req/s      | 请求吞吐量                              |
| Output token throughput | tok/s  | 输出token吞吐量                        |
| Total token throughput | tok/s   | 总token吞吐量                         |
| TTFT                | ms         | Time To First Token，首 token 延迟     |
| TPOT                | ms/token   | Time Per Output Token，每 token 生成时间 |
| ITL                 | ms         | Inter-Token Latency，token间延迟       |


## 🤖 芯片和模型配置信息

| 参数名称                    | NVIDIA_H100 |
|------------------------|-------------|
| **model_name** | MiniMax-M2.5 |
| **quantization_config** | FP16 |
| **model_size** | 215G |
| **max_position_embeddings** | 196608 |
| **temperature** | N/A |
| **top_k** | N/A |
| **top_p** | N/A |
| **transformers_version** | 4.46.1 |
| **vllm_version** | 0.15.1 |
| **python_version** | 3.12.3 |


## 🤖 vLLM启动配置信息

| 参数名称                   | NVIDIA_H100 |
|------------------------|-------------|
| **Model Name** | MiniMax-M2.5 |
| **Max Model Len** | 196608 |
| **Max Num Seqs** | 10 |
| **Max Num Batched Tokens** | 8192 |
| **Gpu Memory Utilization** | 0.85 |
| **Dtype** | default |
| **Block Size** | default |
| **Dp** | 1 |
| **Tp** | 8 |
| **Pp** | 1 |
| **Enable Export Parallel** | True |
| **Enable Auto Tool Choice** | True |
| **Tool Call Parser** | minimax_m2 |
| **Reasoning Parser** | minimax_m2 |

- **NVIDIA_H100**: 英伟达H100标准配置


## 📊 测试概览

| 项目            | 配置                                     | 备注  |
|---------------|----------------------------------------|-----|
| **数据集**       | random                                 |     |
| **并发数**       | 32, 64    |     |
| **总请求数**      | 1000                                    |     |
| **请求输入上下文长度** | 90000（87k）                             |     |
| **请求输出上下文长度** | 2000（1k）                             |     |
| **模型**        | MiniMax-M2.5                           |     |
| **被测芯片**      | NVIDIA_H100 |     |

---

## 📋 测试结果汇总

| 并发数 | 请求吞吐量 (req/s) | 输出Token吞吐量 (tok/s) | 总Token吞吐量 (tok/s) | TTFT P99 (ms) | TPOT P99 (ms) | ITL P99 (ms) |
| ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
| 32 | 0.15 | 307.27 | 14140.63 | 151671.11 | 40.78 | 363.93 |
| 64 | 0.15 | 307.19 | 14136.71 | 345223.92 | 40.79 | 364.12 |


## 📊 各并发级别性能柱状图

<img src="./concurrency_comparison.png" width="1000" />


## 📈 性能趋势分析

<img src="./performance_trends.png" width="1000" />

---

### 🎯 服务基准结果详情

| 指标 | 32 并发 | 64 并发 |
|------|----------- | -----------|
| 成功请求数 | 1000 | 1000 |
| 失败请求数 | 0 | 0 |
| 测试持续时间 (s) | 6508.83 | 6510.64 |
| 总输入 tokens | 90039000 | 90039000 |
| 总生成 tokens | 2000000 | 2000000 |
| **请求吞吐量 (req/s)** | 0.15 | 0.15 |
| **输出 token 吞吐量 (tok/s)** | 307.27 | 307.19 |
| 峰值输出 token 吞吐量 (tok/s) | 567.00 | 560.00 |
| 峰值并发请求数 | 33.00 | 65.00 |
| **总 token 吞吐量 (tok/s)** | 14140.63 | 14136.71 |


### ⏱️ 首Token延迟 (TTFT)

| 指标 | 32 并发 | 64 并发 |
|------|----------- | -----------|
| 平均 TTFT (ms) | 125806.83 | 326061.81 |
| 中位 TTFT (ms) | 120195.62 | 339003.73 |
| P95 TTFT (ms) | 151485.52 | 339316.35 |
| P99 TTFT (ms) | 151671.11 | 345223.92 |


### ⚡ 每Token生成时间 (TPOT)

| 指标 | 32 并发 | 64 并发 |
|------|----------- | -----------|
| 平均 TPOT (ms) | 40.35 | 40.37 |
| 中位 TPOT (ms) | 40.47 | 40.49 |
| P95 TPOT (ms) | 40.71 | 40.73 |
| P99 TPOT (ms) | 40.78 | 40.79 |


### 🔄 Token间延迟 (ITL)

| 指标 | 32 并发 | 64 并发 |
|------|----------- | -----------|
| 平均 ITL (ms) | 47.09 | 47.27 |
| 中位 ITL (ms) | 23.85 | 23.87 |
| P95 ITL (ms) | 255.32 | 255.51 |
| P99 ITL (ms) | 363.93 | 364.12 |

---

## 📝 分析总结

### 1. 吞吐量性能分析

**请求吞吐量 (QPS)**: 随着并发级别增加，QPS持续上升。
中并发(32)平均 QPS: 0.15 req/s；
高并发(64)平均 QPS: 0.15 req/s；
最高 QPS 出现在 32 并发，达到 0.15 req/s。

**Token总吞吐量**: 最高达到 14141 tok/s (32 并发)。

### 2. 首Token延迟 (TTFT) 分析

TTFT随并发增加显著上升。
高并发平均 P99 TTFT: 345224ms；
最高 P99 TTFT 出现在 64 并发，达到 345224ms。

### 3. Token生成时间 (TPOT) 分析

TPOT随并发增加也呈上升趋势。
高并发平均 P99 TPOT: 40.79ms；
最高 P99 TPOT 出现在 64 并发，达到 40.79ms。

### 4. Token间延迟 (ITL) 分析

ITL随并发增加呈上升趋势。
高并发平均 P99 ITL: 364.12ms；
最高 P99 ITL 出现在 64 并发，达到 364.12ms。

### 5. 综合评估

**吞吐量增长**: 从最低并发到最高并发，QPS增长了 0.0%。

---

<div align="center">
*报告生成时间: 2026-05-18*
</div>
