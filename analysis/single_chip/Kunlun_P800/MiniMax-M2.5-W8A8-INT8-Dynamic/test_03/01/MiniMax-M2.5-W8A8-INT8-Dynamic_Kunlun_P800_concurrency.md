# MiniMax-M2.5-W8A8-INT8-Dynamic模型在Kunlun_P800上的Benchmark基准测试报告

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

| 参数名称                    | Kunlun_P800 |
|------------------------|-------------|
| **model_name** | MiniMax-M2.5-W8A8-INT8-Dynamic |
| **quantization_config** | int-8 |
| **model_size** | 215G |
| **max_position_embeddings** | 196608 |
| **temperature** | 1.0 |
| **top_k** | 40 |
| **top_p** | 0.95 |
| **transformers_version** | 4.46.1 |
| **vllm_version** | 0.11.0 |
| **python_version** | 3.10.15 |


## 🤖 vLLM启动配置信息

| 参数名称                   | Kunlun_P800 |
|------------------------|-------------|
| **Model Name** | MiniMax-M2.5-W8A8-INT8-Dynamic |
| **Max Model Len** | 196608 |
| **Max Num Seqs** | 64 |
| **Max Num Batched Tokens** | 8192 |
| **Gpu Memory Utilization** | 0.95 |
| **Dtype** | auto |
| **Block Size** | 128 |
| **Dp** | 1 |
| **Tp** | 8 |
| **Pp** | 1 |
| **Enable Export Parallel** | False |
| **Enable Auto Tool Choice** | True |
| **Tool Call Parser** | minimax_m2 |
| **Reasoning Parser** | minimax_m2 (不生效) |
| **Compilation Config** | {"splitting_ops":["vllm.unified_attention","vllm.unified_attention_with_output","vllm.unified_attention_with_output_kunlun","vllm.mamba_mixer2","vllm.mamba_mixer","vllm.short_conv","vllm.linear_attention","vllm.plamo2_mamba_mixer","vllm.gdn_attention","vllm.sparse_attn_indexer","vllm.sparse_attn_indexer_vllm_kunlun"]} |

- **Kunlun_P800**: 昆仑芯不启用专家并行避免通信问题


## 📊 测试概览

| 项目            | 配置                                     | 备注  |
|---------------|----------------------------------------|-----|
| **数据集**       | random                                 |     |
| **并发数**       | 32, 64    |     |
| **总请求数**      | 1000                                    |     |
| **请求输入上下文长度** | 90000（87k）                             |     |
| **请求输出上下文长度** | 2000（1k）                             |     |
| **模型**        | MiniMax-M2.5-W8A8-INT8-Dynamic                           |     |
| **被测芯片**      | Kunlun_P800 |     |

---

## 📋 测试结果汇总

| 并发数 | 请求吞吐量 (req/s) | 输出Token吞吐量 (tok/s) | 总Token吞吐量 (tok/s) | TTFT P99 (ms) | TPOT P99 (ms) | ITL P99 (ms) |
| ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
| 32 | 0.08 | 20.03 | 7060.97 | 320308.58 | 1136.82 | 1612.65 |
| 64 | 0.08 | 20.70 | 7096.07 | 724153.37 | 1135.81 | 1612.99 |


## 📊 各并发级别性能柱状图

<img src="./concurrency_comparison.png" width="1000" />


## 📈 性能趋势分析

<img src="./performance_trends.png" width="1000" />

---

### 🎯 服务基准结果详情

| 指标 | 32 并发 | 64 并发 |
|------|----------- | -----------|
| 成功请求数 | 512 | 512 |
| 失败请求数 | 0 | 0 |
| 测试持续时间 (s) | 6544.58 | 6512.73 |
| 总输入 tokens | 46080000 | 46080000 |
| 总生成 tokens | 131114 | 134787 |
| **请求吞吐量 (req/s)** | 0.08 | 0.08 |
| **输出 token 吞吐量 (tok/s)** | 20.03 | 20.70 |
| 峰值输出 token 吞吐量 (tok/s) | 240.00 | 253.00 |
| 峰值并发请求数 | 34.00 | 66.00 |
| **总 token 吞吐量 (tok/s)** | 7060.97 | 7096.07 |


### ⏱️ 首Token延迟 (TTFT)

| 指标 | 32 并发 | 64 并发 |
|------|----------- | -----------|
| 平均 TTFT (ms) | 152142.37 | 529931.46 |
| 中位 TTFT (ms) | 144761.98 | 543948.14 |
| P95 TTFT (ms) | 206370.88 | 601842.13 |
| P99 TTFT (ms) | 320308.58 | 724153.37 |


### ⚡ 每Token生成时间 (TPOT)

| 指标 | 32 并发 | 64 并发 |
|------|----------- | -----------|
| 平均 TPOT (ms) | 1017.97 | 985.06 |
| 中位 TPOT (ms) | 1058.57 | 1006.79 |
| P95 TPOT (ms) | 1132.74 | 1131.47 |
| P99 TPOT (ms) | 1136.82 | 1135.81 |


### 🔄 Token间延迟 (ITL)

| 指标 | 32 并发 | 64 并发 |
|------|----------- | -----------|
| 平均 ITL (ms) | 979.89 | 973.01 |
| 中位 ITL (ms) | 1045.99 | 1042.70 |
| P95 ITL (ms) | 1565.66 | 1561.54 |
| P99 ITL (ms) | 1612.65 | 1612.99 |

---

## 📝 分析总结

### 1. 吞吐量性能分析

**请求吞吐量 (QPS)**: 随着并发级别增加，QPS持续上升。
中并发(32)平均 QPS: 0.08 req/s；
高并发(64)平均 QPS: 0.08 req/s；
最高 QPS 出现在 32 并发，达到 0.08 req/s。

**Token总吞吐量**: 最高达到 7096 tok/s (64 并发)。

### 2. 首Token延迟 (TTFT) 分析

TTFT随并发增加显著上升。
高并发平均 P99 TTFT: 724153ms；
最高 P99 TTFT 出现在 64 并发，达到 724153ms。

### 3. Token生成时间 (TPOT) 分析

TPOT随并发增加也呈上升趋势。
高并发平均 P99 TPOT: 1135.81ms；
最高 P99 TPOT 出现在 32 并发，达到 1136.82ms。

### 4. Token间延迟 (ITL) 分析

ITL随并发增加呈上升趋势。
高并发平均 P99 ITL: 1612.99ms；
最高 P99 ITL 出现在 64 并发，达到 1612.99ms。

### 5. 综合评估

**吞吐量增长**: 从最低并发到最高并发，QPS增长了 0.0%。

---

<div align="center">
*报告生成时间: 2026-05-18*
</div>
