# 多模型性能对比报告

<div>

**测试日期：** 2026-04-09

**芯片平台：** hygon_bw1000

**测试套件：** test_03

**Run ID：** 01, 01

**并发级别：** 1并发

**测试配置：** 1-100-i194560-o1024

</div>

---

## 🤖 芯片和模型配置信息

| 芯片名称                        | **MiniMax-M2.5-W8A8**         | **MiniMax-M2.5-bf16**                          |
|-----------------------------|-------------------------------|------------------------------------------------|
| **model_name**              | MiniMax-M2.5-W8A8             | MiniMax-M2.5-bf16                              |
| **quantization_config**     | int-8                         | bf16                                           |
| **model_size**              | 215G                          | 427G                                           |
| **max_position_embeddings** | 196608                        | 196608                                         |
| **temperature**             | N/A                           | N/A                                            |
| **top_k**                   | N/A                           | N/A                                            |
| **top_p**                   | N/A                           | N/A                                            |
| **transformers_version**    | 4.57.6                        | 4.46.1                                         |
| **vllm_version**            | 0.15.1+das.opt1.alpha.dtk2604 | 0.11.0+das.opt1.rc2.dtk2604.20260128.g0bf89b0c |
| **python_version**          | 3.10.12                       | 3.10.12                                        |

---

## 🤖 vLLM启动配置信息

| 参数名称                    | **MiniMax-M2.5-W8A8** | **MiniMax-M2.5-bf16** |
|-------------------------|-----------------------|-----------------------|
| model_name              | MiniMax-M2.5-W8A8     | MiniMax-M2.5-bf16     |
| max-model-len           | 196608                | 196608                |
| max-num-seqs            | 64                    | 64                    |
| max-num-batched-tokens  | default               | default               |
| gpu-memory-utilization  | 0.9                   | 0.98                  |
| dtype                   | bfloat16              | bfloat16              |
| block_size              | default               | default               |
| dp                      | 1                     | 1                     |
| tp                      | 8                     | 8                     |
| pp                      | 1                     | 1                     |
| enable-export-parallel  | True                  | True                  |
| enable-auto-tool-choice | True                  | True                  |
| tool-call-parser        | minimax_m2            | minimax_m2            |
| reasoning-parser        | minimax_m2 (不生效)      | minimax_m2 (不生效)      |

---

## 📊 模型列表

| 模型名称              | Run ID | 状态   |
|-------------------|--------|------|
| MiniMax-M2.5-W8A8 | 01     | [OK] |
| MiniMax-M2.5-bf16 | 01     | [OK] |

---

## 📈 服务基准结果对比

| 指标                       | MiniMax-M2.5-W8A8 | MiniMax-M2.5-bf16 |
|--------------------------|-------------------|-------------------|
| 成功请求数                    | 100               | 100               |
| 失败请求数                    | 0                 | 0                 |
| 测试持续时间 (s)               | 6687.21           | 14207.97          |
| 总输入 tokens               | 19456000          | 19456000          |
| 总生成 tokens               | 102400            | 102400            |
| **请求吞吐量 (req/s)**        | **0.01** ⭐        | **0.01** ⭐        |
| **输出 token 吞吐量 (tok/s)** | **15.31** ⭐       | 7.21              |
| 峰值输出 token 吞吐量 (tok/s)   | **47.00** ⭐       | 39.00             |
| 峰值并发请求数                  | 2.00              | 2.00              |
| **总 token 吞吐量 (tok/s)**  | **2924.75** ⭐     | 1376.58           |

---

## ⏱️ 首 Token 延迟 (TTFT) 对比

| 指标 | MiniMax-M2.5-W8A8 | MiniMax-M2.5-bf16 |
|------|----------- | -----------|
| 平均 TTFT (ms) | **43487.80** ⭐ | 111977.70 |
| 中位 TTFT (ms) | **43910.38** ⭐ | 113467.99 |
| P95 TTFT (ms) | **44091.15** ⭐ | 113900.00 |
| P99 TTFT (ms) | **44121.71** ⭐ | 114142.41 |

---

## ⚡ 每 Token 生成时间 (TPOT) 对比

| 指标 | MiniMax-M2.5-W8A8 | MiniMax-M2.5-bf16 |
|------|----------- | -----------|
| 平均 TPOT (ms) | **22.86** ⭐ | 29.42 |
| 中位 TPOT (ms) | **22.86** ⭐ | 29.42 |
| P95 TPOT (ms) | **22.96** ⭐ | 29.55 |
| P99 TPOT (ms) | **22.97** ⭐ | 29.56 |

---

## 🔄 Token 间延迟 (ITL) 对比

| 指标 | MiniMax-M2.5-W8A8 | MiniMax-M2.5-bf16 |
|------|----------- | -----------|
| 平均 ITL (ms) | **22.88** ⭐ | 29.51 |
| 中位 ITL (ms) | **22.85** ⭐ | 29.42 |
| P95 ITL (ms) | **23.57** ⭐ | 32.11 |
| P99 ITL (ms) | **32.12** ⭐ | 53.91 |

---

## 📊 模型性能对比

![Model Performance Comparison](./concurrency1_comparison.png)

---

## 📝 分析小结

- **请求吞吐量**: MiniMax-M2.5-W8A8 最高，达 0.01 req/s
- **总token吞吐量**: MiniMax-M2.5-W8A8 最高，达 2925 tok/s
- **TTFT P99**: MiniMax-M2.5-W8A8 最优，为 44121.71ms
- **TPOT P99**: MiniMax-M2.5-W8A8 最优，为 22.97ms

---

<div align="center">
*报告生成时间: 2026-04-09*
</div>
