# 多模型性能对比报告

<div>

**测试日期：** 2026-04-09

**芯片平台：** hygon_bw1000

**测试套件：** test_03

**Run ID：** 01, 01

**并发级别：** 4并发

**测试配置：** 4-100-i194560-o1024

</div>

---

## 🤖 芯片和模型配置信息

| 芯片名称                        | **MiniMax-M2.5-W8A8** | **MiniMax-M2.5-bf16** |
|-----------------------------|-------------------------------|-------------------------------|
| **model_name** | MiniMax-M2.5-W8A8 | MiniMax-M2.5-bf16 |
| **quantization_config** | int-8 | bf16 |
| **model_size** | 215G | 427G |
| **max_position_embeddings** | 196608 | 196608 |
| **temperature** | N/A | N/A |
| **top_k** | N/A | N/A |
| **top_p** | N/A | N/A |
| **transformers_version** | 4.57.6 | 4.46.1 |
| **vllm_version** | 0.15.1+das.opt1.alpha.dtk2604 | 0.11.0+das.opt1.rc2.dtk2604.20260128.g0bf89b0c |
| **python_version** | 3.10.12 | 3.10.12 |

---

## 🤖 vLLM启动配置信息

| 参数名称                    | **MiniMax-M2.5-W8A8** | **MiniMax-M2.5-bf16** |
|-------------------------|-------------------|-------------------|
| model_name | MiniMax-M2.5-W8A8 | MiniMax-M2.5-bf16 |
| max-model-len | 196608 | 196608 |
| max-num-seqs | 64 | 64 |
| max-num-batched-tokens | default | default |
| gpu-memory-utilization | 0.9 | 0.98 |
| dtype | bfloat16 | bfloat16 |
| block_size | default | default |
| dp | 1 | 1 |
| tp | 8 | 8 |
| pp | 1 | 1 |
| enable-export-parallel | True | True |
| enable-auto-tool-choice | True | True |
| tool-call-parser | minimax_m2 | minimax_m2 |
| reasoning-parser | minimax_m2 (不生效) | minimax_m2 (不生效) |

---

## 📊 模型列表

| 模型名称 | Run ID | 状态 |
|----------|--------|------|
| MiniMax-M2.5-W8A8 | 01 | [OK] |
| MiniMax-M2.5-bf16 | 01 | [OK] |

---

## 📈 服务基准结果对比

| 指标 | MiniMax-M2.5-W8A8 | MiniMax-M2.5-bf16 |
|------|----------- | -----------|
| 成功请求数 | 100 | 100 |
| 失败请求数 | 0 | 0 |
| 测试持续时间 (s) | 5914.77 | 15215.04 |
| 总输入 tokens | 19456000 | 19456000 |
| 总生成 tokens | 102400 | 102400 |
| **请求吞吐量 (req/s)** | **0.02** ⭐ | 0.01 |
| **输出 token 吞吐量 (tok/s)** | **17.31** ⭐ | 6.73 |
| 峰值输出 token 吞吐量 (tok/s) | **96.00** ⭐ | 42.00 |
| 峰值并发请求数 | 7.00 | 5.00 |
| **总 token 吞吐量 (tok/s)** | **3306.71** ⭐ | 1285.46 |

---

## ⏱️ 首 Token 延迟 (TTFT) 对比

| 指标 | MiniMax-M2.5-W8A8 | MiniMax-M2.5-bf16 |
|------|----------- | -----------|
| 平均 TTFT (ms) | **110901.13** ⭐ | 554725.24 |
| 中位 TTFT (ms) | **96467.29** ⭐ | 566598.23 |
| P95 TTFT (ms) | **174373.75** ⭐ | 570672.93 |
| P99 TTFT (ms) | **178812.59** ⭐ | 571594.19 |

---

## ⚡ 每 Token 生成时间 (TPOT) 对比

| 指标 | MiniMax-M2.5-W8A8 | MiniMax-M2.5-bf16 |
|------|----------- | -----------|
| 平均 TPOT (ms) | 122.83 | **44.79** ⭐ |
| 中位 TPOT (ms) | 120.64 | **40.22** ⭐ |
| P95 TPOT (ms) | 184.15 | **45.51** ⭐ |
| P99 TPOT (ms) | 184.74 | **47.96** ⭐ |

---

## 🔄 Token 间延迟 (ITL) 对比

| 指标 | MiniMax-M2.5-W8A8 | MiniMax-M2.5-bf16 |
|------|----------- | -----------|
| 平均 ITL (ms) | 122.76 | **44.91** ⭐ |
| 中位 ITL (ms) | 43.16 | **29.44** ⭐ |
| P95 ITL (ms) | 52.02 | **32.20** ⭐ |
| P99 ITL (ms) | 3071.54 | **56.06** ⭐ |

---

## 📊 模型性能对比

![Model Performance Comparison](./concurrency4_comparison.png)

---

## 📝 分析小结

- **请求吞吐量**: MiniMax-M2.5-W8A8 最高，达 0.02 req/s
- **总token吞吐量**: MiniMax-M2.5-W8A8 最高，达 3307 tok/s
- **TTFT P99**: MiniMax-M2.5-W8A8 最优，为 178812.59ms
- **TPOT P99**: MiniMax-M2.5-bf16 最优，为 47.96ms

---

<div align="center">
*报告生成时间: 2026-04-09*
</div>
