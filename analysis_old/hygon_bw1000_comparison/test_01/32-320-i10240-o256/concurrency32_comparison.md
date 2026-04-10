# 多模型性能对比报告

<div>

**测试日期：** 2026-04-09

**芯片平台：** hygon_bw1000

**测试套件：** test_01

**Run ID：** 01, 01

**并发级别：** 32并发

**测试配置：** 32-320-i10240-o256

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
| 成功请求数 | 320 | 320 |
| 失败请求数 | 0 | 0 |
| 测试持续时间 (s) | 418.62 | 1972.97 |
| 总输入 tokens | 3276800 | 3276800 |
| 总生成 tokens | 81920 | 81920 |
| **请求吞吐量 (req/s)** | **0.76** ⭐ | 0.16 |
| **输出 token 吞吐量 (tok/s)** | **195.69** ⭐ | 41.52 |
| 峰值输出 token 吞吐量 (tok/s) | **864.00** ⭐ | 253.00 |
| 峰值并发请求数 | 64.00 | 54.00 |
| **总 token 吞吐量 (tok/s)** | **8023.25** ⭐ | 1702.37 |

---

## ⏱️ 首 Token 延迟 (TTFT) 对比

| 指标 | MiniMax-M2.5-W8A8 | MiniMax-M2.5-bf16 |
|------|----------- | -----------|
| 平均 TTFT (ms) | **30731.82** ⭐ | 154015.23 |
| 中位 TTFT (ms) | **31855.15** ⭐ | 109747.80 |
| P95 TTFT (ms) | **31914.75** ⭐ | 245875.42 |
| P99 TTFT (ms) | **31917.86** ⭐ | 246016.06 |

---

## ⚡ 每 Token 生成时间 (TPOT) 对比

| 指标 | MiniMax-M2.5-W8A8 | MiniMax-M2.5-bf16 |
|------|----------- | -----------|
| 平均 TPOT (ms) | **43.62** ⭐ | 152.59 |
| 中位 TPOT (ms) | **39.56** ⭐ | 122.78 |
| P95 TPOT (ms) | **39.90** ⭐ | 514.14 |
| P99 TPOT (ms) | **160.15** ⭐ | 598.81 |

---

## 🔄 Token 间延迟 (ITL) 对比

| 指标 | MiniMax-M2.5-W8A8 | MiniMax-M2.5-bf16 |
|------|----------- | -----------|
| 平均 ITL (ms) | **43.45** ⭐ | 151.99 |
| 中位 ITL (ms) | **39.75** ⭐ | 103.29 |
| P95 ITL (ms) | **48.15** ⭐ | 111.32 |
| P99 ITL (ms) | **59.25** ⭐ | 139.96 |

---

## 📊 模型性能对比

![Model Performance Comparison](./concurrency32_comparison.png)

---

## 📝 分析小结

- **请求吞吐量**: MiniMax-M2.5-W8A8 最高，达 0.76 req/s
- **总token吞吐量**: MiniMax-M2.5-W8A8 最高，达 8023 tok/s
- **TTFT P99**: MiniMax-M2.5-W8A8 最优，为 31917.86ms
- **TPOT P99**: MiniMax-M2.5-W8A8 最优，为 160.15ms

---

<div align="center">
*报告生成时间: 2026-04-09*
</div>
