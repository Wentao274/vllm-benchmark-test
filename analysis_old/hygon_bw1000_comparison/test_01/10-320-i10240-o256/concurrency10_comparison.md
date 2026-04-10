# 多模型性能对比报告

<div>

**测试日期：** 2026-04-09

**芯片平台：** hygon_bw1000

**测试套件：** test_01

**Run ID：** 01, 01

**并发级别：** 10并发

**测试配置：** 10-320-i10240-o256

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
| 测试持续时间 (s) | 526.27 | 2347.19 |
| 总输入 tokens | 3276800 | 3276800 |
| 总生成 tokens | 81920 | 81920 |
| **请求吞吐量 (req/s)** | **0.61** ⭐ | 0.14 |
| **输出 token 吞吐量 (tok/s)** | **155.66** ⭐ | 34.90 |
| 峰值输出 token 吞吐量 (tok/s) | **420.00** ⭐ | 119.00 |
| 峰值并发请求数 | 20.00 | 20.00 |
| **总 token 吞吐量 (tok/s)** | **6382.12** ⭐ | 1430.96 |

---

## ⏱️ 首 Token 延迟 (TTFT) 对比

| 指标 | MiniMax-M2.5-W8A8 | MiniMax-M2.5-bf16 |
|------|----------- | -----------|
| 平均 TTFT (ms) | **9132.16** ⭐ | 44636.12 |
| 中位 TTFT (ms) | **10065.22** ⭐ | 49274.29 |
| P95 TTFT (ms) | **10076.09** ⭐ | 49373.65 |
| P99 TTFT (ms) | **10081.63** ⭐ | 49390.74 |

---

## ⚡ 每 Token 生成时间 (TPOT) 对比

| 指标 | MiniMax-M2.5-W8A8 | MiniMax-M2.5-bf16 |
|------|----------- | -----------|
| 平均 TPOT (ms) | **28.67** ⭐ | 112.59 |
| 中位 TPOT (ms) | **25.19** ⭐ | 95.29 |
| P95 TPOT (ms) | **60.21** ⭐ | 268.35 |
| P99 TPOT (ms) | **60.61** ⭐ | 269.35 |

---

## 🔄 Token 间延迟 (ITL) 对比

| 指标 | MiniMax-M2.5-W8A8 | MiniMax-M2.5-bf16 |
|------|----------- | -----------|
| 平均 ITL (ms) | **28.65** ⭐ | 112.15 |
| 中位 ITL (ms) | **25.20** ⭐ | 95.08 |
| P95 ITL (ms) | **26.22** ⭐ | 98.28 |
| P99 ITL (ms) | **41.33** ⭐ | 106.62 |

---

## 📊 模型性能对比

![Model Performance Comparison](./concurrency10_comparison.png)

---

## 📝 分析小结

- **请求吞吐量**: MiniMax-M2.5-W8A8 最高，达 0.61 req/s
- **总token吞吐量**: MiniMax-M2.5-W8A8 最高，达 6382 tok/s
- **TTFT P99**: MiniMax-M2.5-W8A8 最优，为 10081.63ms
- **TPOT P99**: MiniMax-M2.5-W8A8 最优，为 60.61ms

---

<div align="center">
*报告生成时间: 2026-04-09*
</div>
