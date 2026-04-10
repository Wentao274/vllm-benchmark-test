# 多模型性能对比报告

<div>

**测试日期：** 2026-04-09

**芯片平台：** hygon_bw1000

**测试套件：** test_01

**Run ID：** 01, 01

**并发级别：** 2并发

**测试配置：** 2-320-i10240-o256

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
| 测试持续时间 (s) | 963.16 | 2718.90 |
| 总输入 tokens | 3276800 | 3276800 |
| 总生成 tokens | 81920 | 81920 |
| **请求吞吐量 (req/s)** | **0.33** ⭐ | 0.12 |
| **输出 token 吞吐量 (tok/s)** | **85.05** ⭐ | 30.13 |
| 峰值输出 token 吞吐量 (tok/s) | **136.00** ⭐ | 76.00 |
| 峰值并发请求数 | 4.00 | 4.00 |
| **总 token 吞吐量 (tok/s)** | **3487.19** ⭐ | 1235.32 |

---

## ⏱️ 首 Token 延迟 (TTFT) 对比

| 指标 | MiniMax-M2.5-W8A8 | MiniMax-M2.5-bf16 |
|------|----------- | -----------|
| 平均 TTFT (ms) | **1623.93** ⭐ | 7501.77 |
| 中位 TTFT (ms) | **1153.37** ⭐ | 5186.88 |
| P95 TTFT (ms) | **2156.14** ⭐ | 10045.05 |
| P99 TTFT (ms) | **2165.32** ⭐ | 10088.82 |

---

## ⚡ 每 Token 生成时间 (TPOT) 对比

| 指标 | MiniMax-M2.5-W8A8 | MiniMax-M2.5-bf16 |
|------|----------- | -----------|
| 平均 TPOT (ms) | **17.24** ⭐ | 37.22 |
| 中位 TPOT (ms) | **17.15** ⭐ | 37.31 |
| P95 TPOT (ms) | **19.37** ⭐ | 47.30 |
| P99 TPOT (ms) | **19.41** ⭐ | 47.55 |

---

## 🔄 Token 间延迟 (ITL) 对比

| 指标 | MiniMax-M2.5-W8A8 | MiniMax-M2.5-bf16 |
|------|----------- | -----------|
| 平均 ITL (ms) | **17.23** ⭐ | 37.12 |
| 中位 ITL (ms) | **15.20** ⭐ | 27.55 |
| P95 ITL (ms) | **16.13** ⭐ | 28.71 |
| P99 ITL (ms) | **24.07** ⭐ | 43.95 |

---

## 📊 模型性能对比

![Model Performance Comparison](./concurrency2_comparison.png)

---

## 📝 分析小结

- **请求吞吐量**: MiniMax-M2.5-W8A8 最高，达 0.33 req/s
- **总token吞吐量**: MiniMax-M2.5-W8A8 最高，达 3487 tok/s
- **TTFT P99**: MiniMax-M2.5-W8A8 最优，为 2165.32ms
- **TPOT P99**: MiniMax-M2.5-W8A8 最优，为 19.41ms

---

<div align="center">
*报告生成时间: 2026-04-09*
</div>
