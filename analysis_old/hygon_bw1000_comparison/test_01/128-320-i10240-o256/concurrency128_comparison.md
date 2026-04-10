# 多模型性能对比报告

<div>

**测试日期：** 2026-04-09

**芯片平台：** hygon_bw1000

**测试套件：** test_01

**Run ID：** 01, 01

**并发级别：** 128并发

**测试配置：** 128-320-i10240-o256

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
| 测试持续时间 (s) | 389.72 | 1972.54 |
| 总输入 tokens | 3276800 | 3276800 |
| 总生成 tokens | 81920 | 81920 |
| **请求吞吐量 (req/s)** | **0.82** ⭐ | 0.16 |
| **输出 token 吞吐量 (tok/s)** | **210.20** ⭐ | 41.53 |
| 峰值输出 token 吞吐量 (tok/s) | **1279.00** ⭐ | 253.00 |
| 峰值并发请求数 | 191.00 | 150.00 |
| **总 token 吞吐量 (tok/s)** | **8618.28** ⭐ | 1702.74 |

---

## ⏱️ 首 Token 延迟 (TTFT) 对比

| 指标 | MiniMax-M2.5-W8A8 | MiniMax-M2.5-bf16 |
|------|----------- | -----------|
| 平均 TTFT (ms) | **124081.37** ⭐ | 618139.35 |
| 中位 TTFT (ms) | **140502.36** ⭐ | 658558.23 |
| P95 TTFT (ms) | **140895.64** ⭐ | 794480.39 |
| P99 TTFT (ms) | **140903.29** ⭐ | 798530.33 |

---

## ⚡ 每 Token 生成时间 (TPOT) 对比

| 指标 | MiniMax-M2.5-W8A8 | MiniMax-M2.5-bf16 |
|------|----------- | -----------|
| 平均 TPOT (ms) | **64.17** ⭐ | 151.12 |
| 中位 TPOT (ms) | **61.13** ⭐ | 122.56 |
| P95 TPOT (ms) | **61.49** ⭐ | 513.83 |
| P99 TPOT (ms) | **300.89** ⭐ | 598.80 |

---

## 🔄 Token 间延迟 (ITL) 对比

| 指标 | MiniMax-M2.5-W8A8 | MiniMax-M2.5-bf16 |
|------|----------- | -----------|
| 平均 ITL (ms) | **63.92** ⭐ | 150.60 |
| 中位 ITL (ms) | **57.36** ⭐ | 103.19 |
| P95 ITL (ms) | **61.86** ⭐ | 108.58 |
| P99 ITL (ms) | **85.02** ⭐ | 126.85 |

---

## 📊 模型性能对比

![Model Performance Comparison](./concurrency128_comparison.png)

---

## 📝 分析小结

- **请求吞吐量**: MiniMax-M2.5-W8A8 最高，达 0.82 req/s
- **总token吞吐量**: MiniMax-M2.5-W8A8 最高，达 8618 tok/s
- **TTFT P99**: MiniMax-M2.5-W8A8 最优，为 140903.29ms
- **TPOT P99**: MiniMax-M2.5-W8A8 最优，为 300.89ms

---

<div align="center">
*报告生成时间: 2026-04-09*
</div>
