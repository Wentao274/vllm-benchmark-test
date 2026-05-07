# 多模型性能对比报告

<div>

**测试日期：** 2026-05-07

**芯片平台：** hygon_bw1000

**测试套件：** test_01

**Run ID：** 01, 01

**并发级别：** 1并发

**测试配置：** 1-320-i10240-o256

</div>

---

## 🤖 芯片和模型配置信息

| 芯片名称                        | **MiniMax-M2.5-bf16** | **MiniMax-M2.5-W8A8** |
|-----------------------------|-------------------------------|-------------------------------|
| **model_name** | MiniMax-M2.5-bf16 | MiniMax-M2.5-W8A8 |
| **quantization_config** | bf16 | int-8 |
| **model_size** | 427G | 215G |
| **max_position_embeddings** | 196608 | 196608 |
| **temperature** | N/A | N/A |
| **top_k** | N/A | N/A |
| **top_p** | N/A | N/A |
| **transformers_version** | 4.46.1 | 4.57.6 |
| **vllm_version** | 0.11.0+das.opt1.rc2.dtk2604.20260128.g0bf89b0c | 0.15.1+das.opt1.alpha.dtk2604 |
| **python_version** | 3.10.12 | 3.10.12 |

---

## 🤖 vLLM启动配置信息

| 参数名称                    | **MiniMax-M2.5-bf16** | **MiniMax-M2.5-W8A8** |
|-------------------------|-------------------|-------------------|
| model_name | MiniMax-M2.5-bf16 | MiniMax-M2.5-W8A8 |
| max-model-len | 196608 | 196608 |
| max-num-seqs | 64 | 64 |
| max-num-batched-tokens | default | default |
| gpu-memory-utilization | 0.98 | 0.9 |
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
| MiniMax-M2.5-bf16 | 01 | [OK] |
| MiniMax-M2.5-W8A8 | 01 | [OK] |

---

## 📈 服务基准结果对比

| 指标 | MiniMax-M2.5-bf16 (基准) | MiniMax-M2.5-W8A8 | 差异 | % |
|------|--------------- | --------- | ------- | -------|
| 成功请求数 | 320 | 320 | 0.00 | 0.0% |
| 失败请求数 | 0 | 0 | 0.00 | 0.0% |
| 测试持续时间 (s) | 3322.61 | 1509.26 | -1813.35 | -54.6% |
| 总输入 tokens | 3276800 | 3276800 | 0.00 | 0.0% |
| 总生成 tokens | 81920 | 81920 | 0.00 | 0.0% |
| **请求吞吐量 (req/s)** | 0.10 | 0.21 | +0.11 | +110.0% |
| **输出 token 吞吐量 (tok/s)** | 24.66 | 54.28 | +29.62 | +120.1% |
| 峰值输出 token 吞吐量 (tok/s) | 50.00 | 72.00 | +22.00 | +44.0% |
| 峰值并发请求数 | 2.00 | 2.00 | 0.00 | 0.0% |
| **总 token 吞吐量 (tok/s)** | 1010.87 | 2225.40 | +1214.53 | +120.1% |

---

## ⏱️ 首 Token 延迟 (TTFT) 对比

| 指标 | MiniMax-M2.5-bf16 (基准) | MiniMax-M2.5-W8A8 | 差异 | % |
|------|--------------- | --------- | ------- | -------|
| 平均 TTFT (ms) | 5026.61 | 1112.21 | -3914.40 | -77.9% |
| 中位 TTFT (ms) | 5041.22 | 1111.97 | -3929.25 | -77.9% |
| P95 TTFT (ms) | 5072.83 | 1127.95 | -3944.88 | -77.8% |
| P99 TTFT (ms) | 5130.17 | 1168.49 | -3961.68 | -77.2% |

---

## ⚡ 每 Token 生成时间 (TPOT) 对比

| 指标 | MiniMax-M2.5-bf16 (基准) | MiniMax-M2.5-W8A8 | 差异 | % |
|------|--------------- | --------- | ------- | -------|
| 平均 TPOT (ms) | 21.00 | 14.13 | -6.87 | -32.7% |
| 中位 TPOT (ms) | 21.01 | 14.13 | -6.88 | -32.7% |
| P95 TPOT (ms) | 21.05 | 14.15 | -6.90 | -32.8% |
| P99 TPOT (ms) | 21.07 | 14.16 | -6.91 | -32.8% |

---

## 🔄 Token 间延迟 (ITL) 对比

| 指标 | MiniMax-M2.5-bf16 (基准) | MiniMax-M2.5-W8A8 | 差异 | % |
|------|--------------- | --------- | ------- | -------|
| 平均 ITL (ms) | 20.97 | 14.14 | -6.83 | -32.6% |
| 中位 ITL (ms) | 21.00 | 14.13 | -6.87 | -32.7% |
| P95 ITL (ms) | 21.70 | 14.47 | -7.23 | -33.3% |
| P99 ITL (ms) | 32.03 | 20.17 | -11.86 | -37.0% |

---

## 📊 模型性能对比

![Model Performance Comparison](./concurrency1_comparison.png)

---

## 📝 分析小结

- **请求吞吐量**: MiniMax-M2.5-W8A8 最高，达 0.21 req/s
- **总token吞吐量**: MiniMax-M2.5-W8A8 最高，达 2225 tok/s
- **TTFT P99**: MiniMax-M2.5-W8A8 最优，为 1168.49ms
- **TPOT P99**: MiniMax-M2.5-W8A8 最优，为 14.16ms

---

<div align="center">
*报告生成时间: 2026-05-07*
</div>
