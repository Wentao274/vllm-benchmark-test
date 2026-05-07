# 多模型性能对比报告

<div>

**测试日期：** 2026-05-07

**芯片平台：** hygon_bw1000

**测试套件：** test_01

**Run ID：** 01, 01

**并发级别：** 4并发

**测试配置：** 4-320-i10240-o256

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
| 测试持续时间 (s) | 2365.34 | 672.34 | -1693.00 | -71.6% |
| 总输入 tokens | 3276800 | 3276800 | 0.00 | 0.0% |
| 总生成 tokens | 81920 | 81920 | 0.00 | 0.0% |
| **请求吞吐量 (req/s)** | 0.14 | 0.48 | +0.34 | +242.9% |
| **输出 token 吞吐量 (tok/s)** | 34.63 | 121.84 | +87.21 | +251.8% |
| 峰值输出 token 吞吐量 (tok/s) | 111.00 | 247.00 | +136.00 | +122.5% |
| 峰值并发请求数 | 8.00 | 8.00 | 0.00 | 0.0% |
| **总 token 吞吐量 (tok/s)** | 1419.97 | 4995.56 | +3575.59 | +251.8% |

---

## ⏱️ 首 Token 延迟 (TTFT) 对比

| 指标 | MiniMax-M2.5-bf16 (基准) | MiniMax-M2.5-W8A8 | 差异 | % |
|------|--------------- | --------- | ------- | -------|
| 平均 TTFT (ms) | 16214.96 | 3351.54 | -12863.42 | -79.3% |
| 中位 TTFT (ms) | 19806.56 | 4115.92 | -15690.64 | -79.2% |
| P95 TTFT (ms) | 19874.02 | 4129.04 | -15744.98 | -79.2% |
| P99 TTFT (ms) | 19888.60 | 4137.09 | -15751.51 | -79.2% |

---

## ⚡ 每 Token 生成时间 (TPOT) 对比

| 指标 | MiniMax-M2.5-bf16 (基准) | MiniMax-M2.5-W8A8 | 差异 | % |
|------|--------------- | --------- | ------- | -------|
| 平均 TPOT (ms) | 52.36 | 19.81 | -32.55 | -62.2% |
| 中位 TPOT (ms) | 38.58 | 16.93 | -21.65 | -56.1% |
| P95 TPOT (ms) | 96.58 | 28.78 | -67.80 | -70.2% |
| P99 TPOT (ms) | 97.18 | 28.88 | -68.30 | -70.3% |

---

## 🔄 Token 间延迟 (ITL) 对比

| 指标 | MiniMax-M2.5-bf16 (基准) | MiniMax-M2.5-W8A8 | 差异 | % |
|------|--------------- | --------- | ------- | -------|
| 平均 ITL (ms) | 52.22 | 19.76 | -32.46 | -62.2% |
| 中位 ITL (ms) | 38.47 | 16.86 | -21.61 | -56.2% |
| P95 ITL (ms) | 43.58 | 17.85 | -25.73 | -59.0% |
| P99 ITL (ms) | 64.02 | 22.80 | -41.22 | -64.4% |

---

## 📊 模型性能对比

![Model Performance Comparison](./concurrency4_comparison.png)

---

## 📝 分析小结

- **请求吞吐量**: MiniMax-M2.5-W8A8 最高，达 0.48 req/s
- **总token吞吐量**: MiniMax-M2.5-W8A8 最高，达 4996 tok/s
- **TTFT P99**: MiniMax-M2.5-W8A8 最优，为 4137.09ms
- **TPOT P99**: MiniMax-M2.5-W8A8 最优，为 28.88ms

---

<div align="center">
*报告生成时间: 2026-05-07*
</div>
