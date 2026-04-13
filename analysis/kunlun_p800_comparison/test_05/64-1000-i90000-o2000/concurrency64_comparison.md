# 多模型性能对比报告

<div>

**测试日期：** 2026-04-13

**芯片平台：** kunlun_p800

**测试套件：** test_05

**Run ID：** 01, 01

**并发级别：** 64并发

**测试配置：** 64-1000-i90000-o2000

</div>

---

## 🤖 芯片和模型配置信息

| 芯片名称                        | **MiniMax-M2.5-W8A8-INT8-Dynamic** | **Qwen3.5-122B-A10B** |
|-----------------------------|-------------------------------|-------------------------------|
| **model_name** | MiniMax-M2.5-W8A8-INT8-Dynamic | Qwen3.5-122B-A10B |
| **quantization_config** | int-8 | N/A |
| **model_size** | 215G | 234G |
| **max_position_embeddings** | 196608 | 262144 |
| **temperature** | 1.0 | 0.6 |
| **top_k** | 40 | 20 |
| **top_p** | 0.95 | 0.95 |
| **transformers_version** | 4.46.1 | 4.57.0.dev0 |
| **vllm_version** | 0.11.0 | 0.15.1 |
| **python_version** | 3.10.15 | 3.10.19 |

---

## 🤖 vLLM启动配置信息

| 参数名称                    | **MiniMax-M2.5-W8A8-INT8-Dynamic** | **Qwen3.5-122B-A10B** |
|-------------------------|-------------------|-------------------|
| model_name | MiniMax-M2.5-W8A8-INT8-Dynamic | MiniMax-M2.5-W8A8-INT8-Dynamic |
| max-model-len | 196608 | 196608 |
| max-num-seqs | 64 | 64 |
| max-num-batched-tokens | 8192 | 8192 |
| gpu-memory-utilization | 0.95 | 0.95 |
| dtype | auto | auto |
| block_size | 128 | 128 |
| dp | 1 | 1 |
| tp | 8 | 8 |
| pp | 1 | 1 |
| enable-export-parallel | False | False |
| enable-auto-tool-choice | True | True |
| tool-call-parser | minimax_m2 | minimax_m2 |
| reasoning-parser | minimax_m2 (不生效) | minimax_m2 (不生效) |

---

## 📊 模型列表

| 模型名称 | Run ID | 状态 |
|----------|--------|------|
| MiniMax-M2.5-W8A8-INT8-Dynamic | 01 | [OK] |
| Qwen3.5-122B-A10B | 01 | [OK] |

---

## 📈 服务基准结果对比

| 指标 | MiniMax-M2.5-W8A8-INT8-Dynamic (基准) | Qwen3.5-122B-A10B | 差异 | % |
|------|--------------- | --------- | ------- | -------|
| 成功请求数 | 1000 | 1000 | 0.00 | 0.0% |
| 失败请求数 |  | 0 | N/A | N/A |
| 测试持续时间 (s) | 11739.56 | 20561.76 | +8822.20 | +75.1% |
| 总输入 tokens | 90000000 | 90000000 | 0.00 | 0.0% |
| 总生成 tokens | 263983 | 2000000 | +1736017.00 | +657.6% |
| **请求吞吐量 (req/s)** | 0.09 | 0.05 | -0.04 | -44.4% |
| **输出 token 吞吐量 (tok/s)** | 22.49 | 97.27 | +74.78 | +332.5% |
| 峰值输出 token 吞吐量 (tok/s) | 345.00 | 669.00 | +324.00 | +93.9% |
| 峰值并发请求数 | 67.00 | 65.00 | -2.00 | -3.0% |
| **总 token 吞吐量 (tok/s)** | 7688.87 | 4474.32 | -3214.55 | -41.8% |

---

## ⏱️ 首 Token 延迟 (TTFT) 对比

| 指标 | MiniMax-M2.5-W8A8-INT8-Dynamic (基准) | Qwen3.5-122B-A10B | 差异 | % |
|------|--------------- | --------- | ------- | -------|
| 平均 TTFT (ms) | 496041.50 | 291099.39 | -204942.11 | -41.3% |
| 中位 TTFT (ms) | 501925.88 | 115341.64 | -386584.24 | -77.0% |
| P95 TTFT (ms) | 540083.81 | 790996.18 | +250912.37 | +46.5% |
| P99 TTFT (ms) | 610915.79 | 1069305.63 | +458389.84 | +75.0% |

---

## ⚡ 每 Token 生成时间 (TPOT) 对比

| 指标 | MiniMax-M2.5-W8A8-INT8-Dynamic (基准) | Qwen3.5-122B-A10B | 差异 | % |
|------|--------------- | --------- | ------- | -------|
| 平均 TPOT (ms) | 927.19 | 510.58 | -416.61 | -44.9% |
| 中位 TPOT (ms) | 944.56 | 525.06 | -419.50 | -44.4% |
| P95 TPOT (ms) | 1048.60 | 528.32 | -520.28 | -49.6% |
| P99 TPOT (ms) | 1052.90 | 627.30 | -425.60 | -40.4% |

---

## 🔄 Token 间延迟 (ITL) 对比

| 指标 | MiniMax-M2.5-W8A8-INT8-Dynamic (基准) | Qwen3.5-122B-A10B | 差异 | % |
|------|--------------- | --------- | ------- | -------|
| 平均 ITL (ms) | 914.26 | 510.53 | -403.73 | -44.2% |
| 中位 ITL (ms) | 969.10 | 604.64 | -364.46 | -37.6% |
| P95 ITL (ms) | 1476.89 | 776.69 | -700.20 | -47.4% |
| P99 ITL (ms) | 1524.21 | 804.10 | -720.11 | -47.2% |

---

## 📊 模型性能对比

![Model Performance Comparison](./concurrency64_comparison.png)

---

## 📝 分析小结

- **请求吞吐量**: MiniMax-M2.5-W8A8-INT8-Dynamic 最高，达 0.09 req/s
- **总token吞吐量**: MiniMax-M2.5-W8A8-INT8-Dynamic 最高，达 7689 tok/s
- **TTFT P99**: MiniMax-M2.5-W8A8-INT8-Dynamic 最优，为 610915.79ms
- **TPOT P99**: Qwen3.5-122B-A10B 最优，为 627.30ms

---

<div align="center">
*报告生成时间: 2026-04-13*
</div>
