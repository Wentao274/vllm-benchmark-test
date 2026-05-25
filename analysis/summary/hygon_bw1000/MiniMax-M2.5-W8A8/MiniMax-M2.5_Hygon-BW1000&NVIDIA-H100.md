# 海光BW1000、英伟达H100 - 单节点MiniMax-M2.5模型整体测试比对报告

<div align="center">
*测试日期：2026-04-04 ~ 2026-05-07 <br>
*测试人员：九章云极

</div>

---

## 1. 测试背景
公司需要在多个候选开源大模型中选型，部署基于vLLM或SGLang的推理服务。并需要在满足各项模型服务指标的情况下，选定芯片集采厂商。

## 2. 测试目标
本测试主要评估不同芯片在单机环境下运行大模型推理的能力，为后续集群采购和生产部署提供决策依据。
1. **硬件摸底**：确认各芯片型号实际规格（算力、显存、带宽）与标称值的一致性
2. **功能验证**：各模型在各芯片环境上的推理正确性和算子兼容性
3. **性能基准**：吞吐量、延迟、显存效率等关键指标
4. **单机极限**：8 卡 Tensor Parallel 的性能上限和资源利用率
5. **稳定性验证**：长时间运行下的可靠性
6. **K8S 容器化验证**：单节点 K8S 环境下GPU/DCU/XPU 调度、资源管理和服务编排能力


## 3. 测试环境

### 3.1 硬件规格

| 组件 \ 规格            | 英伟达                                        | 海光                                  | 状态     |
|--------------------|--------------------------------------------|-------------------------------------|--------|
| **节点数量**           | 1 台                                        | 1 台                                 | 确认     |
| **芯片型号**           | H100                                       | BW1000                              | 确认     |
| **芯片数量**           | 8 张                                        | 8 张                                 | 确认     |
| **单卡算力 FP16/BF16** | 1979 TFLOPS （官方理论值）                        | 待确认                                 | ⚠️ 待确认 |
| **单卡算力 FP32**      | 67 TFLOPS （官方理论值）                          | 待确认                                 | ⚠️ 待确认 |
| **单卡算力 FP64**      | 34 TFLOPS （官方理论值）                          | 待确认                                 | ⚠️ 待确认 |
| **单卡显存**           | 80GB                                       | 64GB                                | 确认     |
| **显存类型**           | HBM3                                       | 待确认                                 | ⚠️ 待确认 |
| **显存带宽**           | 3.35 TB/s                                  | 待确认                                 | ⚠️ 待确认 |
| **单卡功耗**           | 700 W                                      | 200 W                               | 确认     |
| **卡间互联**           | NVLink 4.0                                 | HSM                                 | 确认     |
| **CPU**            | Intel(R) Xeon(R) Platinum 8468 (192核)      | Hygon C86 (128核)                    | 确认     |
| **系统内存**           | 2.0 TiB                                    | 503 GiB                             | 确认     |
| **本地存储**           | 894GB 系统盘 + 7TB*4 缓存盘 + 7TB 容器盘 + 25TB 扩展盘 | 437G系统盘 + 1.7TiB (G73M1T9R-C-GD308) | 确认     |


### 3.2 软件栈

| 组件\版本             | 英伟达                   | 海光                              | 说明                |
|-------------------|-----------------------|---------------------------------|-------------------|
| **操作系统**          | Ubuntu 22.04.5 LTS    | Ubuntu 22.04.5 LTS              | 芯片所在物理机系统         |
| **显卡驱动**          | 570.133.20/580.126.09 | 6.3.22-V1.2.0                   | 驱动信息              |
| **Toolkit**       | release 12.9          | DTK-26.04-beta-0130-ubuntu20.04 | CUDA Toolkit版本    |
| **Docker**        | -                     | 28.0.4                          | 容器运行时             |
| **containerd**    | 2.2.0                 | 2.1.1                           | K8S 容器运行时（CRI）    |
| **Kubernetes**    | 1.34.2                | 1.33                            | 单节点 All-in-One 部署 |
| **Device Plugin** | 0.14.5                | v2.4.0                          | K8S GPU 资源管理      |
| **多卡通信库**         | NCCL                  | DTK内置RCCL                       | 多卡通信库             |


### 3.3 模型配置信息

| 参数名称                        | **NVIDIA_H100** | **Hygon_BW1000**              |
|-----------------------------|-----------------|-------------------------------|
| **model_name**              | MiniMax-M2.5    | MiniMax-M2.5-W8A8             |
| **quantization_config**     | FP8             | int-8                         |
| **model_size**              | 215G            | 215G                          |
| **max_position_embeddings** | 196608          | 196608                        |
| **temperature**             | 1.0             | N/A                           |
| **top_k**                   | 40              | N/A                           |
| **top_p**                   | 0.95            | N/A                           |
| **transformers_version**    | 4.46.1          | 4.57.6                        |
| **vllm_version**            | 0.20.0          | 0.15.1+das.opt1.alpha.dtk2604 |
| **python_version**          | 3.12.3          | 3.10.12                       |


### 3.4 推理框架主要启动参数

| 参数名称                        | **NVIDIA_H100** | **Hygon_BW1000**  |
|-----------------------------|-----------------|-------------------|
| **Model Name**              | MiniMax-M2.5    | MiniMax-M2.5-W8A8 |
| **Max Model Len**           | 196608          | 196608            |
| **Max Num Seqs**            | 64              | 64                |
| **Max Num Batched Tokens**  | 8192            | default           |
| **Block Size**              | default         | default           |
| **Gpu Memory Utilization**  | 0.85            | 0.9               |
| **Compilation Config**      | N/A             | N/A               |
| **Dtype**                   | default         | bfloat16          |
| **Dp**                      | 1               | 1                 |
| **Tp**                      | 8               | 8                 |
| **Pp**                      | 1               | 1                 |
| **Reasoning Parser**        | minimax_m2      | minimax_m2        |
| **Tool Call Parser**        | minimax_m2      | minimax_m2        |
| **Enable Auto Tool Choice** | True            | True              |
| **Enable Export Parallel**  | True            | /                 |


## 4. 测试场景及概况

### 4.1 测试场景列表
| 序号  | 测试场景               |
|-----|--------------------|
| 场景一 | vllm benchmark基准测试 |
| 场景二 | 单、多并发超长上下文请求       |
| 场景三 | 多并发长上下文极限验证        |
| 场景四 | 多I/O测试             |
| 场景五 | 模型精度测试             |
| 场景六 | 多轮对话测试             |
| 场景七 | 模型推理功能测试           |


### 4.2 模型部署问题汇总

N/A

### 4.3 模型推理测试问题汇总

N/A

---

---
以下是每个测试场景的详细结果报告
---

---

>Hygon_BW1000平台，Minimax-M2.5模型benchmark测试脚本见本报告 **《附录一》**

## 测试场景一：vllm benchmark基准测试
**测试目标**：在相同请求数、基础长度上下文参数下，使用vllm bench serve工具对并发数逐级增加场景的性能基准验证.

**主要采集指标**：

| 指标              | 单位         | 含义                                 |
|-----------------|------------|------------------------------------|
| TTFT            | ms         | Time To First Token，首 token 延迟     |
| TPOT            | ms/token   | Time Per Output Token，每 token 生成时间 |
| Throughput      | tokens/s   | 系统总吞吐                              |
| QPS             | requests/s | 请求吞吐                               |
| P95/P99 Latency | ms         | 延迟分位数                              |


### 📊 测试概览

| 项目            | 配置                                                           | 备注  |
|---------------|--------------------------------------------------------------|-----|
| **数据集**       | random                                                       |     |
| **并发数**       | 1, 2, 4, 8, 10, 16, 32, 64, 80, 128                          |     |
| **总请求数**      | 320                                                          |     |
| **请求输入上下文长度** | 10240（10k）                                                   |     |
| **请求输出上下文长度** | 256（0.25k）                                                   |     |
| **被测芯片**      | NVIDIA_H100, Hygon_BW1000                                    |     |
| **被测模型**      | NVIDIA_H100 (MiniMax-M2.5);<br/>Hygon_BW1000 (MiniMax-M2.5-W8A8) |     |


### 📊 芯片性能对比柱状图

**1并发**

<img src="./chip_comparison_c1_test_01_nvidia_h100_vs_hygon_bw1000.png" width="1000" />

**2并发**

<img src="./chip_comparison_c2_test_01_nvidia_h100_vs_hygon_bw1000.png" width="1000" />

**4并发**

<img src="./chip_comparison_c4_test_01_nvidia_h100_vs_hygon_bw1000.png" width="1000" />

**8并发**

<img src="./chip_comparison_c8_test_01_nvidia_h100_vs_hygon_bw1000.png" width="1000" />

**10并发**

<img src="./chip_comparison_c10_test_01_nvidia_h100_vs_hygon_bw1000.png" width="1000" />

**16并发**

<img src="./chip_comparison_c16_test_01_nvidia_h100_vs_hygon_bw1000.png" width="1000" />

**32并发**

<img src="./chip_comparison_c32_test_01_nvidia_h100_vs_hygon_bw1000.png" width="1000" />

**64并发**

<img src="./chip_comparison_c64_test_01_nvidia_h100_vs_hygon_bw1000.png" width="1000" />

**80并发**

<img src="./chip_comparison_c80_test_01_nvidia_h100_vs_hygon_bw1000.png" width="1000" />

**128并发**

<img src="./chip_comparison_c128_test_01_nvidia_h100_vs_hygon_bw1000.png" width="1000" />


### 📈 性能趋势对比图 (所有芯片)

<img src="./performance_trends_test_01_nvidia_h100_vs_hygon_bw1000.png" width="1000" />

---

### 📈 各指标随并发级别性能对比详情

#### 请求吞吐量（Request throughput (req/s)）

| 并发数 | NVIDIA_H100 | Hygon_BW1000 | 差值    | 百分比    |
|-----|-------------|--------------|-------|--------|
| 1   | 0.45        | 0.21         | -0.24 | -53.3% |
| 2   | 0.78        | 0.33         | -0.45 | -57.7% |
| 4   | 1.26        | 0.48         | -0.78 | -61.9% |
| 8   | 1.82        | 0.61         | -1.21 | -66.5% |
| 10  | 2.09        | 0.61         | -1.48 | -70.8% |
| 16  | 2.51        | 0.70         | -1.81 | -72.1% |
| 32  | 3.12        | 0.76         | -2.36 | -75.6% |
| 64  | 3.66        | 0.82         | -2.84 | -77.6% |
| 80  | 3.67        | 0.82         | -2.85 | -77.7% |
| 128 | 3.66        | 0.82         | -2.84 | -77.6% |


#### 输出token吞吐量（Output token throughput (tok/s)）

| 并发数 | NVIDIA_H100 | Hygon_BW1000 | 差值      | 百分比    |
|-----|-------------|--------------|---------|--------|
| 1   | 115.31      | 54.28        | -61.03  | -52.9% |
| 2   | 199.70      | 85.05        | -114.65 | -57.4% |
| 4   | 323.45      | 121.84       | -201.61 | -62.3% |
| 8   | 465.99      | 157.18       | -308.81 | -66.3% |
| 10  | 534.52      | 155.66       | -378.86 | -70.9% |
| 16  | 643.80      | 180.19       | -463.61 | -72.0% |
| 32  | 797.81      | 195.69       | -602.12 | -75.5% |
| 64  | 937.16      | 210.25       | -726.91 | -77.6% |
| 80  | 938.91      | 210.32       | -728.59 | -77.6% |
| 128 | 937.61      | 210.20       | -727.41 | -77.6% |


#### 总token吞吐量（Total token throughput (tok/s)）

| 并发数 | NVIDIA_H100 | Hygon_BW1000 | 差值        | 百分比    |
|-----|-------------|--------------|-----------|--------|
| 1   | 4745.14     | 2225.40      | -2519.74  | -53.1% |
| 2   | 8217.92     | 3487.19      | -4730.73  | -57.6% |
| 4   | 13310.92    | 4995.56      | -8315.36  | -62.5% |
| 8   | 19176.39    | 6444.25      | -12732.14 | -66.4% |
| 10  | 21996.95    | 6382.12      | -15614.83 | -71.0% |
| 16  | 26494.03    | 7387.79      | -19106.24 | -72.1% |
| 32  | 32831.81    | 8023.25      | -24808.56 | -75.6% |
| 64  | 38566.45    | 8620.30      | -29946.15 | -77.6% |
| 80  | 38638.21    | 8623.30      | -30014.91 | -77.7% |
| 128 | 38585.00    | 8618.28      | -29966.72 | -77.7% |


#### 首token延迟（P99 TTFT (ms)）

| 并发数 | NVIDIA_H100 | Hygon_BW1000 | 差值         | 百分比     |
|-----|-------------|--------------|------------|---------|
| 1   | 286.01      | 1168.49      | +882.48    | +308.5% |
| 2   | 466.40      | 2165.32      | +1698.92   | +364.3% |
| 4   | 826.89      | 4137.09      | +3310.20   | +400.3% |
| 8   | 1364.44     | 8133.44      | +6769.00   | +496.1% |
| 10  | 1534.23     | 10081.63     | +8547.40   | +557.1% |
| 16  | 2630.84     | 16063.93     | +13433.09  | +510.6% |
| 32  | 6556.86     | 31917.86     | +25361.00  | +386.8% |
| 64  | 12557.76    | 63531.55     | +50973.79  | +405.9% |
| 80  | 19679.20    | 140904.11    | +121224.91 | +616.0% |
| 128 | 29645.32    | 140903.29    | +111257.97 | +375.3% |

#### 每token生成时间（P99 TPOT (ms)）

| 并发数 | NVIDIA_H100 | Hygon_BW1000 | 差值      | 百分比     |
|-----|-------------|--------------|---------|---------|
| 1   | 7.68        | 14.16        | +6.48   | +84.4%  |
| 2   | 9.05        | 19.41        | +10.36  | +114.5% |
| 4   | 11.41       | 28.88        | +17.47  | +153.1% |
| 8   | 35.95       | 46.97        | +11.02  | +30.7%  |
| 10  | 17.75       | 60.61        | +42.86  | +241.5% |
| 16  | 23.92       | 85.10        | +61.18  | +255.8% |
| 32  | 38.79       | 160.15       | +121.36 | +312.9% |
| 64  | 66.03       | 300.74       | +234.71 | +355.5% |
| 80  | 66.45       | 301.17       | +234.72 | +353.2% |
| 128 | 66.52       | 300.89       | +234.37 | +352.3% |

#### token间延迟（P99 ITL (ms)）

| 并发数 | NVIDIA_H100 | Hygon_BW1000 | 差值      | 百分比     |
|-----|-------------|--------------|---------|---------|
| 1   | 8.57        | 20.17        | +11.60  | +135.4% |
| 2   | 16.54       | 24.07        | +7.53   | +45.5%  |
| 4   | 18.61       | 22.80        | +4.19   | +22.5%  |
| 8   | 152.68      | 24.15        | -128.53 | -84.2%  |
| 10  | 155.84      | 41.33        | -114.51 | -73.5%  |
| 16  | 161.83      | 42.55        | -119.28 | -73.7%  |
| 32  | 166.38      | 59.25        | -107.13 | -64.4%  |
| 64  | 170.95      | 64.72        | -106.23 | -62.1%  |
| 80  | 171.22      | 60.21        | -111.01 | -64.8%  |
| 128 | 170.62      | 85.02        | -85.60  | -50.2%  |


---

---

## 测试场景二：超长上下文请求测试
**测试目标**：对超长上下文的请求，使用vllm bench serve工具对并发数逐级增加场景的性能基准验证.

### 📊 测试概览

| 项目            | 配置                                                               | 备注  |
|---------------|------------------------------------------------------------------|-----|
| **数据集**       | random                                                           |     |
| **并发数**       | 1, 2, 4, 8, 10                                                   |     |
| **总请求数**      | 100                                                              |     |
| **请求输入上下文长度** | 194560（190k）                                                     |     |
| **请求输出上下文长度** | 1024（1k）                                                         |     |
| **被测芯片**      | NVIDIA_H100, Hygon_BW1000                                        |     |
| **被测模型**      | NVIDIA_H100 (MiniMax-M2.5);<br/>Hygon_BW1000 (MiniMax-M2.5-W8A8) |     |


### 📊 芯片性能对比柱状图

**1并发**

<img src="./chip_comparison_c1_test_02_nvidia_h100_vs_hygon_bw1000.png" width="1000" />

**2并发**

<img src="./chip_comparison_c2_test_02_nvidia_h100_vs_hygon_bw1000.png" width="1000" />

**4并发**

<img src="./chip_comparison_c4_test_02_nvidia_h100_vs_hygon_bw1000.png" width="1000" />

**8并发**

<img src="./chip_comparison_c8_test_02_nvidia_h100_vs_hygon_bw1000.png" width="1000" />

**10并发**

<img src="./chip_comparison_c10_test_02_nvidia_h100_vs_hygon_bw1000.png" width="1000" />


### 📈 性能趋势对比图 (所有芯片)

<img src="./performance_trends_test_02_nvidia_h100_vs_hygon_bw1000.png" width="1000" />

---

### 📈 各指标随并发级别性能对比详情

#### 请求吞吐量（Request throughput (req/s)）

| 并发数 | NVIDIA_H100 | Hygon_BW1000 | 差值    | 百分比    |
|-----|-------------|--------------|-------|--------|
| 1   | 0.05        | 0.01         | -0.04 | -80.0% |
| 2   | 0.06        | 0.02         | -0.04 | -66.7% |
| 4   | 0.07        | 0.02         | -0.05 | -71.4% |
| 8   | 0.07        | 0.01         | -0.06 | -85.7% |
| 10  | 0.07        | 0.01         | -0.06 | -85.7% |


#### 输出token吞吐量（Output token throughput (tok/s)）

| 并发数 | NVIDIA_H100 | Hygon_BW1000 | 差值     | 百分比    |
|-----|-------------|--------------|--------|--------|
| 1   | 46.70       | 15.31        | -31.39 | -67.2% |
| 2   | 62.03       | 17.32        | -44.71 | -72.1% |
| 4   | 71.85       | 17.31        | -54.54 | -75.9% |
| 8   | 75.61       | 15.33        | -60.28 | -79.7% |
| 10  | 75.37       | 15.00        | -60.37 | -80.1% |


#### 总token吞吐量（Total token throughput (tok/s)）

| 并发数 | NVIDIA_H100 | Hygon_BW1000 | 差值        | 百分比    |
|-----|-------------|--------------|-----------|--------|
| 1   | 8921.40     | 2924.75      | -5996.65  | -67.2% |
| 2   | 11850.06    | 3308.40      | -8541.66  | -72.1% |
| 4   | 13726.45    | 3306.71      | -10419.74 | -75.9% |
| 8   | 14443.68    | 2927.85      | -11515.83 | -79.7% |
| 10  | 14398.41    | 2865.09      | -11533.32 | -80.1% |


#### 首token延迟（P99 TTFT (ms)）

| 并发数 | NVIDIA_H100 | Hygon_BW1000 | 差值         | 百分比     |
|-----|-------------|--------------|------------|---------|
| 1   | 10539.10    | 44121.71     | +33582.61  | +318.6% |
| 2   | 20205.87    | 88441.47     | +68235.60  | +337.7% |
| 4   | 37530.99    | 178812.59    | +141281.60 | +376.4% |
| 8   | 81506.35    | 372041.15    | +290534.80 | +356.5% |
| 10  | 108342.29   | 474297.92    | +365955.63 | +337.8% |


#### 每token生成时间（P99 TPOT (ms)）

| 并发数 | NVIDIA_H100 | Hygon_BW1000 | 差值      | 百分比     |
|-----|-------------|--------------|---------|---------|
| 1   | 11.39       | 22.97        | +11.58  | +101.7% |
| 2   | 22.27       | 71.79        | +49.52  | +222.4% |
| 4   | 45.32       | 184.74       | +139.42 | +307.6% |
| 8   | 70.19       | 272.32       | +202.13 | +288.0% |
| 10  | 69.61       | 279.90       | +210.29 | +302.1% |


#### token间延迟（P99 ITL (ms)）

| 并发数 | NVIDIA_H100 | Hygon_BW1000 | 差值       | 百分比     |
|-----|-------------|--------------|----------|---------|
| 1   | 22.97       | 32.12        | +9.15    | +39.8%  |
| 2   | 291.30      | 69.78        | -221.52  | -76.0%  |
| 4   | 518.54      | 3071.54      | +2553.00 | +492.3% |
| 8   | 642.87      | 4014.05      | +3371.18 | +524.4% |
| 10  | 639.97      | 4012.34      | +3372.37 | +527.0% |


---

---


## 测试场景三：长上下文高并发极限验证

**测试目标**：多并发长上下文的情况下，验证各芯片单节点同时能处理的最大请求数。

### 📊 测试概览

| 项目            | 配置                                                               | 备注  |
|---------------|------------------------------------------------------------------|-----|
| **数据集**       | random                                                           |     |
| **并发数**       | 32, 64                                                           |     |
| **总请求数**      | 1000                                                             |     |
| **请求输入上下文长度** | 90000（约90k）                                                      |     |
| **请求输出上下文长度** | 2000（约2k）                                                        |     |
| **被测芯片**      | NVIDIA_H100, Hygon_BW1000                                        |     |
| **被测模型**      | NVIDIA_H100 (MiniMax-M2.5);<br/>Hygon_BW1000 (MiniMax-M2.5-W8A8) |     |


### 监控处理请求极限

#### nvidia_h100芯片
**测试平稳后，最大可同时处理请求数13个，且非常稳定，缓存命中率波动正常**
![test_03_nvidia_h100.png](test_03_nvidia_h100.png)

#### Hygon_BW1000芯片
**测试平稳后，最大可同时处理请求数在10~18之间波动，缓存命中率正常（在0~3%之间）**
![test_03_hygon_bw1000.png](test_03_hygon_bw1000.png)

### 📊 芯片性能对比柱状图

**32并发**

<img src="./chip_comparison_c32_test_03_nvidia_h100_vs_hygon_bw1000.png" width="1000" />

**64并发**

<img src="./chip_comparison_c64_test_03_nvidia_h100_vs_hygon_bw1000.png" width="1000" />


### 📈 各并发级别性能对比详情


#### 32 并发

| 指标                                          | NVIDIA_H100     | Hygon_BW1000 |
|---------------------------------------------|-----------------|--------------|
| 请求吞吐量（Request throughput (req/s)）           | **0.15** ⭐      | 0.03         |
| 输出token吞吐量（Output token throughput (tok/s)） | **307.27** ⭐    | 58.42        |
| 总token吞吐量（Total token throughput (tok/s)）   | **14140.63** ⭐  | 2687.12      |
| 首token延迟（P99 TTFT (ms)）                     | **151671.11** ⭐ | 840076.37    |
| 每token生成时间（P99 TPOT (ms)）                   | **40.78** ⭐     | 164.03       |
| token间延迟（P99 ITL (ms)）                      | **363.93** ⭐    | 2600.66      |


#### 64 并发

| 指标                                          | NVIDIA_H100     | Hygon_BW1000 |
|---------------------------------------------|-----------------|--------------|
| 请求吞吐量（Request throughput (req/s)）           | **0.15** ⭐      | 0.03         |
| 输出token吞吐量（Output token throughput (tok/s)） | **307.19** ⭐    | 58.35        |
| 总token吞吐量（Total token throughput (tok/s)）   | **14136.71** ⭐  | 2683.97      |
| 首token延迟（P99 TTFT (ms)）                     | **345223.92** ⭐ | 1925575.70   |
| 每token生成时间（P99 TPOT (ms)）                   | **40.79** ⭐     | 164.14       |
| token间延迟（P99 ITL (ms)）                      | **364.12** ⭐    | 2600.66      |

---

---

## 测试场景四：多I/O测试

### 测试目标
**测试不同输入输出长度和并发级别下的性能表现，分析同一芯片同一模型在不同输入输出长度和并发级别下的性能指标变化趋势。**

### 📊 测试概览

| 项目         | 配置                                                                            | 备注  |
|------------|-------------------------------------------------------------------------------|-----|
| **数据集**    | random                                                                        |     |
| **并发数**    | 1, 4, 8, 16, 32, 64, 128                                                      |     |
| **总请求数**   | 1000                                                                          |     |
| **输入输出长度** | (128, 128), (512, 256), (1024, 512), (2048, 1024), (4096, 2048), (8192, 1024) |     |
| **被测芯片**   | NVIDIA_H100, Hygon_BW1000                                                     |     |
| **被测模型**   | NVIDIA_H100 (MiniMax-M2.5);<br/>Hygon_BW1000 (MiniMax-M2.5-W8A8)              |     |

### 📋 各I/O测试汇总（固定请求上下文长度，随并发数变化）
> **报告说明: 由于本测试场景比较多，此报告仅列出一组测试结果作为示例，且仅列出Hygon_BW1000测试结果**

#### input: 8192, output: 1024

| 并发数 | 请求吞吐量 (req/s) | 输出Token吞吐量 (tok/s) | 总Token吞吐量 (tok/s) | TTFT P99 (ms) | TPOT P99 (ms) | ITL P99 (ms) |
|-----|---------------|--------------------|-------------------|---------------|---------------|--------------|
| 1   | 0.06          | 66.11              | 594.97            | 892.34        | 14.31         | 23.17        |
| 4   | 0.20          | 202.97             | 1826.69           | 3335.25       | 19.43         | 24.63        |
| 8   | 0.31          | 320.18             | 2881.59           | 6485.86       | 24.72         | 32.22        |
| 16  | 0.41          | 424.67             | 3822.06           | 12596.58      | 37.34         | 43.69        |
| 32  | 0.50          | 516.05             | 4644.47           | 19688.71      | 61.36         | 64.07        |
| 64  | 0.59          | 602.89             | 5425.98           | 45274.45      | 112.09        | 1157.52      |
| 128 | 0.59          | 602.62             | 5423.60           | 152443.57     | 114.39        | 2621.29      |

![性能图表](./i8192_o1024/concurrency_comparison.png)

### 📊 I/O对比（固定并发数, 随请求上下文长度变化）

#### 并发数 = 1

| 指标                 | i128_o128 | i512_o256 | i1024_o512 | i2048_o1024 | i4096_o2048 | i8192_o1024 |
|--------------------|-----------|-----------|------------|-------------|-------------|-------------|
| 请求吞吐量 (req/s)      | 0.54      | 0.27      | 0.14       | 0.07        | 0.03        | 0.06        |
| 输出Token吞吐量 (tok/s) | 69.71     | 69.57     | 69.93      | 69.74       | 69.33       | 66.11       |
| 总Token吞吐量 (tok/s)  | 139.42    | 208.70    | 209.80     | 209.22      | 208.00      | 594.97      |
| TTFT P99 (ms)      | 148.98    | 155.38    | 213.04     | 312.85      | 486.84      | 892.34      |
| TPOT P99 (ms)      | 13.50     | 13.93     | 13.98      | 14.09       | 14.22       | 14.31       |
| ITL P99 (ms)       | 23.34     | 22.43     | 21.33      | 24.12       | 22.92       | 23.17       |

![I/O对比](./compare_by_io_conc1/io_comparison.png)

---

---


## 测试场景五：模型精度测试

### 测试目标
模型精度测试目标主要是通过标准指标（如准确率、精确率、召回率、F1值、mAP、AUC）衡量模型在测试集上的输出与真实标签的一致性，评估其基本判别能力。

### MiniMax-M2.5模型 - 各测试任务整体比对


| Task                        | NVIDIA_H100(FP8) | Hygon_BW1000(W8A8) | 差值      | 百分比      |
|-----------------------------|------------------|--------------------|---------|----------|
| IFBench (Strict)            | 0.6067           | 0.2467             | -0.3600 | - 59.34% |
| IFBench (Loose)             | 0.6433           | 0.2767             | -0.3667 | - 56.99% |
| lm-eval:gsm_plus (Flexible) | 0.6863           | 0.6655             | -0.0208 | - 3.03%  |
| lm-eval:gsm_plus (Strict)   | 0.7307           | 0.7301             | -0.0006 | - 0.08%  |
| lm-eval:mmlu_pro            | 0.7378           | N/A                | N/A     | N/A      |
| lm-eval:ruler               | 0.5461           | N/A                | N/A     | N/A      |

**注：** 海光的测试环境由于无法进行uv虚拟环境依赖安装，mmlu_pro和ruler任务未能执行


>- IFBench模型精度测试脚本参见《附录二》
>- lm-eval模型精度测试脚本参见《附录三》

---

---

## 测试场景六：多轮对话测试

### 📊 测试概览

**测试工具：vllm 内置的多轮对话测试脚本**

| 条目                              | 值                |
|:--------------------------------|:-----------------|
| 并发客户端 (num_clients)             | 50               |
| 总对话轮数 (num_conversations)       | 1000             |
| 活跃对话数 (active_conversations)    | 100              |
| 输入轮数 (input_num_turns)          | 10               |
| 输入 Token 长度 (input_num_tokens)  | 200              |
| 输出 Token 长度 (output_num_tokens) | 100              |
| 语料来源                            | 推荐的语料 pg1184.txt |
| seed                            | 0                |

#### 和英伟达比对

| 指标                | NVIDIA_H100 | Hygon_BW1000 |
|:------------------|:------------|:-------------|
| 总耗时               | 140.633 秒   | 416.808 秒      |
| 请求吞吐量（RPS）        | 28.087        | 9.477        |


#### Hygon_BW1000多轮对话详细测试结果
| 指标                | count | mean    | std    | 99%     | 99.9%   | max     |
|:------------------|:------|:--------|:-------|:--------|:--------|:--------|
| ttft_ms           | 3950  | 1334.39 | 489.78 | 2497.44 | 2869.67 | 5106.10 |
| tpot_ms           | 3950  | 38.24   | 6.24   | 63.09   | 69.61    | 80.53   |
| latency_ms        | 3950  | 5265.98 | 550.01  | 8109.88 | 8233.94 | 9837.93 |
| input_num_turns   | 3950  | 5.96    | 2.22   | 9.00    | 9.00    | 9.00    |
| input_num_tokens  | 3950  | 949.43  | 336.67 | 1419.00 | 1425.00 | 1430.00 |
| output_num_tokens | 3950  | 103.91  | 2.07   | 110.0  | 112.05  | 120.00  |
| output_num_chunks | 3950  | 97.99   | 5.28   | 99.00   | 99.00   | 99.00   |

>多轮对话测试脚本参见本报告《附录四》

---

---

## 测试场景七：基础推理能力验证

### 测试目标：
验证模型在芯片环境上的基础推理和兼容性的支持能力情况，可作为快速选型的一个基础指标。

### 测试说明
> 比对说明：本章节只列出在Hygon_BW1000芯片平台上的测试结果

> 状态说明：✅ 已通过，⏳ 未测试，❌ 未通过，⚠️ 部分通过

### A. 基础推理能力

| #   | 测试点            | 测试内容                      | 状态  | 备注  |
|-----|----------------|---------------------------|-----|-----|
| A1  | 单轮对话           | 发送单条prompt，验证正常生成         | ✅   |     |
| A2  | 多轮对话           | 5轮对话，验证上下文保持和连贯性          | ✅   |     |
| A3  | System Prompt  | 设置系统角色，验证模型遵循程度           | ✅   |     |
| A4  | 流式输出           | stream=true，验证SSE逐token返回 | ✅   |     |
| A5  | 非流式输出          | stream=false，验证完整返回       | ✅   |     |
| A6  | Temperature 控制 | temp=0 vs temp=1.0，验证输出差异 | ✅   |     |
| A7  | Top-p/Top-k采样  | 不同top_p/top_k值，验证多样性控制    | ✅   |     |
| A8  | Max Tokens限制   | 设置max_tokens，验证输出不超限      | ✅   |     |
| A9  | Stop Sequences | 设置stop token，验证截断         | ✅   |     |
| A10 | Seed 可复现性      | 相同seed+temp=0，验证输出一致      | ✅   |     |
| A11 | 多语言能力          | 中/英/日/韩/法等多语言输入输出         | ✅   |     |
| A12 | 特殊Token处理      | 含emoji、代码块、数学符号、HTML标签    | ✅   |     |


### B. 高级生成功能

| #   | 测试点             | 测试内容                       | 状态  | 备注                                    |
|-----|-----------------|----------------------------|-----|---------------------------------------|
| B1  | 思考模式（Thinking）  | 开启thinking mode，验证返回思考链... | ✅   |                                       |
| B2  | 非思考模式（Instant）  | 关闭thinking，验证无hidden th... | ❌   | 关闭思考模式，依然有思考内容以<\/think>结尾出现在content里 |
| B3  | 思考模式切换          | 同一会话内thinking↔non-think... | ❌   | 使用默认temperature 0.7                   |
| B4  | 工具调用-单工具        | 定义单个function，验证模型正确调用并传参   | ✅   |                                       |
| B5  | 工具调用-多工具        | 定义多个function，验证模型选择正确的工具   | ❌   | 未正常调用多个工具                             |
| B6  | 工具调用-并行调用       | 单次回复中并行调用多个工具              | ✅   |                                       |
| B7  | 工具调用-多步链式       | 工具结果作为下一步输入，验证3+步链式执行      | ❌   | 未正常进行链式工具调用                           |
| B8  | JSON Mode       | response_format=json_ob... | ✅   |                                       |
| B9  | 结构化输出           | JSON Schema约束输出格式，验证字段完整性  | ✅   |                                       |
| B10 | Prefix/Suffix约束 | 指定输出前缀或格式模板，验证遵循度          | ❌   |                                       |


### C. 多模态能力 （MiniMax-M2.5模型为文本模型，此测试组请跳过）


| #   | 测试点     | 测试内容          | 状态  | 备注  |
|-----|---------|---------------|-----|-----|
| C1  | 单图理解    | 图片+文本提问       | ❌   |     |
| C2  | 多图对比    | 跨图比较          | ❌   |     |
| C3  | 高分辨率图片  | 4K分辨率         | ❌   |     |
| C4  | 图表/OCR  | 表格截图          | ❌   |     |
| C5  | 视频理解    | 视频文件          | ❌   |     |
| C6  | 代码截图→代码 | UI截图          | ❌   |     |
| C7  | 多模态工具调用 | 图片触发工具        | ❌   |     |
| C8  | 图片格式兼容性 | PNG/JPEG/WebP | ❌   |     |

### D. 长上下文处理

| #   | 测试点       | 测试内容                        | 状态 | 备注  |
|-----|------------|-----------------------------|------|-----|
| D1  | 短上下文基线     | 1K tokens                  | ✅ |     |
| D2  | 中等上下文      | 8K-16K tokens              | ✅ |     |
| D3  | 长上下文       | 32K-64K tokens             | ✅ |     |
| D4  | 超长上下文      | 128K+ tokens               | ✅ |     |
| D5  | 大海捞针       | NIAH                       | ✅ |     |
| D6  | 上下文边界行为    | max_model_len              | ✅ |     |
| D7  | 超出上下文截断    | 截断/拒绝                      | ✅ |     |
| D8  | 长输出生成      | 4K-8K tokens               | ✅ |     |


### F. 稳定性与边界

| #   | 测试点    | 测试内容                                        | 状态  | 备注                                                                        |
|-----|--------|---------------------------------------------|-----|---------------------------------------------------------------------------|
| F1  | 空输入    | 空prompt                                     | ✅   | 返回 Empty message handled: We have a conversation. The user wrote only " " |
| F2  | 超大输入   | 超max_model_len                              | ✅   | 返回 Oversized input handled, finish_reason: length                         |
| F3  | 非法参数   | temperature=-1, max_tokens=0,非法温度值：超过范围（>2） | ✅   | 返回 API request failed with status 400                                     |
| F4  | 特殊字符注入 | SQL/Prompt注入                                | ✅   |                                                                           |
| F5  | 并发稳定性  | 200+并发（实际测试50并发）                            | ✅   |                                                                           |
| F6  | OOM恢复  | 显存耗尽                                        | ⏳   |                                                                           |
| F7  | 长时间运行  | 24小时                                        | ⏳   |                                                                           |
| F8  | 请求超时处理 | 超时断开                                        | ⏳   |                                                                           |


### G. API兼容性

| #   | 测试点                     | 测试内容                       | 状态  | 备注  |
|-----|-------------------------|----------------------------|-----|-----|
| G1  | OpenAI Chat Completions | /v1/chat/completions 接口兼容  | ✅   |     |
| G2  | OpenAI Completions      | /v1/completions 接口兼容       | ✅   |     |
| G3  | 模型列表                    | /v1/models 返回可用模型          | ✅   |     |
| G4  | Usage 统计                | usage 字段准确                 | ✅   |     |
| G5  | 错误码规范                   | 400/401/404/429/500 错误码    | ⏳   |     |
| G6  | 客户端 SDK 兼容              | Python openai / JS @ope... | ⏳   |     |
| G7  | 响应格式变体                  | 不同response_format          | ✅   |     |
| G8  | Stream参数                | stream参数测试                 | ✅   |     |

### H. 质量评估

| #   | 测试点   | 测试内容  | 状态  | 备注  |
|-----|-------|-------|-----|-----|
| H1  | 生成质量  | 质量对比  | ✅   |     |
| H2  | 生成一致性 | 多次生成  | ✅   |     |
| H3  | 幻觉率   | 事实错误  | ✅   |     |
| H4  | 指令遵循度 | 格式/角色 | ✅   |     |
| H5  | 响应相关性 | 问答相关性 | ✅   |     |


### I. 超长上下文验证


| #   | 测试点       | 测试内容                        | 状态 | 备注  |
|-----|------------|-----------------------------|------|-----|
| I1  | 超长上下文（非流式） | 验证超长上下文请求的非流式输出            | ✅ |     |
| I2  | 超长上下文（流式）  | 验证超长上下文请求的流式输出             | ✅ |     |
| I3  | 超长上下文（边界验证） | 使用二分法逼近模型最大上下文长度           | ✅ |     |
| I4  | 超长上下文（思考模式） | 验证超长上下文下reasoning_conte... | ✅ |     |


---

---

## 附录一：benchmark执行脚本

**执行命令**
```shell
python run_benchmark.py --chip hygon_bw1000 --model MiniMax-M2.5-W8A8 --test-suite test_01,test_02,test_03,test_04,test_05

# 如果不指定test-suite参数，默认执行run_benchmark.py里TEST_SUITES定义的测试套件列表，可以根据自己需要修改默认测试套件
```

**run_benchmark.py**
```python
import os
import yaml
import subprocess
import requests
import time
from datetime import datetime
from itertools import product
from pathlib import Path

API_KEY = os.environ.get("API_KEY", "abc123")

TEST_SUITES = ["test_01"]

RUN_ID = "01"

try:
    from gpu_monitor import GPUMonitor, generate_gpu_charts

    HAS_GPU_MONITOR = True
except ImportError:
    HAS_GPU_MONITOR = False
    print("Warning: GPU monitor module not available")


def get_model_info_from_api(base_url, api_key):
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        response = requests.get(f"{base_url}/v1/models", headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data.get("data") and len(data["data"]) > 0:
                model_info = data["data"][0]
                model_name = model_info.get("id")
                owned_by = model_info.get("owned_by")
                model_path = model_info.get("root")
                if owned_by == "vllm" and model_path:
                    return model_name, model_path
                else:
                    return model_name, None
    except Exception as e:
        print(f"Failed to get model info from API: {e}")
    return None, None


def run_benchmark(chip_name, base_config, model_config, test_suites, run_id):
    base_url = base_config.get("base_url", "http://127.0.0.1:8080")

    model_name_yaml = model_config.get("name")
    served_model_name = model_config.get("served-model-name")
    model_path_yaml = model_config.get("model_path")

    model_name, model_path = get_model_info_from_api(base_url, API_KEY)

    if not model_name:
        model_name = served_model_name
    if not model_path:
        model_path = model_path_yaml

    print(f"Model Name: {model_name}")
    print(f"Model Path: {model_path}")
    print(f"Running test suites: {', '.join(test_suites)}")

    temperature = base_config.get("temperature", 0.7)
    seed = base_config.get("seed", 123)
    ready_timeout = base_config.get("ready-check-timeout-sec", 30)

    M = model_name_yaml
    output_base = f"reports/benchmark/{chip_name}/{M}"

    params_config = base_config.get("params", {})

    for test_suite in test_suites:
        test_params = params_config.get(test_suite, {})
        max_concurrency = test_params.get("max-concurrency", [10])
        num_prompts = test_params.get("num-prompts", [300])
        random_input_output_len = test_params.get(
            "random-input-output-len", [[20000, 100]]
        )

        run_id_dir = os.path.join(output_base, test_suite, run_id)
        if os.path.exists(run_id_dir):
            print(
                f"Error: Run ID '{run_id}' already exists for test suite '{test_suite}' at path: {run_id_dir}"
            )
            print(f"Please either:")
            print(f"  1. Use a different RUN_ID (--run-id)")
            print(f"  2. Delete the existing directory: {run_id_dir}")
            continue

        print(f"\n=== Running test suite: {test_suite} ===")

        gpu_monitor = GPUMonitor(interval=10) if HAS_GPU_MONITOR else None

        for nc, np, io_len in product(
            max_concurrency, num_prompts, random_input_output_len
        ):
            ni = io_len[0]
            no = io_len[1]
            param_dir = f"{test_suite}/{run_id}/{nc}-{np}-i{ni}-o{no}"
            output_dir = os.path.join(output_base, param_dir)
            Path(output_dir).mkdir(parents=True, exist_ok=True)

            log_file = os.path.join(
                output_dir, f"bench-{test_suite}-{nc}-{np}-i{ni}-o{no}.log"
            )

            if gpu_monitor:
                monitor_param_dir = f"{test_suite}/{run_id}/{nc}-{np}-i{ni}-o{no}"
                gpu_monitor.start_monitoring(
                    "monitor", chip_name, model_name_yaml, monitor_param_dir
                )

            cmd = [
                "vllm",
                "bench",
                "serve",
                "--backend",
                "openai-chat",
                "--endpoint",
                "/v1/chat/completions",
                "--dataset-name",
                test_params.get("dataset-name", "random"),
                "--random-input-len",
                str(ni),
                "--random-output-len",
                str(no),
                "--model",
                str(model_path),
                "--trust-remote-code",
                "--base-url",
                base_url,
                "--num-prompts",
                str(np),
                "--max-concurrency",
                str(nc),
                "--temperature",
                str(temperature),
                "--seed",
                str(seed),
                "--metric_percentiles",
                "95,99",
                "--served-model-name",
                str(model_name),
                "--ready-check-timeout-sec",
                str(ready_timeout),
            ]

            print(f"Running: {' '.join(cmd)}")
            print(f"Log file: {log_file}")

            log_f = open(log_file, "w")
            process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
            )

            for line in process.stdout:
                print(line, end="")
                log_f.write(line)

            process.wait()
            log_f.close()

            if gpu_monitor:
                gpu_log = gpu_monitor.stop_monitoring()
                if gpu_log:
                    gpu_log_dir = os.path.dirname(gpu_log)
                    generate_gpu_charts(gpu_log, gpu_log_dir)

            print(f"Completed: {log_file}")
            time.sleep(30)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run vLLM benchmark")
    parser.add_argument(
        "--chip",
        type=str,
        required=True,
        help="Chip name to test (e.g., hygon_bw1000, kunlun_p800, nvidia_h100)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name to test (e.g., MiniMax-M2.5, Qwen3.5-122B-A10B). If not specified, uses the first model in config.",
    )
    parser.add_argument(
        "--test-suite",
        type=str,
        default=None,
        help=f"Test suite to run (default: all). Available: {', '.join(TEST_SUITES)}",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=RUN_ID,
        help=f"Run ID to identify this test run (default: {RUN_ID})",
    )
    args = parser.parse_args()

    yaml_path = os.path.join(
        os.path.dirname(__file__), "config", "models_scenarios.yaml"
    )

    with open(yaml_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    base_config = config.get("base_config", {})
    params_config = base_config.get("params", {})
    models = config.get("models", {})

    chip_name = args.chip.lower()
    if chip_name not in models:
        print(
            f"Error: Chip '{chip_name}' not found in config. Available chips: {', '.join(models.keys())}"
        )
        return

    available_models = models[chip_name]

    if args.model:
        model_name_lower = args.model.lower()
        selected_model = None
        for m in available_models:
            if m.get("name", "").lower() == model_name_lower:
                selected_model = m
                break
        if not selected_model:
            print(
                f"Error: Model '{args.model}' not found for chip '{chip_name}'. Available models:"
            )
            for m in available_models:
                print(f"  - {m.get('name')} (served: {m.get('served-model-name')})")
            return
        model_configs = [selected_model]
    else:
        model_configs = [available_models[0]]
        print(f"No model specified, using default: {model_configs[0].get('name')}")

    test_suites_to_run = []
    if args.test_suite:
        test_suites_to_run = [s.strip() for s in args.test_suite.split(",")]
    else:
        test_suites_to_run = TEST_SUITES

    invalid_suites = [s for s in test_suites_to_run if s not in params_config]
    if invalid_suites:
        print(
            f"Error: Test suite(s) {invalid_suites} not found in config. Available: {', '.join(params_config.keys())}"
        )
        return

    run_id = args.run_id

    for model_config in model_configs:
        print(f"Processing chip: {chip_name}, model: {model_config.get('name')}")
        run_benchmark(chip_name, base_config, model_config, test_suites_to_run, run_id)
        print(f"Finished chip: {chip_name}, model: {model_config.get('name')}")


if __name__ == "__main__":
    main()


```

## 附录二：IFBench精度测试脚本

以MiniMax-M2.5-W8A8模型为例

---

**ifbench_mm25_w8a8.sh**

```shell
#!/bin/bash
ROOT_PATH=$(cd `dirname $0`; pwd)

echo $ROOT_PATH
cd ${ROOT_PATH}

CurDate=`date +'%Y%m%d'`
export NLTK_DATA=${ROOT_PATH}/nltk_data

cat > .env << 'EOF'
api_base=http://127.0.0.1:8000/v1
api_key=abc123
model=minimax-m2.5
temperature=1.0
top_p=0.95
top_k=40
max_tokens=8192
seed=42
input_file=data/IFBench_test.jsonl
output_file=data/mm25-responses.jsonl
workers=32
EOF

# 2. 生成模型响应
uv run python generate_responses.py

# 3. Thinking 模型后处理（重要！）
uv run python postprocess_thinking.py data/mm25-responses.jsonl -o data/mm25-clean.jsonl

# 4. 运行评估
uv run python -m run_eval \
	--input_data=data/IFBench_test.jsonl \
	--input_response_data=data/mm25-clean.jsonl \
	--output_dir=eval


```

## 附录三： lm-eval精度测试脚本

以MiniMax-M2.5-W8A8模型为例

---

**lm_eval_test.sh**

```shell
#!/bin/bash
ROOT_PATH=$(cd `dirname $0`; pwd)

echo $ROOT_PATH
cd ${ROOT_PATH}

CurDate=`date +'%Y%m%d'`

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

#export HF_ENDPOINT=https://hf-mirror.com

ADDR=${ADDR:-127.0.0.1}
PORT=${PORT:-8000}
API_KEY=${API_KEY:-abc123}
LLM_ADDR="http://$ADDR:$PORT"

# 自动获取模型名和 tokenizer 路径
#MODEL_NAME=$(curl -s --header "Authorization: Bearer $API_KEY" $LLM_ADDR/v1/models | jq -r .data[0].id)
#MODEL_PATH=$(curl -s --header "Authorization: Bearer $API_KEY" $LLM_ADDR/v1/models | jq -r .data[0].root)
MODEL_NAME="minimax-m2.5"
LOCAL_MODEL_PATH="/data/models/MiniMax-M2.5-W8A8"

# model_args 构造
MODEL_ARGS_BASE_1="{\"model\":\"$MODEL_NAME\",\"base_url\":\"$LLM_ADDR/v1/completions\",\"max_length\":131072,\"tokenizer\":\"$LOCAL_MODEL_PATH\",\"trust_remote_code\":true,\"num_concurrent\":10,\"max_retries\":3,\"timeout\":12000,\"tokenized_requests\":false,\"headers\":{\"Authorization\":\"Bearer $API_KEY\"}}"
MODEL_ARGS_BASE_2="{\"model\":\"$MODEL_NAME\",\"base_url\":\"$LLM_ADDR/v1/completions\",\"max_length\":192512,\"tokenizer\":\"$LOCAL_MODEL_PATH\",\"trust_remote_code\":true,\"num_concurrent\":10,\"max_retries\":3,\"timeout\":12000,\"tokenized_requests\":false,\"headers\":{\"Authorization\":\"Bearer $API_KEY\"}}"

# 运行单个任务的函数
run_task_1() {
	local task_name=$1
	local max_tokens=$2
	local temperature=$3
	local unsafe_code=$4
	
	local do_sample="false"
	[ "$temperature" = "1.0" ] && do_sample="true"

	GEN_KWARGS="{\"max_gen_toks\":$max_tokens,\"do_sample\":$do_sample,\"temperature\":$temperature,\"top_p\":0.95,\"top_k\":40}"

	local unsafe_flag=""
	[ "$unsafe_code" = "true" ] && unsafe_flag="--confirm_run_unsafe_code" && export HF_ALLOW_CODE_EVAL=1
	
	lm_eval \
		--model local-completions \
		--tasks $task_name \
		--output_path ./output/${task_name}/${MODEL_NAME}_${CurDate} \
		--model_args "$MODEL_ARGS_BASE_1" \
		--batch_size auto \
		--gen_kwargs "$GEN_KWARGS" \
		$unsafe_flag
}


run_task_2() {
	local task_name=$1
	local max_tokens=$2
	local temperature=$3
	local unsafe_code=$4
	
	local do_sample="false"
	[ "$temperature" = "1.0" ] && do_sample="true"

	GEN_KWARGS="{\"max_gen_toks\":$max_tokens,\"do_sample\":$do_sample,\"temperature\":$temperature,\"top_p\":0.95,\"top_k\":40}"

	local unsafe_flag=""
	[ "$unsafe_code" = "true" ] && unsafe_flag="--confirm_run_unsafe_code" && export HF_ALLOW_CODE_EVAL=1
	
	lm_eval \
		--model local-completions \
		--tasks $task_name \
		--output_path ./output/${task_name}/${MODEL_NAME}_${CurDate} \
		--model_args "$MODEL_ARGS_BASE_2" \
		--batch_size auto \
		--limit 32 \
		--gen_kwargs "$GEN_KWARGS" \ 
		$unsafe_flag
}


run_task_1 mmlu_pro 8192 0.0 false

sleep 120
run_task_1 gsm_plus 8192 0.0 false

sleep 120
run_task_2 ruler 8192 0.0 false

```

## 附录四： 多轮对话测试命令

```shell
python benchmark_serving_multi_turn.py \
  --model /data/models/MiniMax-M2.5-W8A8 \
  --served-model-name minimax-m2.5 \
  --url http://127.0.0.1:8080/v1 \
  --input-file generate_conversations.json \
  --output-file generate_conversations_output.json \
  --num-clients 50 \
  --max-active-conversations 100 \
  --warmup-step \
  --extra-body-json {"chat_template_kwargs": {"enable_thinking": false}}
```

---

---
