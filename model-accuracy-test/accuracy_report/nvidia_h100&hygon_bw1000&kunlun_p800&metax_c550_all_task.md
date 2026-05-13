# Accuracy Test Comparison Report - Chips: nvidia_h100&hygon_bw1000&kunlun_p800&metax_c550

## GLM-5模型

| Task | nvidia_h100(FP8) | nvidia_h100(INT4) | hygon_bw1000(W8A8) | 差值 | 百分比 | metax_c550(W8A8) | 差值 | 百分比 |
|------|------|------|------|------|------|------|------|------|
| IFBench (Strict) | 0.6667 | 0.6367 | 0.3767 | -0.2900 | - 43.50% | 0.6467 | -0.0200 | - 3.00% |
| IFBench (Loose) | 0.7133 | 0.6867 | 0.4100 | -0.3033 | - 42.52% | 0.6967 | -0.0167 | - 2.34% |
| lm-eval:gsm_plus (Flexible) | 0.7638 | 0.7634 | N/A | N/A | N/A | 0.7651 | 0.0013 | + 0.17% |
| lm-eval:gsm_plus (Strict) | 0.7642 | 0.7637 | N/A | N/A | N/A | 0.7656 | 0.0014 | + 0.18% |
| lm-eval:mmlu_pro | 0.7858 | 0.7867 | N/A | N/A | N/A | 0.7893 | 0.0035 | + 0.45% |
| lm-eval:ruler | N/A | 0.9467 | N/A | N/A | N/A | N/A | N/A | N/A |

## GLM-5.1模型

| Task | nvidia_h100(FP8) |
|------|------|
| IFBench (Strict) | 0.6000 |
| IFBench (Loose) | 0.6233 |

## Kimi-K2.5模型

| Task | metax_c550(INT4) | 差值 | 百分比 |
|------|------|------|------|
| IFBench (Strict) | 0.6067 | N/A | N/A |
| IFBench (Loose) | 0.6600 | N/A | N/A |
| lm-eval:gsm_plus (Flexible) | 0.7514 | N/A | N/A |
| lm-eval:gsm_plus (Strict) | 0.7510 | N/A | N/A |
| lm-eval:mmlu_pro | 0.8264 | N/A | N/A |
| lm-eval:ruler | 0.9672 | N/A | N/A |

## MiniMax-M2.5模型

| Task | nvidia_h100(FP8) | hygon_bw1000(W8A8) | 差值 | 百分比 | kunlun_p800(W8A8-INT8-Dynamic) | 差值 | 百分比 | metax_c550(W8A8) | 差值 | 百分比 |
|------|------|------|------|------|------|------|------|------|------|------|
| IFBench (Strict) | 0.6067 | 0.2467 | -0.3600 | - 59.34% | 0.6233 | 0.0167 | + 2.75% | 0.5700 | -0.0367 | - 6.04% |
| IFBench (Loose) | 0.6433 | 0.2767 | -0.3667 | - 56.99% | 0.6600 | 0.0167 | + 2.59% | 0.5967 | -0.0467 | - 7.25% |
| lm-eval:gsm_plus (Flexible) | 0.6863 | 0.6655 | -0.0208 | - 3.03% | 0.7398 | 0.0535 | + 7.80% | 0.7486 | 0.0623 | + 9.08% |
| lm-eval:gsm_plus (Strict) | 0.7307 | 0.7301 | -0.0006 | - 0.08% | 0.7251 | -0.0056 | - 0.77% | 0.7334 | 0.0027 | + 0.37% |
| lm-eval:mmlu_pro | 0.7378 | N/A | N/A | N/A | 0.6622 | -0.0756 | - 10.25% | 0.6686 | -0.0692 | - 9.38% |
| lm-eval:ruler | 0.5461 | N/A | N/A | N/A | N/A | N/A | N/A | 0.8972 | 0.3511 | + 64.30% |

# 任务子数据集详细比对结果

## GLM-5模型 - mmlu_pro任务子数据集详细比对结果

| Item | nvidia_h100(FP8) | nvidia_h100(INT4) | metax_c550(W8A8) | 差值 | 百分比 |
|------|------|------|------|------|------|
| biology | 0.8968 | 0.9010 | 0.8898 | -0.0070 | - 0.78% |
| business | 0.8390 | 0.8264 | 0.8137 | -0.0253 | - 3.02% |
| chemistry | 0.8030 | 0.8048 | 0.8101 | 0.0071 | + 0.88% |
| computer_science | 0.8122 | 0.8439 | 0.8390 | 0.0268 | + 3.30% |
| economics | 0.8519 | 0.8566 | 0.8578 | 0.0059 | + 0.69% |
| engineering | 0.6213 | 0.6099 | 0.6316 | 0.0103 | + 1.66% |
| health | 0.7885 | 0.7995 | 0.7922 | 0.0037 | + 0.47% |
| history | 0.7297 | 0.7218 | 0.7349 | 0.0052 | + 0.71% |
| law | 0.6013 | 0.5831 | 0.5985 | -0.0028 | - 0.47% |
| math | 0.8779 | 0.8912 | 0.8934 | 0.0155 | + 1.77% |
| other | 0.7684 | 0.7619 | 0.7619 | -0.0065 | - 0.85% |
| philosophy | 0.7735 | 0.7836 | 0.7776 | 0.0041 | + 0.53% |
| physics | 0.8122 | 0.8168 | 0.8245 | 0.0123 | + 1.51% |
| psychology | 0.8333 | 0.8346 | 0.8308 | -0.0025 | - 0.30% |

## GLM-5模型 - ruler任务子数据集详细比对结果

| Item | nvidia_h100(INT4) |
|------|------|
| niah_multikey_1 | 1.0000 |
| niah_multikey_2 | 1.0000 |
| niah_multikey_3 | 1.0000 |
| niah_multiquery | 1.0000 |
| niah_multivalue | 1.0000 |
| niah_single_1 | 1.0000 |
| niah_single_2 | 1.0000 |
| niah_single_3 | 1.0000 |
| ruler_cwe | 0.9375 |
| ruler_fwe | 0.9896 |
| ruler_qa_hotpot | 0.7188 |
| ruler_qa_squad | 0.6615 |
| ruler_vt | 1.0000 |

## Kimi-K2.5模型 - mmlu_pro任务子数据集详细比对结果

| Item | metax_c550(INT4) | 差值 | 百分比 |
|------|------|------|------|
| biology | 0.9358 | N/A | N/A |
| business | 0.8657 | N/A | N/A |
| chemistry | 0.8587 | N/A | N/A |
| computer_science | 0.8780 | N/A | N/A |
| economics | 0.8614 | N/A | N/A |
| engineering | 0.7668 | N/A | N/A |
| health | 0.7958 | N/A | N/A |
| history | 0.7113 | N/A | N/A |
| law | 0.6213 | N/A | N/A |
| math | 0.9275 | N/A | N/A |
| other | 0.7965 | N/A | N/A |
| philosophy | 0.8096 | N/A | N/A |
| physics | 0.8645 | N/A | N/A |
| psychology | 0.8333 | N/A | N/A |

## Kimi-K2.5模型 - ruler任务子数据集详细比对结果

| Item | metax_c550(INT4) | 差值 | 百分比 |
|------|------|------|------|
| niah_multikey_1 | 1.0000 | N/A | N/A |
| niah_multikey_2 | 1.0000 | N/A | N/A |
| niah_multikey_3 | 1.0000 | N/A | N/A |
| niah_multiquery | 1.0000 | N/A | N/A |
| niah_multivalue | 0.9922 | N/A | N/A |
| niah_single_1 | 1.0000 | N/A | N/A |
| niah_single_2 | 1.0000 | N/A | N/A |
| niah_single_3 | 1.0000 | N/A | N/A |
| ruler_cwe | 0.9719 | N/A | N/A |
| ruler_fwe | 0.9896 | N/A | N/A |
| ruler_qa_hotpot | 0.8438 | N/A | N/A |
| ruler_qa_squad | 0.7760 | N/A | N/A |
| ruler_vt | 1.0000 | N/A | N/A |

## MiniMax-M2.5模型 - mmlu_pro任务子数据集详细比对结果

| Item | nvidia_h100(FP8) | kunlun_p800(W8A8-INT8-Dynamic) | 差值 | 百分比 | metax_c550(W8A8) | 差值 | 百分比 |
|------|------|------|------|------|------|------|------|
| biology | 0.8703 | 0.8006 | -0.0697 | - 8.01% | 0.8173 | -0.0530 | - 6.09% |
| business | 0.8238 | 0.7947 | -0.0291 | - 3.53% | 0.7997 | -0.0241 | - 2.93% |
| chemistry | 0.7836 | 0.5565 | -0.2271 | - 28.98% | 0.5857 | -0.1979 | - 25.26% |
| computer_science | 0.8049 | 0.7537 | -0.0512 | - 6.36% | 0.7341 | -0.0708 | - 8.80% |
| economics | 0.7974 | 0.7500 | -0.0474 | - 5.94% | 0.7701 | -0.0273 | - 3.42% |
| engineering | 0.5851 | 0.5882 | 0.0031 | + 0.53% | 0.5686 | -0.0165 | - 2.82% |
| health | 0.7702 | 0.6932 | -0.0770 | - 10.00% | 0.6675 | -0.1027 | - 13.33% |
| history | 0.6115 | 0.5853 | -0.0262 | - 4.28% | 0.5591 | -0.0524 | - 8.57% |
| law | 0.4759 | 0.4732 | -0.0027 | - 0.57% | 0.4759 | 0.0000 | + 0.00% |
| math | 0.8312 | 0.6425 | -0.1887 | - 22.70% | 0.6758 | -0.1554 | - 18.70% |
| other | 0.7240 | 0.6526 | -0.0714 | - 9.86% | 0.7045 | -0.0195 | - 2.69% |
| philosophy | 0.6814 | 0.5691 | -0.1123 | - 16.48% | 0.5671 | -0.1143 | - 16.77% |
| physics | 0.7691 | 0.7506 | -0.0185 | - 2.41% | 0.7375 | -0.0316 | - 4.11% |
| psychology | 0.7870 | 0.7318 | -0.0552 | - 7.01% | 0.7206 | -0.0664 | - 8.44% |

## MiniMax-M2.5模型 - ruler任务子数据集详细比对结果

| Item | nvidia_h100(FP8) | hygon_bw1000(W8A8) | 差值 | 百分比 | kunlun_p800(W8A8-INT8-Dynamic) | 差值 | 百分比 | metax_c550(W8A8) | 差值 | 百分比 |
|------|------|------|------|------|------|------|------|------|------|------|
| niah_multikey_1 | 0.3125 | N/A | N/A | N/A | N/A | N/A | N/A | 0.9688 | 0.6563 | + 210.02% |
| niah_multikey_2 | 0.7812 | N/A | N/A | N/A | N/A | N/A | N/A | 1.0000 | 0.2188 | + 28.01% |
| niah_multikey_3 | 0.5312 | N/A | N/A | N/A | N/A | N/A | N/A | 1.0000 | 0.4688 | + 88.25% |
| niah_multiquery | 0.2969 | N/A | N/A | N/A | N/A | N/A | N/A | 0.9844 | 0.6875 | + 231.56% |
| niah_multivalue | 0.1484 | N/A | N/A | N/A | N/A | N/A | N/A | 0.9531 | 0.8047 | + 542.25% |
| niah_single_1 | 0.3438 | N/A | N/A | N/A | N/A | N/A | N/A | 1.0000 | 0.6562 | + 190.87% |
| niah_single_2 | 0.2812 | N/A | N/A | N/A | N/A | N/A | N/A | 1.0000 | 0.7188 | + 255.62% |
| niah_single_3 | 0.4375 | N/A | N/A | N/A | N/A | N/A | N/A | 1.0000 | 0.5625 | + 128.57% |
| ruler_cwe | 0.6281 | N/A | N/A | N/A | N/A | N/A | N/A | 0.5312 | -0.0969 | - 15.43% |
| ruler_fwe | 0.8958 | N/A | N/A | N/A | N/A | N/A | N/A | 0.9479 | 0.0521 | + 5.82% |
| ruler_qa_hotpot | 0.7500 | N/A | N/A | N/A | N/A | N/A | N/A | 0.6562 | -0.0938 | - 12.51% |
| ruler_qa_squad | 0.6927 | N/A | N/A | N/A | N/A | N/A | N/A | 0.6224 | -0.0703 | - 10.15% |
| ruler_vt | 1.0000 | N/A | N/A | N/A | N/A | N/A | N/A | 1.0000 | 0.0000 | + 0.00% |
