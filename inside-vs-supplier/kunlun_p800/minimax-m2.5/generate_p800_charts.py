import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

supplier_data = {
    1: {128: None, 256: None, 512: None, 1024: 89.01, 2048: 89.05},
    4: {128: 299.92, 256: 314.34, 512: 332.86, 1024: 330.46, 2048: 336.59},
    8: {128: 533.47, 256: 557.83, 512: 591.31, 1024: 583.25, 2048: 586.64},
    16: {128: 955.40, 256: 1036.62, 512: 1087.22, 1024: 1098.82, 2048: 1067.90},
    32: {128: 1675.04, 256: 1836.49, 512: 2036.83, 1024: 2028.13, 2048: 1918.50}
}
supplier_ttft = {
    1: {128: None, 256: None, 512: None, 1024: 173.68, 2048: 199.33},
    4: {128: 318.48, 256: 323.92, 512: 311.47, 1024: 323.81, 2048: 326.93},
    8: {128: 336.27, 256: 356.30, 512: 344.08, 1024: 344.63, 2048: 328.55},
    16: {128: 357.31, 256: 367.73, 512: 366.98, 1024: 369.05, 2048: 398.40},
    32: {128: 464.82, 256: 469.09, 512: 480.84, 1024: 501.59, 2048: 537.22}
}
supplier_tpot = {
    1: {128: None, 256: None, 512: None, 1024: 22.09, 2048: 22.37},
    4: {128: 24.33, 256: 24.25, 512: 23.46, 1024: 23.91, 2048: 23.62},
    8: {128: 27.53, 256: 27.37, 512: 26.43, 1024: 27.12, 2048: 27.12},
    16: {128: 30.89, 256: 29.52, 512: 28.76, 1024: 28.78, 2048: 29.78},
    32: {128: 34.78, 256: 33.11, 512: 30.52, 1024: 31.09, 2048: 33.11}
}

internal_data = {
    1: {128: 99.40, 256: 101.17, 512: 101.58, 1024: 100.69, 2048: 98.24},
    4: {128: 338.52, 256: 360.74, 512: 373.16, 1024: 365.97, 2048: 366.51},
    8: {128: 544.17, 256: 589.82, 512: 613.72, 1024: 615.82, 2048: 606.87},
    16: {128: 700.75, 256: 1082.95, 512: 1140.18, 1024: 1134.16, 2048: 1101.30},
    32: {128: 1720.53, 256: 1908.08, 512: 2064.01, 1024: 2042.86, 2048: 1995.83}
}
internal_ttft = {
    1: {128: 128.34, 256: 131.11, 512: 133.90, 1024: 133.50, 2048: 143.02},
    4: {128: 227.20, 256: 229.39, 512: 230.28, 1024: 236.49, 2048: 245.06},
    8: {128: 251.62, 256: 259.37, 512: 260.37, 1024: 272.82, 2048: 284.84},
    16: {128: 1974.08, 256: 315.65, 512: 326.59, 1024: 323.06, 2048: 411.08},
    32: {128: 472.10, 256: 478.05, 512: 497.48, 1024: 526.74, 2048: 586.90}
}
internal_tpot = {
    1: {128: 19.26, 256: 19.33, 512: 19.46, 1024: 19.75, 2048: 20.30},
    4: {128: 21.96, 256: 21.33, 512: 21.01, 1024: 21.64, 2048: 21.71},
    8: {128: 27.58, 256: 26.18, 512: 25.59, 1024: 25.73, 2048: 26.23},
    16: {128: 30.38, 256: 28.38, 512: 27.46, 1024: 27.91, 2048: 28.86},
    32: {128: 33.61, 256: 31.72, 512: 30.05, 1024: 30.82, 2048: 31.79}
}

input_lens = [128, 256, 512, 1024, 2048]
request_counts = [1, 4, 8, 16, 32]

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

for req in request_counts:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'Request={req} Comparison', fontsize=14, fontweight='bold')

    supplier_vals = [supplier_data[req].get(l) for l in input_lens]
    internal_vals = [internal_data[req].get(l) for l in input_lens]
    axes[0].plot(input_lens, supplier_vals, 'o-', label='Kunlun Supplier', linewidth=2, markersize=8)
    axes[0].plot(input_lens, internal_vals, 's--', label='Internal', linewidth=2, markersize=8)
    axes[0].set_xlabel('Input Length')
    axes[0].set_ylabel('Throughput (token/s)')
    axes[0].set_title('Throughput')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    for i, (sv, iv) in enumerate(zip(supplier_vals, internal_vals)):
        if sv is not None:
            axes[0].annotate(f'{sv:.2f}', (input_lens[i], sv), textcoords="offset points", xytext=(0,8), ha='center', fontsize=8)
        if iv is not None:
            axes[0].annotate(f'{iv:.2f}', (input_lens[i], iv), textcoords="offset points", xytext=(0,-12), ha='center', fontsize=8)
    axes[0].set_xticks(input_lens)

    supplier_ttft_vals = [supplier_ttft[req].get(l) for l in input_lens]
    internal_ttft_vals = [internal_ttft[req].get(l) for l in input_lens]
    axes[1].plot(input_lens, supplier_ttft_vals, 'o-', label='Kunlun Supplier', linewidth=2, markersize=8)
    axes[1].plot(input_lens, internal_ttft_vals, 's--', label='Internal', linewidth=2, markersize=8)
    axes[1].set_xlabel('Input Length')
    axes[1].set_ylabel('TTFT (ms)')
    axes[1].set_title('TTFT')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    for i, (sv, iv) in enumerate(zip(supplier_ttft_vals, internal_ttft_vals)):
        if sv is not None:
            axes[1].annotate(f'{sv:.2f}', (input_lens[i], sv), textcoords="offset points", xytext=(0,8), ha='center', fontsize=8)
        if iv is not None:
            axes[1].annotate(f'{iv:.2f}', (input_lens[i], iv), textcoords="offset points", xytext=(0,-12), ha='center', fontsize=8)
    axes[1].set_xticks(input_lens)

    supplier_tpot_vals = [supplier_tpot[req].get(l) for l in input_lens]
    internal_tpot_vals = [internal_tpot[req].get(l) for l in input_lens]
    axes[2].plot(input_lens, supplier_tpot_vals, 'o-', label='Kunlun Supplier', linewidth=2, markersize=8)
    axes[2].plot(input_lens, internal_tpot_vals, 's--', label='Internal', linewidth=2, markersize=8)
    axes[2].set_xlabel('Input Length')
    axes[2].set_ylabel('TPOT (ms)')
    axes[2].set_title('TPOT')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    for i, (sv, iv) in enumerate(zip(supplier_tpot_vals, internal_tpot_vals)):
        if sv is not None:
            axes[2].annotate(f'{sv:.2f}', (input_lens[i], sv), textcoords="offset points", xytext=(0,8), ha='center', fontsize=8)
        if iv is not None:
            axes[2].annotate(f'{iv:.2f}', (input_lens[i], iv), textcoords="offset points", xytext=(0,-12), ha='center', fontsize=8)
    axes[2].set_xticks(input_lens)

    plt.tight_layout()
    plt.savefig(f'compare_req{req}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Generated compare_req{req}.png')

print('All figures generated!')