import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

supplier_total = {1: 2412.85, 2: 3388.17, 4: 5324.75, 8: 7122.15, 10: 10807.82, 16: 6145.69, 32: 4409.89, 64: 2979.47, 128: 2884.52}
supplier_gen = {1: 50.62, 2: 71.08, 4: 111.71, 8: 149.42, 10: 226.74, 16: 128.93, 32: 92.52, 64: 62.51, 128: 60.51}
supplier_ttft = {1: 3776.46, 2: 5969.72, 4: 4342.18, 8: 6735.45, 10: 1553.54, 16: 9903.80, 32: 164584.73, 64: 649448, 128: 1522083.76}
supplier_tpot = {1: 17.25, 2: 24.13, 4: 32.84, 8: 48.98, 10: 42.86, 16: 93.61, 32: 115.04, 64: 236.12, 128: 221.69}

internal_total = {1: 2914.01, 2: 2907.96, 4: 2441.07, 8: 2442.06, 16: 2442.64, 32: 2101.2, 64: 2097.38, 128: 2097.83}
internal_gen = {1: 61.13, 2: 61.01, 4: 51.21, 8: 51.23, 16: 51.24, 32: 44.08, 64: 44, 128: 44.01}
internal_ttft = {1: 500.63, 2: 544.87, 4: 5259.96, 8: 5258.44, 16: 5261.4, 32: 10059.87, 64: 10061.06, 128: 10053.8}
internal_tpot = {1: 16.03, 2: 16.04, 4: 16.03, 8: 16.02, 16: 16.02, 32: 15.99, 64: 16.03, 128: 16.03}

batches = [1, 2, 4, 8, 10, 16, 32, 64, 128]
batches_internal = [1, 2, 4, 8, 16, 32, 64, 128]

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('BW1000 Comparison: Supplier vs Internal', fontsize=14, fontweight='bold')

supplier_vals = [supplier_total.get(b) for b in batches]
internal_vals = [internal_total.get(b) for b in batches]
axes[0, 0].plot(batches, supplier_vals, 'o-', label='Supplier', linewidth=2, markersize=8)
axes[0, 0].plot(batches_internal, [internal_total[b] for b in batches_internal], 's--', label='Internal', linewidth=2, markersize=8)
axes[0, 0].set_xlabel('Batch')
axes[0, 0].set_ylabel('TOTAL THROUGHPUT (toks/s)')
axes[0, 0].set_title('TOTAL THROUGHPUT')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)
for i, (b, sv) in enumerate(zip(batches, supplier_vals)):
    if sv is not None:
        axes[0, 0].annotate(f'{sv:.2f}', (b, sv), textcoords="offset points", xytext=(0,8), ha='center', fontsize=7)
for b in batches_internal:
    axes[0, 0].annotate(f'{internal_total[b]:.2f}', (b, internal_total[b]), textcoords="offset points", xytext=(0,-10), ha='center', fontsize=7)
axes[0, 0].set_xscale('log')
axes[0, 0].set_xticks(batches)
axes[0, 0].set_xticklabels(batches)

supplier_vals = [supplier_gen.get(b) for b in batches]
internal_vals = [internal_gen.get(b) for b in batches]
axes[0, 1].plot(batches, supplier_vals, 'o-', label='Supplier', linewidth=2, markersize=8)
axes[0, 1].plot(batches_internal, [internal_gen[b] for b in batches_internal], 's--', label='Internal', linewidth=2, markersize=8)
axes[0, 1].set_xlabel('Batch')
axes[0, 1].set_ylabel('Generate Throughput (toks/s)')
axes[0, 1].set_title('Generate Throughput')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)
for i, (b, sv) in enumerate(zip(batches, supplier_vals)):
    if sv is not None:
        axes[0, 1].annotate(f'{sv:.2f}', (b, sv), textcoords="offset points", xytext=(0,8), ha='center', fontsize=7)
for b in batches_internal:
    axes[0, 1].annotate(f'{internal_gen[b]:.2f}', (b, internal_gen[b]), textcoords="offset points", xytext=(0,-10), ha='center', fontsize=7)
axes[0, 1].set_xscale('log')
axes[0, 1].set_xticks(batches)
axes[0, 1].set_xticklabels(batches)

supplier_vals = [supplier_ttft.get(b) for b in batches]
internal_vals = [internal_ttft.get(b) for b in batches]
axes[1, 0].plot(batches, supplier_vals, 'o-', label='Supplier', linewidth=2, markersize=8)
axes[1, 0].plot(batches_internal, [internal_ttft[b] for b in batches_internal], 's--', label='Internal', linewidth=2, markersize=8)
axes[1, 0].set_xlabel('Batch')
axes[1, 0].set_ylabel('TTFT (ms)')
axes[1, 0].set_title('TTFT')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)
for i, (b, sv) in enumerate(zip(batches, supplier_vals)):
    if sv is not None:
        axes[1, 0].annotate(f'{sv:.2f}', (b, sv), textcoords="offset points", xytext=(0,8), ha='center', fontsize=7)
for b in batches_internal:
    axes[1, 0].annotate(f'{internal_ttft[b]:.2f}', (b, internal_ttft[b]), textcoords="offset points", xytext=(0,-10), ha='center', fontsize=7)
axes[1, 0].set_xscale('log')
axes[1, 0].set_xticks(batches)
axes[1, 0].set_xticklabels(batches)

supplier_vals = [supplier_tpot.get(b) for b in batches]
internal_vals = [internal_tpot.get(b) for b in batches]
axes[1, 1].plot(batches, supplier_vals, 'o-', label='Supplier', linewidth=2, markersize=8)
axes[1, 1].plot(batches_internal, [internal_tpot[b] for b in batches_internal], 's--', label='Internal', linewidth=2, markersize=8)
axes[1, 1].set_xlabel('Batch')
axes[1, 1].set_ylabel('TPOT (ms)')
axes[1, 1].set_title('TPOT')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)
for i, (b, sv) in enumerate(zip(batches, supplier_vals)):
    if sv is not None:
        axes[1, 1].annotate(f'{sv:.2f}', (b, sv), textcoords="offset points", xytext=(0,8), ha='center', fontsize=7)
for b in batches_internal:
    axes[1, 1].annotate(f'{internal_tpot[b]:.2f}', (b, internal_tpot[b]), textcoords="offset points", xytext=(0,-10), ha='center', fontsize=7)
axes[1, 1].set_xscale('log')
axes[1, 1].set_xticks(batches)
axes[1, 1].set_xticklabels(batches)

plt.tight_layout()
plt.savefig('D:/MaaS/github/vllm-benchmark-test/inside-vs-supplier/hygon_bw1000/minimax-m2.5/bw1000_compare.png', dpi=150, bbox_inches='tight')
plt.close()
print('Generated bw1000_compare.png')