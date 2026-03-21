import os
import subprocess
import threading
import time
import re
from pathlib import Path
from datetime import datetime

class GPUMonitor:
    def __init__(self, interval=10):
        self.interval = interval
        self.monitor_process = None
        self.log_file = None
        self.monitor_cmd = "nvidia-smi"
        
    def check_gpu_tool(self):
        try:
            result = subprocess.run(["nvidia-smi", "--version"], capture_output=True, text=True)
            if result.returncode == 0:
                self.monitor_cmd = "nvidia-smi"
                return True
        except:
            pass
        try:
            result = subprocess.run(["rocm-smi", "--version"], capture_output=True, text=True)
            if result.returncode == 0:
                self.monitor_cmd = "rocm-smi"
                return True
        except:
            pass
        return False
    
    def start_monitoring(self, log_dir, model_name, param_dir):
        if not self.check_gpu_tool():
            print("Warning: No GPU monitoring tool available")
            return False
        
        monitor_dir = os.path.join(log_dir, model_name, param_dir)
        Path(monitor_dir).mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        self.log_file = os.path.join(monitor_dir, f"gpu_monitor_{timestamp}.log")
        
        header = f"Time,GPU,Used_MB,Total_MB,Utilization_%,Memory_%,Temperature_C\n"
        with open(self.log_file, 'w') as f:
            f.write(f"GPU Memory Monitor Started at {datetime.now()}\n")
            f.write(f"Log file: {self.log_file}\n")
            f.write(f"Monitoring interval: {self.interval}s\n")
            f.write(f"Using {self.monitor_cmd} for GPU monitoring\n")
            f.write("-" * 50 + "\n")
            f.write(header)
        
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        print(f"GPU monitoring started: {self.log_file}")
        return True
    
    def _monitor_loop(self):
        while self.running:
            try:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                if self.monitor_cmd == "nvidia-smi":
                    result = subprocess.run([
                        "nvidia-smi", 
                        "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,utilization.memory,temperature.gpu",
                        "--format=csv,noheader,nounits"
                    ], capture_output=True, text=True, timeout=5)
                    
                    if result.returncode == 0:
                        with open(self.log_file, 'a') as f:
                            for line in result.stdout.strip().split('\n'):
                                if line.strip():
                                    parts = [p.strip() for p in line.split(',')]
                                    if len(parts) >= 7:
                                        f.write(f"{timestamp},{','.join(parts)}\n")
                else:
                    time.sleep(self.interval)
                    
            except Exception as e:
                print(f"GPU monitoring error: {e}")
            
            time.sleep(self.interval)
    
    def stop_monitoring(self):
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        
        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write("-" * 50 + "\n")
                f.write(f"GPU Memory Monitor Stopped at {datetime.now()}\n")
        
        print(f"GPU monitoring stopped: {self.log_file}")
        return self.log_file


def parse_gpu_log(log_file):
    gpu_data = {i: [] for i in range(8)}
    
    if not os.path.exists(log_file):
        return None
    
    with open(log_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('-') or line.startswith('GPU') or 'Time,GPU' in line:
                continue
            
            parts = line.split(',')
            if len(parts) >= 7:
                try:
                    timestamp = parts[0]
                    gpu_idx = int(parts[1])
                    used_mb = float(parts[2])
                    total_mb = float(parts[3])
                    util_pct = float(parts[4])
                    mem_pct = float(parts[5])
                    temp_c = float(parts[6])
                    
                    gpu_data[gpu_idx].append({
                        'timestamp': timestamp,
                        'used_mb': used_mb,
                        'total_mb': total_mb,
                        'utilization': util_pct,
                        'memory_util': mem_pct,
                        'temperature': temp_c
                    })
                except:
                    continue
    
    return gpu_data


def generate_gpu_charts(log_file, output_dir):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping GPU chart generation")
        return
    
    gpu_data = parse_gpu_log(log_file)
    if not gpu_data:
        print("No GPU data to plot")
        return
    
    num_gpus = sum(1 for g in gpu_data if gpu_data[g])
    if num_gpus == 0:
        print("No GPU data available")
        return
    
    num_gpus = min(num_gpus, 8)
    cols = 4
    rows = (num_gpus + cols - 1) // cols
    if rows > 4:
        rows = 4
        cols = 4
    
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
    if num_gpus == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes[:num_gpus]]
    
    fig.suptitle('GPU Monitoring Trends', fontsize=14, fontweight='bold')
    
    for idx in range(num_gpus):
        row = idx // cols
        col = idx % cols
        ax = axes[row][col] if rows > 1 else axes[col]
        
        data = gpu_data.get(idx, [])
        if not data:
            ax.set_title(f'GPU {idx} (No Data)')
            continue
        
        times = range(len(data))
        used_mb = [d['used_mb'] for d in data]
        util = [d['utilization'] for d in data]
        temp = [d['temperature'] for d in data]
        
        ax.plot(times, used_mb, 'b-', label='Used (MB)', linewidth=1.5)
        ax.set_xlabel('Sample')
        ax.set_ylabel('Memory (MB)', color='b')
        ax.tick_params(axis='y', labelcolor='b')
        
        ax2 = ax.twinx()
        ax2.plot(times, util, 'r-', label='Util (%)', linewidth=1.5)
        ax2.set_ylabel('Utilization (%)', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        
        ax.set_title(f'GPU {idx}: Memory & Utilization')
        ax.grid(True, alpha=0.3)
        
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8)
    
    for idx in range(num_gpus, rows * cols):
        row = idx // cols
        col = idx % cols
        if row < len(axes) and col < len(axes[row]):
            axes[row][col].set_visible(False)
    
    plt.tight_layout()
    
    chart_file = os.path.join(output_dir, 'gpu_trends.png')
    plt.savefig(chart_file, dpi=150)
    plt.close()
    
    print(f"GPU charts generated: {chart_file}")