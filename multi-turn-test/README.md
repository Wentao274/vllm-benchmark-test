# multi-turn-test
multi turn test for maas


## 方式一：使用配置文件（推荐）

### 1. 编辑配置文件 config.yaml

```yaml
model: "/data/models/MiniMax-M2.5-W8A8"
served_model_name: "minimax-m2.5"
url: "http://127.0.0.1:8080/v1"
input_file: "generate_conversations.json"
output_file: "generate_conversations_output.json"
num_clients: 50
max_active_conversations: 100
warmup_step: true
extra_body_json:
  chat_template_kwargs:
    enable_thinking: false
```

### 2. 运行测试

```bash
python run_multi_turn.py
```

### 其他选项

```bash
# 指定配置文件
python run_multi_turn.py -c my_config.yaml

# 预览命令不执行
python run_multi_turn.py --dry-run
```


## 方式二：直接运行脚本

```bash
python benchmark_serving_multi_turn.py \
--model /data/models/MiniMax-M2.5-W8A8 \
--served-model-name "minimax-m2.5" \
--url "http://127.0.0.1:8080/v1" \
--input-file "generate_conversations.json" \
--output-file "generate_conversations_output.json" \
--num-clients 50 \
--max-active-conversations 100 \
--warmup-step \
--extra-body-json '{"chat_template_kwargs":{"enable_thinking":false}}'
```