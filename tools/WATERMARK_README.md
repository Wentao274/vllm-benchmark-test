# 水印工具使用说明

## 工具说明

`add_watermark_print.py` - 为HTML文件添加打印友好的水印

**水印覆盖范围：**
- 页面水印：居中 + 四角（覆盖所有内容，包括代码片段）
- 图片水印：每个图片独立添加
- 表格水印：每个表格独立添加

## 使用方法

```bash
python tools/add_watermark_print.py <HTML文件> --text "水印文字"
```

### 参数说明

| 参数           | 说明           | 默认值   |
|--------------|--------------|-------|
| html_file    | HTML文件路径（必填） | -     |
| --text, -t   | 水印文字         | 九章云极  |
| --output, -o | 输出文件路径       | 覆盖原文件 |

### 示例

```bash
# 基本用法
python tools/add_watermark_print.py "report.html"

# 指定水印文字
python tools/add_watermark_print.py "report.html" --text "九章云极"

# 指定输出文件
python tools/add_watermark_print.py "input.html" --output "output.html"
```

## 水印样式

### 字体大小
- 页面水印：屏幕 60px / 打印 50px
- 图片水印：屏幕 35px / 打印 30px
- 表格水印：屏幕 45px / 打印 40px

### 透明度
- 页面水印：屏幕 0.12 / 打印 0.15
- 图片水印：屏幕 0.25 / 打印 0.3
- 表格水印：屏幕 0.15 / 打印 0.2

## 查看效果

1. 浏览器打开HTML文件
2. 按 `Ctrl+P` 打印预览
3. 检查水印是否覆盖所有内容
4. 导出PDF查看最终效果

## 注意事项

1. 打印时启用"背景图形"选项
2. 使用浏览器"另存为PDF"导出PDF
3. 代码片段保持原格式，由页面水印覆盖
4. 工具会直接修改原文件，建议先备份
