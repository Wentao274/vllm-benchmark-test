#!/usr/bin/env python3
"""
为HTML添加打印友好水印 - 确保PDF中每页都有完整水印
解决：
1. 水印在PDF中溢出问题
2. 图片上显示水印
3. 每页都有水印
"""

import os
import sys
import re


def add_print_friendly_watermark(
    html_file, watermark_text="九章云极", output_file=None
):
    """
    添加打印友好的水印
    """
    with open(html_file, "r", encoding="utf-8") as f:
        content = f.read()

    # 移除旧的水印代码
    content = re.sub(r"<style>.*?watermark.*?</style>", "", content, flags=re.DOTALL)
    content = re.sub(r'<div class="watermark.*?</div>', "", content, flags=re.DOTALL)
    content = re.sub(r'<div class="page-footer.*?</div>', "", content, flags=re.DOTALL)
    content = re.sub(r"<!-- 水印元素.*?-->", "", content, flags=re.DOTALL)

    # 新的水印CSS - 优化打印效果
    watermark_css = f'''
<style>
/* ========== 水印系统 ========== */

/* 页面水印容器 - 每页显示 */
@page {{
    margin: 15mm;
    @top-center {{
        content: "{watermark_text}";
        font-size: 10pt;
        color: rgba(180, 180, 180, 0.4);
        font-family: "Microsoft YaHei", Arial, sans-serif;
    }}
    @bottom-center {{
        content: "{watermark_text}";
        font-size: 10pt;
        color: rgba(180, 180, 180, 0.4);
        font-family: "Microsoft YaHei", Arial, sans-serif;
    }}
}}

/* 屏幕显示水印 - 固定在页面中央，确保完整显示，覆盖所有内容 */
.screen-watermark {{
    position: fixed !important;
    top: 50% !important;
    left: 50% !important;
    transform: translate(-50%, -50%) rotate(-30deg) !important;
    font-size: 60px !important;
    font-weight: bold !important;
    color: rgba(200, 200, 200, 0.12) !important;
    pointer-events: none !important;
    z-index: 99999 !important;
    white-space: nowrap !important;
    font-family: "Microsoft YaHei", "SimHei", Arial, sans-serif !important;
    letter-spacing: 10px !important;
}}

/* 打印时的水印 - 使用绝对定位确保不溢出，覆盖所有内容 */
@media print {{
    .screen-watermark {{
        position: fixed !important;
        font-size: 50px !important;
        color: rgba(180, 180, 180, 0.15) !important;
        z-index: 99999 !important;
        transform: translate(-50%, -50%) rotate(-30deg) !important;
    }}
    
    /* 打印时为每个图片添加水印 */
    .image-watermark {{
        position: absolute !important;
        top: 50% !important;
        left: 50% !important;
        transform: translate(-50%, -50%) rotate(-30deg) !important;
        font-size: 30px !important;
        font-weight: bold !important;
        color: rgba(180, 180, 180, 0.3) !important;
        pointer-events: none !important;
        z-index: 9999 !important;
        white-space: nowrap !important;
        font-family: "Microsoft YaHei", Arial, sans-serif !important;
    }}
    
    /* 确保图片容器支持水印定位 */
    .image-container {{
        position: relative !important;
    }}
}}

/* 图片水印容器 */
.image-container {{
    position: relative !important;
    display: inline-block !important;
    width: 100% !important;
}}

/* 图片上的水印 - 屏幕显示 */
.image-watermark {{
    position: absolute !important;
    top: 50% !important;
    left: 50% !important;
    transform: translate(-50%, -50%) rotate(-30deg) !important;
    font-size: 35px !important;
    font-weight: bold !important;
    color: rgba(200, 200, 200, 0.25) !important;
    pointer-events: none !important;
    z-index: 9999 !important;
    white-space: nowrap !important;
    font-family: "Microsoft YaHei", Arial, sans-serif !important;
    letter-spacing: 8px !important;
}}

/* 确保图片正常显示 */
.image-container img {{
    width: 100% !important;
    height: auto !important;
    display: block !important;
}}

/* 表格水印容器 */
.table-container {{
    position: relative !important;
    display: block !important;
    width: 100% !important;
    margin: 1em 0 !important;
}}

/* 表格上的水印 - 屏幕显示 */
.table-watermark {{
    position: absolute !important;
    top: 50% !important;
    left: 50% !important;
    transform: translate(-50%, -50%) rotate(-30deg) !important;
    font-size: 45px !important;
    font-weight: bold !important;
    color: rgba(200, 200, 200, 0.15) !important;
    pointer-events: none !important;
    z-index: 9999 !important;
    white-space: nowrap !important;
    font-family: "Microsoft YaHei", Arial, sans-serif !important;
    letter-spacing: 8px !important;
}}

/* 确保表格正常显示 */
.table-container table {{
    width: 100% !important;
    border-collapse: collapse !important;
}}

/* 打印时表格水印 */
@media print {{
    .table-watermark {{
        position: absolute !important;
        font-size: 40px !important;
        color: rgba(180, 180, 180, 0.2) !important;
        z-index: 9999 !important;
    }}
}}

/* 页面四个角的水印 - 确保不溢出 */
.corner-watermark {{
    position: fixed !important;
    font-size: 14px !important;
    color: rgba(180, 180, 180, 0.3) !important;
    pointer-events: none !important;
    z-index: 9999 !important;
    font-family: "Microsoft YaHei", Arial, sans-serif !important;
}}

.corner-watermark.top-left {{
    top: 20px !important;
    left: 20px !important;
}}

.corner-watermark.top-right {{
    top: 20px !important;
    right: 20px !important;
}}

.corner-watermark.bottom-left {{
    bottom: 20px !important;
    left: 20px !important;
}}

.corner-watermark.bottom-right {{
    bottom: 20px !important;
    right: 20px !important;
}}

@media print {{
    .corner-watermark {{
        position: absolute !important;
        font-size: 12px !important;
        color: rgba(180, 180, 180, 0.4) !important;
    }}
}}
</style>

'''

    # 在</head>前插入CSS
    if "</head>" in content:
        content = content.replace("</head>", watermark_css + "</head>")

    # 水印HTML元素
    watermark_html = f"""
<!-- 水印元素 -->
<div class="screen-watermark">{watermark_text}</div>
<div class="corner-watermark top-left">{watermark_text}</div>
<div class="corner-watermark top-right">{watermark_text}</div>
<div class="corner-watermark bottom-left">{watermark_text}</div>
<div class="corner-watermark bottom-right">{watermark_text}</div>
"""

    # 在<body>标签后插入水印
    body_match = re.search(r"<body[^>]*>", content)
    if body_match:
        insert_pos = body_match.end()
        content = content[:insert_pos] + watermark_html + content[insert_pos:]

    # 为所有图片添加水印容器
    # 查找所有<img>标签并包裹在容器中
    img_pattern = r'(<img\s+[^>]*src=["\'][^"\']*["\'][^>]*(?:/>|>))'

    def wrap_image_with_watermark(match):
        img_tag = match.group(1)
        # 如果图片已经在容器中，跳过
        return f"""<div class="image-container">
{img_tag}
<div class="image-watermark">{watermark_text}</div>
</div>"""

    content = re.sub(img_pattern, wrap_image_with_watermark, content)

    # 为所有表格添加水印容器
    # 查找所有<table>标签并包裹在容器中
    table_pattern = r"(<table[^>]*>.*?</table>)"

    def wrap_table_with_watermark(match):
        table_tag = match.group(1)
        # 如果表格已经在容器中，跳过
        if "table-container" in table_tag:
            return table_tag
        return f"""<div class="table-container">
{table_tag}
<div class="table-watermark">{watermark_text}</div>
</div>"""

    content = re.sub(table_pattern, wrap_table_with_watermark, content, flags=re.DOTALL)

    # 不为代码片段添加单独水印容器
    # 页面水印会覆盖所有内容（包括代码片段），不影响代码格式

    # 保存文件
    output = output_file if output_file else html_file
    with open(output, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"已添加打印友好水印到: {output}")
    print(f"  - 页面水印: 居中 + 四角（覆盖所有内容，包括代码片段）")
    print(f"  - 图片水印: 每个图片都有")
    print(f"  - 表格水印: 每个表格都有")
    print(f"  - 打印优化: 确保PDF中完整显示，不影响代码格式")
    return output


def main():
    import argparse

    parser = argparse.ArgumentParser(description="为HTML添加打印友好水印")
    parser.add_argument("html_file", help="HTML文件路径")
    parser.add_argument("--text", "-t", default="九章云极", help="水印文字")
    parser.add_argument("--output", "-o", help="输出文件路径")

    args = parser.parse_args()

    if not os.path.exists(args.html_file):
        print(f"错误: 文件不存在: {args.html_file}")
        sys.exit(1)

    add_print_friendly_watermark(args.html_file, args.text, args.output)


if __name__ == "__main__":
    main()
