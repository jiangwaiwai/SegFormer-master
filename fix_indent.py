#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 读取文件
with open('tools/train.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 修复缩进
for i in range(len(lines)):
    # 修复第103行附近的缩进
    if i > 0 and lines[i-1].strip() == 'try:' and lines[i].strip().startswith('cfg.dump'):
        lines[i] = '        ' + lines[i].lstrip()
    # 修复第133行附近的缩进
    if i > 0 and lines[i-1].strip() == 'try:' and lines[i].strip().startswith('logger.info(f\'Config:'):
        lines[i] = '        ' + lines[i].lstrip()

# 写回文件
with open('tools/train.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print('修复完成！')
