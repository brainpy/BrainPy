# Copyright 2024 Brain Simulation Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


import glob
import json
import os
import sys


def fix_ipython2_lexer_in_notebooks(directory_path):
    """
    批量修复指定目录中所有 Jupyter Notebook 文件的 ipython2 lexer 问题
    """
    # 查找所有.ipynb文件
    notebook_files = glob.glob(os.path.join(directory_path, "*.ipynb"))

    if not notebook_files:
        print(f"在目录 {directory_path} 中未找到任何 .ipynb 文件")
        return

    fixed_count = 0

    for file_path in notebook_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            needs_fix = False

            # 检查并修复顶层元数据
            if 'metadata' in data:
                # 修复 language_info
                if 'language_info' in data['metadata']:
                    lang_info = data['metadata']['language_info']
                    if lang_info.get('name') == 'ipython2':
                        lang_info['name'] = 'ipython3'
                        needs_fix = True
                        print(f"修复 {os.path.basename(file_path)}: 顶层 language_info.name")

                    if lang_info.get('pygments_lexer') == 'ipython2':
                        lang_info['pygments_lexer'] = 'ipython3'
                        needs_fix = True
                        print(f"修复 {os.path.basename(file_path)}: 顶层 language_info.pygments_lexer")

                # 修复 kernelspec
                if 'kernelspec' in data['metadata']:
                    kernelspec = data['metadata']['kernelspec']
                    if kernelspec.get('language') == 'ipython2':
                        kernelspec['language'] = 'python'
                        needs_fix = True
                        print(f"修复 {os.path.basename(file_path)}: 顶层 kernelspec.language")

                    if kernelspec.get('name') == 'ipython2':
                        kernelspec['name'] = 'python3'
                        needs_fix = True
                        print(f"修复 {os.path.basename(file_path)}: 顶层 kernelspec.name")

            # 检查并修复单元格元数据
            for i, cell in enumerate(data.get('cells', [])):
                if 'metadata' in cell:
                    # 修复单元格级别的语言设置
                    if 'language' in cell['metadata'] and cell['metadata']['language'] == 'ipython2':
                        cell['metadata']['language'] = 'ipython3'
                        needs_fix = True
                        print(f"修复 {os.path.basename(file_path)}: 单元格 {i} 的语言设置")

                    # 修复其他可能的 lexer 设置
                    if 'pygments_lexer' in cell['metadata'] and cell['metadata']['pygments_lexer'] == 'ipython2':
                        cell['metadata']['pygments_lexer'] = 'ipython3'
                        needs_fix = True
                        print(f"修复 {os.path.basename(file_path)}: 单元格 {i} 的 pygments_lexer 设置")

            # 如果需要修复，保存文件
            if needs_fix:
                # 创建备份
                backup_path = file_path + '.backup'
                with open(backup_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)

                # 保存修复后的文件
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)

                fixed_count += 1
                print(f"已修复并备份: {os.path.basename(file_path)}")
            else:
                print(f"无需修复: {os.path.basename(file_path)}")

        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {str(e)}")

    print(f"\n处理完成! 共修复了 {fixed_count} 个文件")
    return fixed_count


if __name__ == "__main__":
    import os
    print(os.path.dirname(os.path.abspath(__file__)))

    # 使用当前目录，或者指定您的文档目录路径
    target_directory = input("请输入包含.ipynb文件的目录路径(直接回车使用当前目录): ").strip()

    if not target_directory:
        target_directory = "."

    if not os.path.isdir(target_directory):
        print(f"错误: 目录 '{target_directory}' 不存在")
        sys.exit(1)

    print(f"开始处理目录: {os.path.abspath(target_directory)}")
    fix_ipython2_lexer_in_notebooks(target_directory)
