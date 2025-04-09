 #!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
使用pydoc生成LLMGraph的HTML文档
"""

import os
import sys
import subprocess
import argparse

def generate_pydoc(module_name, output_dir=None):
    """
    使用pydoc生成指定模块的HTML文档
    
    Args:
        module_name: 模块名称
        output_dir: 输出目录，默认为当前目录下的docs
    """
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 构建pydoc命令
    cmd = ["pydoc", "-w", module_name]
    
   
            
    # 执行命令
    try:
        subprocess.run(cmd, check=True)
        print(f"已成功为 {module_name} 生成文档，保存在 {output_dir} 目录")
    except subprocess.CalledProcessError as e:
        print(f"生成 {module_name} 的文档时出错: {e}")
    except Exception as e:
        print(f"发生错误: {e}")

def generate_all_pydoc(llmgraph_dir, output_dir=None):
    """
    为LLMGraph目录中的所有Python模块生成文档
    
    Args:
        llmgraph_dir: LLMGraph目录路径
        output_dir: 输出目录，默认为当前目录下的docs
    """
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 将LLMGraph目录添加到Python路径
    if llmgraph_dir not in sys.path:
        sys.path.insert(0, os.path.dirname(llmgraph_dir))
    
    # 生成LLMGraph包的文档
    generate_pydoc("LLMGraph", output_dir)
    
    # 遍历LLMGraph目录中的所有Python文件
    for root, dirs, files in os.walk(llmgraph_dir):
        for file in files:
            if file.endswith('.py') and not file.startswith('__'):
                # 获取模块路径
                module_path = os.path.join(root, file)
                relative_path = os.path.relpath(module_path, os.path.dirname(llmgraph_dir))
                module_name = os.path.splitext(relative_path)[0].replace(os.sep, '.')
                
                # 生成文档
                generate_pydoc(module_name, output_dir)
                
                
     # 如果指定了输出目录，则添加-o参数并移动生成的HTML文件到输出目录

    print(f"已生成所有文档，保存在 {output_dir} 目录")

if __name__ == "__main__":

    output_dir = "LLMGraph/docs/html"    
    # 获取当前目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # LLMGraph目录路径
    llmgraph_dir = os.path.join(current_dir,"..", "LLMGraph")
    
    # evaluate目录路径
    evaluate_dir = os.path.join(current_dir,"..", "evaluate")
    
    # 生成文档
    generate_all_pydoc(llmgraph_dir, os.path.join(output_dir, "LLMGraph"))
    generate_all_pydoc(evaluate_dir, os.path.join(output_dir, "evaluate"))
    
    