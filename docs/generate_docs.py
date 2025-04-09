 #!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
生成LLMGraph中所有函数的文档
"""

import os
import sys
import pydoc
import inspect
from pathlib import Path

def generate_docs_for_module(module_path, output_dir):
    """
    为指定模块生成文档
    
    Args:
        module_path: 模块路径
        output_dir: 输出目录
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取模块名称
    module_name = os.path.basename(module_path).replace('.py', '')
    
    # 创建输出文件路径
    output_file = os.path.join(output_dir, f"{module_name}.md")
    
    # 导入模块
    try:
        # 将模块路径添加到sys.path
        module_dir = os.path.dirname(module_path)
        if module_dir not in sys.path:
            sys.path.insert(0, module_dir)
        
        # 导入模块
        module = __import__(module_name)
        
        # 生成文档
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"# {module_name} 模块文档\n\n")
            
            # 模块文档
            if module.__doc__:
                f.write(f"## 模块描述\n\n{module.__doc__}\n\n")
            
            # 类文档
            f.write("## 类\n\n")
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if obj.__module__ == module.__name__:
                    f.write(f"### {name}\n\n")
                    if obj.__doc__:
                        f.write(f"{obj.__doc__}\n\n")
                    
                    # 方法文档
                    f.write("#### 方法\n\n")
                    for method_name, method in inspect.getmembers(obj, inspect.isfunction):
                        if method.__module__ == module.__name__:
                            f.write(f"##### {method_name}\n\n")
                            if method.__doc__:
                                f.write(f"{method.__doc__}\n\n")
                            else:
                                f.write("无文档\n\n")
            
            # 函数文档
            f.write("## 函数\n\n")
            for name, obj in inspect.getmembers(module, inspect.isfunction):
                if obj.__module__ == module.__name__:
                    f.write(f"### {name}\n\n")
                    if obj.__doc__:
                        f.write(f"{obj.__doc__}\n\n")
                    else:
                        f.write("无文档\n\n")
        
        print(f"已生成 {output_file} 的文档")
    except Exception as e:
        print(f"生成 {module_path} 的文档时出错: {e}")

def generate_all_docs(llmgraph_dir, output_dir):
    """
    为LLMGraph目录中的所有Python文件生成文档
    
    Args:
        llmgraph_dir: LLMGraph目录路径
        output_dir: 输出目录
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 遍历LLMGraph目录中的所有Python文件
    for root, dirs, files in os.walk(llmgraph_dir):
        for file in files:
            if file.endswith('.py') and not file.startswith('__'):
                module_path = os.path.join(root, file)
                relative_path = os.path.relpath(module_path, llmgraph_dir)
                relative_dir = os.path.dirname(relative_path)
                
                # 创建对应的输出目录
                module_output_dir = os.path.join(output_dir, relative_dir)
                os.makedirs(module_output_dir, exist_ok=True)
                
                # 生成文档
                generate_docs_for_module(module_path, module_output_dir)
    
    # 生成索引文件
    with open(os.path.join(output_dir, 'index.md'), 'w', encoding='utf-8') as f:
        f.write("# LLMGraph 文档索引\n\n")
        
        # 遍历输出目录中的所有.md文件
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                if file.endswith('.md') and file != 'index.md':
                    relative_path = os.path.relpath(os.path.join(root, file), output_dir)
                    module_name = os.path.basename(file).replace('.md', '')
                    f.write(f"- [{module_name}]({relative_path})\n")
    
    print(f"已生成所有文档，索引文件位于 {os.path.join(output_dir, 'index.md')}")

if __name__ == "__main__":
    # 获取当前目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # LLMGraph目录路径
    llmgraph_dir = os.path.join(current_dir, "LLMGraph")
    
    # 输出目录
    output_dir = os.path.join(current_dir, "docs")
    
    # 生成文档
    generate_all_docs(llmgraph_dir, output_dir)