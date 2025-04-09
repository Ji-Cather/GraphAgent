 #!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
生成LLMGraph中所有函数的文档，以Markdown格式输出
"""

import os
import sys
import inspect
import importlib
from pathlib import Path

def get_function_signature(func):
    """
    获取函数的签名
    
    Args:
        func: 函数对象
    
    Returns:
        函数签名字符串
    """
    try:
        return str(inspect.signature(func))
    except (ValueError, TypeError):
        return "()"

def generate_function_docs(module_path, output_file):
    """
    为指定模块中的所有函数生成文档
    
    Args:
        module_path: 模块路径
        output_file: 输出文件
    """
    # 获取模块名称
    module_name = os.path.basename(module_path).replace('.py', '')
    
    # 导入模块
    try:
        # 将模块路径添加到sys.path
        module_dir = os.path.dirname(module_path)
        if module_dir not in sys.path:
            sys.path.insert(0, module_dir)
        
        # 导入模块
        module = importlib.import_module(module_name)
        
        # 写入模块名称
        output_file.write(f"# {module_name} 模块\n\n")
        
        # 写入模块文档
        if module.__doc__:
            output_file.write(f"## 模块描述\n\n{module.__doc__}\n\n")
        
        # 写入类文档
        output_file.write("## 类\n\n")
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if obj.__module__ == module.__name__:
                output_file.write(f"### {name}\n\n")
                if obj.__doc__:
                    output_file.write(f"{obj.__doc__}\n\n")
                
                # 写入方法文档
                output_file.write("#### 方法\n\n")
                for method_name, method in inspect.getmembers(obj, inspect.isfunction):
                    if method.__module__ == module.__name__:
                        signature = get_function_signature(method)
                        output_file.write(f"##### {method_name}{signature}\n\n")
                        if method.__doc__:
                            output_file.write(f"{method.__doc__}\n\n")
                        else:
                            output_file.write("无文档\n\n")
        
        # 写入函数文档
        output_file.write("## 函数\n\n")
        for name, obj in inspect.getmembers(module, inspect.isfunction):
            if obj.__module__ == module.__name__:
                signature = get_function_signature(obj)
                output_file.write(f"### {name}{signature}\n\n")
                if obj.__doc__:
                    output_file.write(f"{obj.__doc__}\n\n")
                else:
                    output_file.write("无文档\n\n")
        
        output_file.write("\n---\n\n")
    except Exception as e:
        output_file.write(f"生成 {module_path} 的文档时出错: {e}\n\n")

def generate_all_function_docs(llmgraph_dir, output_path):
    """
    为LLMGraph目录中的所有Python文件生成函数文档
    
    Args:
        llmgraph_dir: LLMGraph目录路径
        output_path: 输出文件路径
    """
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 打开输出文件
    with open(output_path, 'w', encoding='utf-8') as output_file:
        # 写入标题
        output_file.write("# LLMGraph 函数文档\n\n")
        
        # 遍历LLMGraph目录中的所有Python文件
        for root, dirs, files in os.walk(llmgraph_dir):
            for file in files:
                if file.endswith('.py') and not file.startswith('__'):
                    module_path = os.path.join(root, file)
                    generate_function_docs(module_path, output_file)
    
    print(f"已生成所有函数文档，保存在 {output_path}")

if __name__ == "__main__":
    # 获取当前目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # LLMGraph目录路径
    llmgraph_dir = os.path.join(current_dir, "LLMGraph")
    
    # 输出文件路径
    output_path = os.path.join(current_dir, "LLMGraph函数文档.md")
    
    # 生成文档
    generate_all_function_docs(llmgraph_dir, output_path)