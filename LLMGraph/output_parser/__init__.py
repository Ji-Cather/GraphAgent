"""
输出解析器模块

这个模块包含了不同类型的输出解析器注册表:

- output_parser_registry: 通用输出解析器注册表
- article_output_parser_registry: 文章相关输出解析器注册表  
- movie_output_parser_registry: 电影相关输出解析器注册表
- social_output_parser_registry: 社交相关输出解析器注册表
- control_output_parser_registry: 控制相关输出解析器注册表
- general_output_parser_registry: 通用输出解析器注册表
"""

from LLMGraph.registry import Registry

output_parser_registry = Registry(name="OutputParserRegistry")
article_output_parser_registry = Registry(name="ArticleOutputParserRegistry")
movie_output_parser_registry = Registry(name="MovieOutputParserRegistry")
social_output_parser_registry = Registry(name="SocialOutputParserRegistry")
control_output_parser_registry = Registry(name="ControlOutputParserRegistry")
general_output_parser_registry = Registry(name="GeneralOutputParserRegistry")

from .article import *
from .movie import *
from .social import *
from .control import *
from .general import *