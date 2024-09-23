from LLMGraph.registry import Registry
import yaml
def select_to_last_period(s, upper_token = 4e3):
    upper_token = int(upper_token)
    s = s[-upper_token:]
    # 查找最后一个句号的位置
    last_period_index = s.rfind('.')
    # 如果找到句号，返回从开始到最后一个句号之前的部分
    if last_period_index != -1:
        return s[:last_period_index]
    else:
        # 如果没有找到句号，返回整个字符串
        return s
summary_prompt_default = yaml.safe_load(open("LLMGraph/memory/summary.yaml"))
memory_registry = Registry(name="MemoryRegistry")

from .base import BaseMemory



from .article import article_memory_registry
from .movie import movie_memory_registry, MovieMemory
from .social import social_memory_registry

