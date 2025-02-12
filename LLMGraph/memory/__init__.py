from LLMGraph.registry import Registry
import yaml
from LLMGraph import select_to_last_period
summary_prompt_default = yaml.safe_load(open("LLMGraph/memory/summary.yaml"))
memory_registry = Registry(name="MemoryRegistry")

from .base import BaseMemory
from .general import GeneralMemory

from .article import article_memory_registry
from .movie import movie_memory_registry, MovieMemory
from .social import social_memory_registry

