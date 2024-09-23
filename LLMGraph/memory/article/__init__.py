from LLMGraph.registry import Registry
article_memory_registry = Registry(name="ArticleMemoryRegistry")



from .social_memory import  SocialMemory
from .rational_memory import  RationalMemory

