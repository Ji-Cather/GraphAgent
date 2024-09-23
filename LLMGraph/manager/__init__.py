from LLMGraph.registry import Registry
manager_registry = Registry(name="ManagerRegistry")


from .article import ArticleManager
from .movie import MovieManager
from .social import SocialManager