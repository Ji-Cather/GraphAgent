from typing import Dict
from LLMGraph.registry import Registry
env_registry = Registry(name="EnvironmentRegistry")
from .base import BaseEnvironment
from .article import ArticleEnvironment
from .movie import MovieEnvironment
from .social import SocialEnvironment