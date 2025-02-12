import yaml
from .. import MODEL
article_prompt_default = yaml.safe_load(open(f"LLMGraph/prompt/article/{MODEL}.yaml"))

from LLMGraph.registry import Registry
article_prompt_registry = Registry(name="ArticlePromptRegistry")

from .group_discuss import *
from .get_author import *
from .choose import *
from .write_article import *
