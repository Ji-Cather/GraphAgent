import yaml
from .. import MODEL
social_prompt_default = yaml.safe_load(open(f"LLMGraph/prompt/social/{MODEL}.yaml"))

from LLMGraph.registry import Registry
social_prompt_registry = Registry(name="SocialPromptRegistry")

from .social_action import (ForumActionPromptTemplate,
                            ForumActionBigNamePromptTemplate)
from .search import ForumSearchPromptTemplate
from .choose import ChooseTopicPromptTemplate
from .plan import ForumActionPlanPromptTemplate