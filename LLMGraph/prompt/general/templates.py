from LLMGraph.message import Message
from . import general_prompt_registry,general_prompt_default
import copy
from LLMGraph.prompt.base import BaseChatPromptTemplate
import yaml
prompts = {}
from LLMGraph.prompt import MODEL
for structure_type in ["node","edge","interact"]:
    prompt_one = yaml.safe_load(open(f"LLMGraph/prompt/general/{MODEL}/{structure_type}.yaml"))
    replaced_dict = {}
    for k, v in prompt_one.items():
        replaced_dict[f"{structure_type}_{k}"] = v
    prompts.update(replaced_dict)

for k,v in general_prompt_default.items():
    node_mode = k
    class_name = node_mode + 'PromptTemplate'
    def create_prompt_class(node_mode):
        class_name = node_mode + 'PromptTemplate'
        prompt_class = type(class_name, (BaseChatPromptTemplate,), {
            '__init__': lambda self, **kwargs: super(type(self), self).__init__(
                template=copy.deepcopy(general_prompt_default.get(node_mode, "")),
                **kwargs
            )
        })
        return prompt_class

    for k, v in general_prompt_default.items():
        prompt_class = create_prompt_class(k)
        general_prompt_registry.register(k)(prompt_class)

general_prompt_registry
