import yaml
from .. import MODEL
prompts = {}
for structure_type in ["node","edge","interact"]:
    prompt_one = yaml.safe_load(open(f"LLMGraph/prompt/general/{MODEL}/{structure_type}.yaml"))
    replaced_dict = {}
    for k, v in prompt_one.items():
        replaced_dict[f"{structure_type}_{k}"] = v
    prompts.update(replaced_dict)
general_prompt_default = prompts

from LLMGraph.registry import Registry
general_prompt_registry = Registry(name="GeneralPromptRegistry")

from .templates import *