import yaml
from .. import MODEL
control_prompt_default = yaml.safe_load(open(f"LLMGraph/prompt/control/{MODEL}.yaml"))

from LLMGraph.registry import Registry
control_prompt_registry = Registry(name="ControlPromptRegistry")

from .control import *