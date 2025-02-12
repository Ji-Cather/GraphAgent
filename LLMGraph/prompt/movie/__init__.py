import yaml
from .. import MODEL
movie_prompt_default = yaml.safe_load(open(f"LLMGraph/prompt/movie/{MODEL}.yaml"))

from LLMGraph.registry import Registry
movie_prompt_registry = Registry(name="MoviePromptRegistry")

from .watch_movie import WatchMovieBatchPromptTemplate
from .filter import ChooseGenrePromptTemplate
from .rate import RatePromptTemplate
from .plan import WatchPlanPromptTemplate