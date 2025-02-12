from LLMGraph.registry import Registry

output_parser_registry = Registry(name="OutputParserRegistry")
article_output_parser_registry = Registry(name="ArticleOutputParserRegistry")
movie_output_parser_registry = Registry(name="MovieOutputParserRegistry")
social_output_parser_registry = Registry(name="SocialOutputParserRegistry")
control_output_parser_registry = Registry(name="ControlOutputParserRegistry")
general_output_parser_registry = Registry(name="GeneralOutputParserRegistry")

from .article import *
from .movie import *
from .social import *
from .control import *
from .general import *