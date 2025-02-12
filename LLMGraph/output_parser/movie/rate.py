from .. import movie_output_parser_registry
from typing import  Union
from ..base_parser import (AgentOutputParser,
                           find_and_load_json)

import re
@movie_output_parser_registry.register("rate_movie")
class RateMovierParser(AgentOutputParser):
    
    def parse(self, llm_output: str) :
        ratings = []
        try:
            # outputs = llm_output.split("\n")
            
            # regex = r".*?Movie.*?:(.*?);.*?Score.*?:(.*?);.*?Reason.*?:(.*?)\n"
            # for output in outputs:
            #     if output.strip() == "":
            #         continue
            #     output += "\n"
            #     match = re.search(regex, output,re.DOTALL|re.IGNORECASE)
            #     ratings.append({
            #         "movie":match.group(1).strip(),
            #         "thought":match.group(3).strip(),
            #         "rating":float(match.group(2).strip()),
            #     })
            ratings = find_and_load_json(llm_output,"list")
            assert isinstance(ratings,list)
            json_filed = {"return_values":
                    {"ratings":ratings}}
            return json_filed
           
        except Exception as e:
            return {"fail":True}