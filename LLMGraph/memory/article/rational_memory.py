from . import article_memory_registry
from LLMGraph.memory.base import BaseMemory
from typing import List, Sequence,Union,Dict


from agentscope.message import Msg
from .. import select_to_last_period
import random

@article_memory_registry.register("rational_memory")
class RationalMemory(BaseMemory):

    articles:List[Msg] = []
    cited_articles:Dict[str,List[float]] ={}
    interested_topics: list = []

    def __init__(self,**kwargs):
        super().__init__(**kwargs)
       
    
    def add_message(self, 
                    research_content:Msg):
        research_content = research_content.content
        self.articles.append(research_content)
        citation_reasons = research_content.get("citation_reasons",[])
        citation_titles = research_content.get("citations",[])
        for citation_reason in citation_reasons:
            cite_index = int(citation_reason["index"]-1)
            try:
                cite_title = citation_titles[cite_index]
                if cite_title not in self.cited_articles:
                    self.cited_articles[cite_title] = []
                self.cited_articles[cite_title].append(float(citation_reason.get("importance",0)))
            except Exception as e:
                continue
        
        self.cited_articles = dict(
            sorted(self.cited_articles.items(), 
            key=lambda x: sum(x[1])/len(x[1]), reverse=True))

    def retrieve_article_memory(self,
                         topk = 10,
                        upper_token = 1e3):
        

        citation_header = """
Here's the articles you have cited in your past research:
"""

        citation_template = """
{idx}. {title}:{importance:.1f}
"""     

        most_used_papers = self.cited_articles.items()
        if len(most_used_papers)>topk:
            most_used_papers = most_used_papers[:topk]

        papers_str = "\n".join([
            citation_template.format(
                idx=idx+1,
                title=title,
                importance=sum(importance)/len(importance)
            ) for idx, (title, importance) in enumerate(most_used_papers)
        ])
        
        memory = citation_header + papers_str
        if len(memory)> upper_token:
            return select_to_last_period(memory, upper_token)
        return memory
    
    def update_interested_topics(self,topics:list):
        if isinstance(topics,str):topics = [topics]
        self.interested_topics.extend(topics)

    def return_interested_topics(self,k = 10):
        return_topics = []
        if k < len(self.interested_topics):
            return_topics = random.sample(self.interested_topics,k)
        else:
            return_topics = self.interested_topics
        return list(set(return_topics))