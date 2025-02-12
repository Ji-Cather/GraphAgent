from pydantic import BaseModel

class BaseManager(BaseModel):
    control_profile:dict = {
        "hub_rate":0.2,
    }
    tool_kwargs:dict = {}
    retriever_kwargs:dict = {
        "type": "graph_vector_retriever",
        "k": 6}