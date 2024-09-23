from typing import Dict

from pydantic import BaseModel


class Registry(BaseModel):
    """Registry for storing and building classes."""

    name: str
    entries: Dict = {}

    def register(self, key: str):

        def decorator(class_builder):
            self.entries[key] = class_builder
            return class_builder

        return decorator

    def build(self, type: str, **kwargs):
        if type not in self.entries:
            raise ValueError(
                f'{type} is not registered. Please register with the .register("{type}") method provided in {self.name} registry'
            )
        return self.entries[type](**kwargs)
    
    def load_data(self,type: str, ** kwargs):
        if type not in self.entries:
            raise ValueError(
                f'{type} is not registered. Please register with the .register("{type}") method provided in {self.name} registry'
            )
        return self.entries[type].load_data(**kwargs)
    
    def from_db(self,type:str,vectorstore = None,**kwargs):
        if type not in self.entries:
            raise ValueError(
                f'{type} is not registered. Please register with the .register("{type}") method provided in {self.name} registry'
            )
        if vectorstore is None:
            return self.entries[type].from_db(**kwargs)
        else:
            return self.entries[type].from_db(vectorstore = vectorstore,
                                          **kwargs)
    
    def get_entry(self,type):
        return self.entries.get(type)
    
    def get_all_entries(self):
        return self.entries
