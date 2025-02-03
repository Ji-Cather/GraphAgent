from typing import Sequence
from langchain_community.document_loaders.directory import DirectoryLoader
from langchain_community.document_loaders.unstructured import UnstructuredFileLoader
from pathlib import Path
from typing import Any, List, Optional, Sequence, Type, Union

from langchain_core.documents import Document

from langchain_community.document_loaders.base import BaseLoader
from langchain_community.document_loaders.html_bs import BSHTMLLoader
from langchain_community.document_loaders.text import TextLoader
import os

import concurrent
import logging
import random
from pathlib import Path
from typing import Any, List, Optional, Sequence, Type, Union

from langchain_core.documents import Document

from langchain_community.document_loaders.base import BaseLoader
from langchain_community.document_loaders.html_bs import BSHTMLLoader
from langchain_community.document_loaders.text import TextLoader
from langchain_community.document_loaders.unstructured import UnstructuredFileLoader

from datetime import datetime, date
import copy 


class DirectoryArticleLoader(DirectoryLoader):
    def __init__(self,
                 article_meta_data:dict ={},
                 *args,
                 **kargs):
        super().__init__(*args,**kargs)
        self.docs = []
        self.paths =[]
        self.doc_map = {}
        self.map_key = "title" # map meta_data[map_key]: index for docs
        self.add_doc_dir(article_meta_data = article_meta_data)
        
    def load(self) -> List[Document]:
        return self.docs
    
    def add_doc_dir(self,
                    article_meta_data:dict ={}):
        add_docs = self.load_dir(article_meta_data)
        left_p = len(self.docs)
        right_p = len(add_docs) + left_p
           
        for idx, doc in zip(range(left_p,right_p),add_docs):
            # assert idx not in self.doc_map.keys()
            self.doc_map[doc.metadata[self.map_key]] = idx
            
        self.docs = [*self.docs,*add_docs]
        return add_docs

    
    def get_article_docs(self,
                         titles,
                    article_meta_data,
                    author_data):
        docs_str = []
        """按照cite选择best author"""
        for title in titles:
            if title not in self.doc_map.keys():
                continue
            doc_str = self.format_document(article_meta_data,
                                           author_data,
                                           self.docs[self.doc_map[title]])
            docs_str.append(doc_str)
        
        return "\n\n".join(docs_str)
    
    def format_document(self,
                        article_meta_data,
                        author_data,
                        doc):
        choose_key = "cited"
        title = doc.metadata["title"]
        try:
            author_ids = article_meta_data[title]["author_ids"]
            author_ids = list(filter(lambda x:x in author_data.keys(), author_ids))

            best_author_idx = author_ids[0]
            for author_id in author_ids:
                author_info = author_data[author_id]
                if author_data[best_author_idx].get(choose_key,0) < author_info.get(choose_key,0):
                    best_author_idx = author_id
            
            best_author_info ={
                "author_cited": author_data[best_author_idx].get("cited",0),
                "country": author_data[best_author_idx].get("country",0),
                "institution": author_data[best_author_idx].get("institution",0),
                "author_name": author_data[best_author_idx].get("name",0),
            }
        except:
            best_author_info ={
                "author_cited": "Unknown",
                "country":"",
                "institution": "",
                "author_name": "Unknown",
            }
        prompt = """\
Title: {title}
Cited: {cited}
Author: {author_name} {institution} {country} cited: {author_cited}
Publish Time: {time}
Content: {page_content}"""
        doc_infos ={
                **doc.metadata,
                "page_content":doc.page_content[:200],
                **best_author_info
            }
        return prompt.format_map(doc_infos)

    def load_dir(self,
                article_meta_data:dict = {}) -> List[Document]:
        import pathlib
        import platform
        paths = []
        filtered_article_meta_data = {}
        article_meta_data = copy.deepcopy(article_meta_data)
        
        for key,item in article_meta_data.items():
            item["path"] = pathlib.Path(item["path"])
            if item["path"] not in self.paths:
                paths.append(item["path"])
                filtered_article_meta_data[key] = item
        if len(paths) == 0:
            return []        
        
    
        docs = list(self.lazy_load_dir(paths))
        self.paths.extend(paths)

        for doc, doc_meta_info in zip(docs, filtered_article_meta_data.items()):
            title, info = doc_meta_info
            doc.metadata["title"] = title
            try:
                doc.metadata["cited"] = info["cited"]
                from LLMGraph.utils.process_time import transfer_time
                if not isinstance(info["time"],date):
                    info["time"] = transfer_time(info["time"])
                doc.metadata["time"] = info["time"].strftime("%Y-%m")
                doc.metadata["topic"] = info.get("topic","AI")
            except:
                continue

        return docs
    
    def lazy_load_dir(self,
                        paths):
        """Load documents."""
        items = [
            path
            for path in paths
            if not (self.exclude and any(path.match(glob) for glob in self.exclude))
        ]

        if self.sample_size > 0:
            if self.randomize_sample:
                randomizer = random.Random(
                    self.sample_seed if self.sample_seed else None
                )
                randomizer.shuffle(items)
            items = items[: min(len(items), self.sample_size)]

        pbar = None
        if self.show_progress:
            try:
                from tqdm import tqdm

                pbar = tqdm(total=len(items))
            except ImportError as e:
                pass

        if self.use_multithreading:
            futures = []
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.max_concurrency
            ) as executor:
                for i in items:
                    futures.append(
                        executor.submit(
                            self._lazy_load_file_to_non_generator(self._lazy_load_file),
                            i,
                            pbar,
                        )
                    )
                for future in concurrent.futures.as_completed(futures):
                    yield future.result()
        else:
            for i in items:
                yield from self._lazy_load_file(i,  pbar)

        if pbar:
            pbar.close()

        # return docs
    
    def _lazy_load_file(
        self, item: Path, pbar: Optional[Any]
    ) :
        """Load a file.

        Args:
            item: File path.
            path: Directory path.
            pbar: Progress bar. Defaults to None.

        """
        if item.is_file():
            try:
                loader = self.loader_cls(str(item), **self.loader_kwargs)
                try:
                    for subdoc in loader.lazy_load():
                        yield subdoc
                except NotImplementedError:
                    for subdoc in loader.load():
                        yield subdoc
            except Exception as e:
                raise e
            finally:
                if pbar:
                    pbar.update(1)

    def _lazy_load_file_to_non_generator(self, func) :
        def non_generator(item: Path,pbar: Optional[Any]) -> List:
            return [x for x in func(item, pbar)]

        return non_generator
        