# 关于推荐功能

现阶段的所有推荐功能集成在LLMGraph/retriever中，以citation场景为例，假设需要设定检索工具，那么一共存在四步：

1. retriver目录中：设定retriever，基本继承了VectorStoreRetriever进行开发
    LLMGraph/retriever 中按照不同场景提供了四个retriever，GraphArxivRetriever、GraphVectorRetriever、GraphGeneralVectorRetriever、GoogleScholarAPIWrapper
    可以根据需要进行进一步优化。


2.  tool目录中：基于retrievr，设定基于agent的搜索逻辑。其中一共提供了6个tool和一个tool_wrapper.

    以citation场景举例，设定类似 _get_article_relevant_documents函数，处理agent的各类query信息
    

    ```python
    def _get_article_relevant_documents(
        query: str,
        retriever: BaseRetriever,
        article_meta_data:dict,
        author_data:dict,
        document_prompt: BasePromptTemplate,
        document_separator: str,
        experiment:list = [], # default/shuffle/false cite
        filter_keys: list = [
            "topic", "big_name", "write_topic"
        ],
        max_search:int = 5,
        big_name_list:list = [],
        interested_topics:List[str] = [],
        research_topic:str =""
    ) -> str:
        """Search for relevant papers, so as to refine your paper. \
    These papers should be included in your paper's citations if you use them in your paper. 

        Args:
            query (str): keywords split by commas. The informations about the papers you want to cite, you can enter some keywords or other info.

        Returns:
            str: information about some searched papers.
        """
    ```

    其中按照两步设计
    
    基础召回为 docs = retriever.get_relevant_documents(query)
    
    排序部分则按照预定规则设定为filter_pipeline
    
    ```python
    filter_pipeline = []
    filter_keys_set_map = {
        "big_name":generate_compare_function_big_name(big_name_list), 
        "topic":generate_compare_function_topic(interested_topics),
        "write_topic":generate_compare_function_topic([research_topic]),
        "cite":generate_compare_function_cite(),
        }
    ```

    按照各个模拟环境不同可以自定义搜索逻辑

3. tool目录中：tool_wrapper
    将搜索函数包装为service，供agent调用
    以citation场景举例，类似create_article_retriever_tool，通过GraphServiceFactory.get进行包装为service

4. agent目录中：将包装好的service，放在manageragent中，作为manageragent可调用的工具之一。
    
    以citation场景举例：
    ```python
    if update_retriever:
            retriever = self.article_manager.get_retriever()
            retriever_tool_func,retriever_tool_dict = create_article_retriever_tool(
                retriever,
                "search",
        "Search for relevant papers, so as to refine your paper. \
    These papers should be included in your paper's citations if you use them in your paper. \
    Your paper should cite at least {min_citations_db} papers!".format_map(self.article_write_configs),
                
                author_data = self.article_manager.author_data,
                article_meta_data = self.article_manager.article_meta_data,
                experiment = self.article_manager.experiment,
                document_prompt = self.document_prompt,
                filter_keys= self.article_manager.tool_kwargs["filter_keys"],
                big_name_list = self.article_manager.big_name_list,
                interested_topics = interested_topics,
                research_topic = research_topic)
            
            tools = {
            "search":ToolAgent("search",
                        tool_func=retriever_tool_func,
                        func_desc=retriever_tool_dict
                        )}
            self.update_tools(tools)
    ```
    上述代码首先根据manageragent中存储的article进行retriver更新，并更新搜索函数的各类超参（例如big_name_list, interested_topics）

    这边tools只设定了一个搜索函数，同样可以设定多个工具，例如
    ```python
    tools = {
            "search_offline":ToolAgent("search",
                        tool_func=retriever_tool_func,
                        func_desc=retriever_tool_dict
                        )
            "search_arxiv":ToolAgent("search",
                        tool_func=retriever_arxiv_func,
                        func_desc=retriever_arxiv_dict
                        )
            }
            self.update_tools(tools)
    ```
    但需要将上述包装工具的过程重复两次[1-3步骤]


5. EXAMPLE：具体的执行过程。
    为了并行的序列化要求，我们将所有的manageragent+actionagent包装在一个wrapperagent中。

    同样以citation场景举例，ArticleAgentWrapper中, 存在ArticleAgent实例 self.agent、以及ArticleManagerAgent实例 self.manager_agent

    因而在调用ArticleAgent中的write_article函数时，首先要调用ArticleAgentWrapper的write_article函数（方便找到ArticleManagerAgent的搜索service。从而在搜索时， ArticleAgentWrapper通过下面的步骤调用service
    ```python
    response = self.step(agent_msgs,use_tools=True,return_intermediate_steps=True)
    ```

    这一步代码会执行两步
    ```python
    # Step 1: Thought ，这一步骤返回function calling的命令，给出retriver的query词
    # Step 2: Action，这一步骤返回service执行结果，给出搜索结果
    ```

6. TBD：LLMGraph/recs
    现阶段的搜索逻辑较为简单，接下来我们计划在recs模块设计更为复杂的推荐系统，用于增强模拟的真实性和仿真度。