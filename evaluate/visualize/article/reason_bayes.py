from causalnex.network import BayesianNetwork
from causalnex.structure.structuremodel import StructureModel
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm 
font_path = 'test/Times_New_Roman/TimesNewerRoman-Regular.otf'
font_prop = fm.FontProperties(fname=font_path)
fm.fontManager.addfont(font_path)

# 设置全局字体
plt.rcParams['font.family'] = font_prop.get_name()
plt.rcParams['font.sans-serif'] = [font_prop.get_name()]
import os

index_map ={
#     "Background":"Background",
# "Fundamental_idea":"Fundamental Idea",
# "Technical_basis":"Technical Basis",
# "Comparison":"Comparison",
"Similar_Content":"Paper Content",
"High_Citation":"Paper Citation",
"Timeliness":"Paper Timeliness",
"Topic":"Paper Topic",
"Highly_cited_Author":"Author Citation",
"known_author":"Author Name",
"Country_Institution":"Author Country",
}

sections_map ={
    "Related_Work":"Related Work",
    "Introduction":"Introduction",
    "Method":"Method",
    "Experiment":"Experiment"
}


config ="search_shuffle_base_gpt3.5_ver1"
config = "search_shuffle_base_gpt3.5"

task ="llm_agent"
evaluate_root =f"LLMGraph/tasks/{task}/configs/{config}/evaluate"

df_citation = pd.read_csv(os.path.join(evaluate_root,"impact_citation_bayes.csv"),
                          index_col=0)
df_section = pd.read_csv(os.path.join(evaluate_root,"impact_section_bayes.csv"),
                         index_col=0)
df_section.rename(columns=sections_map,inplace=True)
df_citation = df_citation.loc[df_citation.index.isin(index_map.keys())]
df_section = df_section.loc[df_section.index.isin(index_map.keys())]
df_citation.index =[index_map[idx] for idx in df_citation.index]
df_section.index =[index_map[idx] for idx in df_section.index]

threhold = 3
sm = StructureModel()
citation_reasons = df_citation.sort_values(by = "Citation").index.to_list()
sections = df_section.columns.to_list()
edges = []
for section in sections:
    section_reasons = df_section.sort_values(by = section).index.to_list()[:threhold]
    edges.extend([
        (section_reason,section)
       for section_reason in section_reasons
       ]
    )
edges.extend([
    (citation_reason, "Citation")
    for citation_reason in citation_reasons[:threhold]
])
edges.extend([
    ("Citation",section)
    for section in sections
])

sm.add_edges_from([
    # ('SourcePaper', 'Citation'),
    # ('TargetPaper', 'Citation'),
    *edges,
])

sorted_nodes = ["Citation",*sections, *list(filter(lambda x:x in index_map.values(),
                                                   sm.nodes))]
# sorted_nodes = [*sections, *list(filter(lambda x:x in df_citation.index,
#                                                   sm.nodes))]
node_size_dict = {node: 1000 for node in sorted_nodes}
node_size_dict.update({node: 2000 for node in sections})
node_size_dict["Citation"] = 3000

# 设置节点颜色
node_color_dict = {node: 'lightblue' for node in sorted_nodes}  # 默认节点颜色
node_color_dict.update({node: 'lightgreen' for node in sections})  # Section节点颜色
node_color_dict["Citation"] = 'salmon'  # Citation节点颜色
# 设置标签字体大小
font_size_dict = {node: 10 for node in sorted_nodes}  # 默认字体大小
font_size_dict.update({node: 12 for node in sections})  # Section节点字体大小
font_size_dict["Citation"] = 14  # Citation节点字体大小

node_size = [node_size_dict[node] for node in sorted_nodes]
node_color = [node_color_dict[node] for node in sorted_nodes]
font_size = [font_size_dict[node] for node in sorted_nodes]

pos = nx.circular_layout(G=sm)

nx.draw(sm, pos, with_labels=True, nodelist = sorted_nodes,node_size=node_size, 
        node_color=node_color, font_size=10,
          font_weight='bold', font_color='black', arrows=True)

# # 单独调整每个label的字体大小
# for p in pos.keys():
#     x, y = pos[p]
#     plt.text(x, y, s=p,
#             bbox=dict(facecolor='lightblue', alpha=0.6),
#             horizontalalignment='center', fontsize=font_size_dict[p])

plt.title(f"Bayesian Network")
plt.tight_layout()
plt.savefig(os.path.join(evaluate_root,"reason_bayes.pdf"))