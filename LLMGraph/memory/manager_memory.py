import random
from collections import deque
from LLMGraph.registry import Registry

manager_memory_registry = Registry(name="ManagerMemoryRegistry")

# Add random walk method to aggregate neighbor messages
# Memory Bank aggregation methods
@manager_memory_registry.register("random_walk")
class RandomWalk:
    def __call__(self, edges, nodes, start_node_id, walk_length=3, num_walks=10):
        """
        Perform random walks from a starting node to aggregate neighbor messages
        
        Args:
            start_node_id: ID of starting node
            walk_length: Length of each random walk
            num_walks: Number of random walks to perform
            
        Returns:
            List of node texts encountered in walks
        """
        neighbor_node_texts = []
        neighbor_edge_texts = []
        for _ in range(num_walks):
            current_node = start_node_id
            walk = []
            walk_edges = []
            
            for _ in range(walk_length):
                # Get neighbors and edges of current node
                neighbor_edges = [edge for edge in edges if edge["actor_id"] == current_node]
                if not neighbor_edges:
                    break
                    
                # Randomly select next edge and node
                next_edge = random.choice(neighbor_edges)
                next_node = next_edge["item_id"]
                walk.append(next_node)
                walk_edges.append(next_edge)
                current_node = next_node
            
            # Get node and edge texts from walk
            for node_id in walk:
                if node_id in nodes:
                    neighbor_node_texts.append(nodes[node_id]["node_text"])
                    
            for edge in walk_edges:
                if "edge_text" in edge:
                    neighbor_edge_texts.append(edge["edge_text"])
                    
        return neighbor_node_texts, neighbor_edge_texts

@manager_memory_registry.register("k_hop")
class KHop:
    def __call__(self, edges, nodes, start_node_id, k=2):
        """
        Aggregate k-hop neighbors of a given node.
        
        Args:
            edges: List of edge dictionaries, each containing 'actor_id' and 'item_id'.
            nodes: Dictionary mapping node IDs to their details, including 'node_text'.
            start_node_id: ID of the starting node.
            k: The number of hops (depth) to explore from the start node.
            
        Returns:
            A tuple (neighbor_node_texts, neighbor_edge_texts), where
            - neighbor_node_texts is a list of texts for all unique nodes visited up to k-hops,
            - neighbor_edge_texts is a list of texts for all unique edges traversed during the exploration.
        """
        # 使用队列来存储待访问的节点及其距离起点的距离
        queue = deque([(start_node_id, 0)])  # (当前节点ID, 当前距离)
        visited_nodes = set()  # 记录已访问过的节点
        visited_edges = set()  # 记录已遍历过的边
        neighbor_node_texts = []
        neighbor_edge_texts = []

        while queue:
            current_node, distance = queue.popleft()
            
            if current_node in visited_nodes or distance > k:
                continue  # 如果已经访问过此节点或超出探索深度，则跳过
            
            visited_nodes.add(current_node)  # 标记为已访问
            if current_node in nodes and "node_text" in nodes[current_node]:
                neighbor_node_texts.append(nodes[current_node]["node_text"])
            
            # 获取当前节点的所有邻接边
            for edge in [e for e in edges if e["actor_id"] == current_node]:
                edge_key = (edge["actor_id"], edge["item_id"])  # 边的唯一标识
                
                if edge_key not in visited_edges:
                    visited_edges.add(edge_key)
                    if "edge_text" in edge:
                        neighbor_edge_texts.append(edge["edge_text"])
                    
                    # 将邻接点加入队列
                    queue.append((edge["item_id"], distance + 1))
        
        return neighbor_node_texts, neighbor_edge_texts


if __name__ == "__main__":
    edges = [
        {"actor_id": "1", "item_id": "2", "edge_text": "edge from 1 to 2"},
        {"actor_id": "2", "item_id": "3", "edge_text": "edge from 2 to 3"},
        {"actor_id": "3", "item_id": "4", "edge_text": "edge from 3 to 4"}
    ]
    nodes = {"1": {"node_text": "node 1"}, "2": {"node_text": "node 2"}, "3": {"node_text": "node 3"}, "4": {"node_text": "node 4"}}
    print(manager_memory_registry.build("random_walk")(edges, nodes, "1"))
    print(manager_memory_registry.build("k_hop")(edges, nodes, "1"))
