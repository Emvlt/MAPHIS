import heapq
from typing import List, Tuple, Dict
import math
from edge import Edge
from feature_node import Node 
import heapq
import statistics
from sklearn.linear_model import LinearRegression

class PriorityQueue:
    def __init__(self):
        self.elements: List[Tuple[float, Node]] = []
    
    def empty(self) -> bool:
        return not self.elements
    
    def put(self, item: Node, priority: float):
        heapq.heappush(self.elements, (priority, item))
    
    def get(self) -> Node:
        return heapq.heappop(self.elements)[1]

def heuristic(a: Node, b: Node) -> float:
        return abs(a.x - b.x) + abs(a.y - b.y)

def a_star_search(start: Node, goal: Node, dict_of_nodes:Dict[str, Node], background):
        frontier = PriorityQueue()
        frontier.put(start.key, 0)
        came_from: Dict[Node, Node] = {}
        cost_so_far: Dict[Node, float] = {}
        came_from[start.key] = start
        cost_so_far[start.key] = 0
        visited = []
        while not frontier.empty():            
            current_node:Node = dict_of_nodes[frontier.get()]
            
            if current_node.is_equal_to(goal):
                break

            neighbours:List[Node] = [node for node in dict_of_nodes.values() if not current_node.intersect(node, background) and not node.is_equal_to(current_node) and not node.is_in(visited)]
            neighbours.sort(key=lambda x: math.sqrt((current_node.x-x.x)**2 +  (current_node.y-x.y)**2))
            for next in neighbours:
                visited.append(next)
                new_cost = cost_so_far[current_node.key] + Edge(current_node, next).length
                if not next.key in cost_so_far or new_cost < cost_so_far[next.key]:
                    cost_so_far[next.key] = new_cost
                    priority = new_cost + heuristic(next, goal)
                    frontier.put(next.key, priority)
                    came_from[next.key] = current_node
        
        return came_from, cost_so_far

def reconstruct_path(came_from: Dict[Node, Node],
                    start: Node, goal: Node) -> List[Node]:

    current: Node = goal
    path: List[Node] = []
    while not current.is_equal_to(start): # note: this will fail if no path found
        path.append(current)
        current = came_from[current.key]
    path.append(start) # optional
    path.reverse() # optional
    return path

def compute_path(start_key:str, end_key:str, dict_of_nodes, background):        
    start = dict_of_nodes[start_key]
    end = dict_of_nodes[end_key]
    print('Computing A*')
    came_from, cost_so_far = a_star_search(start, end , dict_of_nodes, background)
    print('Reconstructing path')
    path_nodes = reconstruct_path(came_from, start, end)
    return path_nodes
    '''print('Computing reachable nodes')
    if are_nodes_reachable(start, end, dict_of_nodes, background):
        print('Computing A*')
        came_from, cost_so_far = a_star_search(start, end , dict_of_nodes, background)
        print('Reconstructing path')
        path_nodes = reconstruct_path(came_from, start, end)
        return path_nodes
    else:
        print(f'Nodes {start_key} and {end_key} are not reachable')
        return None'''

def explore(background, current_node:Node, dict_of_nodes, visited:list):
        # Appending new node to the list of visited nodes
        visited.append(current_node)
        # Get all children (all reachable nodes that were not visited) and sort them in distance order from the current node
        children:List[Node] = [node for node in dict_of_nodes.values() if not current_node.intersect(node, background) and (not node.is_in(visited))]
        children.sort(key=lambda x: math.sqrt((current_node.x-x.x)**2 +  (current_node.y-x.y)**2))
                
        # Iterate over the children
        for child in children:
            if not child.is_in(visited):
                visited = explore(background, child, dict_of_nodes, visited)
    
        return visited

def are_nodes_reachable(start:Node, end:Node, dict_of_nodes, background) ->bool:
    visited = explore(background, current_node=start, dict_of_nodes=dict_of_nodes, visited=[])
    if end.is_in(visited):
        return True
    return False  

def explore_street_names(background, current_node:Node, dict_of_nodes:Dict[str, Node], visited:List[Node], distances:List[float], inv_map) -> Tuple[List[Node], List[float], List[float]]:
        # Appending new node to the list of visited nodes
        visited.append(current_node)
        # Get all children (all reachable nodes that were not visited) and sort them in distance order from the current node
        children:List[Node] = [node for node in dict_of_nodes.values() if not current_node.intersect(node, background) and (not node.is_in(visited))]
        children.sort(key=lambda x: math.sqrt((current_node.x-x.x)**2 +  (current_node.y-x.y)**2))
        X = [[i, n.x] for i, n in enumerate(visited)]
        ys = [n.y for n in visited]
        reg:LinearRegression = LinearRegression().fit(X, ys )   
        # Iterate over the children
        for child in children:
            L = Edge(current_node, child).length
            N = len(visited)
            if len(distances)==0:
                distances.append(L)
            #if  reg.predict([[N+1, child.x]])-child.y)<20 and L < 3*distances[-1]  and not child.is_in(visited) :
            #ys_temp = ys+[child.y]
            #X_temp = X + [[N, child.x]]
            #if  0.995<reg.score(X_temp, ys_temp ) and L < 3*distances[-1]  and not child.is_in(visited) :
            #if L < 3*distances[-1]  and not child.is_in(visited) :
            #if abs(reg.predict([[N, child.x]]) - child.y) < 15 and L < 3*distances[-1]  and not child.is_in(visited) :

            if  L < 3*distances[-1]  and not child.is_in(visited) :
                distances.append(L)
                visited, distances  = explore_street_names(background, child, dict_of_nodes, visited, distances, inv_map)
                
    
        return visited, distances
