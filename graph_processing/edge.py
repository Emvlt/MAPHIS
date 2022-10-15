"""This module holds the class Edge used to connect two Nodes"""
import math
from dataclasses import dataclass
from feature_node import Node

@dataclass
class Edge():
    """Class definition for the edge object, linking two Nodes
    """
    def __init__(self, node_0:Node, node_1:Node) -> None:
        self.node_0 = node_0
        self.node_1 = node_1
        self.length = math.sqrt(math.pow(self.node_0.x - self.node_1.x,2) + math.pow(self.node_0.y - self.node_1.y,2))

    def link(self, other_edge:object):
        return Edge(self.node_0, other_edge.node_1)

    @property
    def to_string(self) -> str:
        return f'({self.node_0.x} , {self.node_1.x})'

    @property
    def serialise_road(self):
        edge_dict = {
            'node_0':self.node_0.serialise_node,
            'node_1':self.node_1.serialise_node
        }
        return edge_dict

    def reverse(self):
        return Edge(self.node_1, self.node_0)

    def is_equal_to(self, other_edge:object):
        if self.node_0.is_equal_to(other_edge.node_0) and self.node_1.is_equal_to(other_edge.node_1):
            return True
        return False

    def is_in(self, target_list:list) -> bool:
        for edge in target_list:
            if self.is_equal_to(edge):
                return True
        return False
