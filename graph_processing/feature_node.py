from dataclasses import dataclass
from typing import Dict

@dataclass
class Node():
    def __init__(self, pos_x, pos_y, label=None, key=None) -> None:
        self.pos_x:int = pos_x
        self.pos_y:int = pos_y
        self.label:str = label
        self.key:str = key

    def is_equal_to(self, other_node: object) -> bool:
        if self.key == other_node.key:
            return True
        return False

    def to_string(self, inv_map:Dict) -> str:
        return f'pos_x : {self.pos_x}, pos_y : {self.pos_y}, class : { inv_map[int(self.label)]}, key : {self.key}'

    @property
    def serialise_node(self) -> Dict:
        node_dict = {
            'pos_x':self.pos_x,
            'pos_y':self.pos_y,
            'label':self.label,
            'key':self.key
        }
        return node_dict

    def scale_coordinates(self, ratio:float):
        return Node(int(ratio*self.pos_x), int(ratio*self.pos_y), label=self.label, key=self.key)

    def shift(self, width_shift:int, height_shift:int) -> object:
        return Node(self.pos_x+width_shift, self.pos_y+height_shift, label=self.label, key=self.key)

    def change_key(self, new_key:str) -> object:
        return Node(self.pos_x, self.pos_y, label=self.label, key=new_key)

    def is_in(self, list_of_nodes:list) -> bool:
        for node in list_of_nodes:
            if self.is_equal_to(node):
                return True
        return False

    def intersect(self, other_node:object, background):
        if abs(other_node.pos_y - self.pos_y) < abs(other_node.pos_x - self.pos_x):
            if self.pos_x > other_node.pos_x:
                return plot_line_low(other_node.pos_x, other_node.pos_y, self.pos_x, self.pos_y, background)
            return plot_line_low(self.pos_x, self.pos_y, other_node.pos_x, other_node.pos_y, background)


        if self.pos_y > other_node.pos_y:
            return plot_line_high(other_node.pos_x, other_node.pos_y, self.pos_x, self.pos_y, background)
        return plot_line_high(self.pos_x, self.pos_y, other_node.pos_x, other_node.pos_y, background)

def plot_line_low(pos_x0:int, pos_y0:int, pos_x1:int, pos_y1:int, background):
    cost = 0
    diff_x = pos_x1 - pos_x0
    diff_y = pos_y1 - pos_y0
    pos_yi = 1
    if diff_y < 0:
        pos_yi = -1
        diff_y = -diff_y
    delta = (2 * diff_y) - diff_x
    pos_y = pos_y0
    for pos_x in range(pos_x0,pos_x1):
        if background[pos_x,pos_y]!=0 and 20<=cost:
            return True
        cost+=1

        if delta > 0:
            pos_y = pos_y + pos_yi
            delta = delta + (2 * (diff_y - diff_x))
        else:
            delta = delta + 2*diff_y
    return False

def plot_line_high(pos_x0:int, pos_y0:int, pos_x1:int, pos_y1:int, background):
    cost = 0
    diff_x = pos_x1 - pos_x0
    diff_y = pos_y1 - pos_y0
    pos_xi = 1
    if diff_x < 0:
        pos_xi = -1
        diff_x = -diff_x
    delta = (2 * diff_x) - diff_y
    pos_x = pos_x0
    for pos_y in range(pos_y0,pos_y1):
        if background[pos_x,pos_y]!=0 and 20<=cost:
            return True
        cost+=1
        if delta > 0:
            pos_x = pos_x + pos_xi
            delta = delta + (2 * (diff_x - diff_y))
        else:
            delta = delta + 2*diff_x
    return False
