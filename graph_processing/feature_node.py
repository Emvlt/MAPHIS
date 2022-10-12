from dataclasses import dataclass
from typing import Dict

@dataclass
class Node():
    def __init__(self, x, y, label=None, key=None) -> None:
        self.x:int = x
        self.y:int = y
        self.label:str = label
        self.key:str = key

    def is_equal_to(self, other_node: object) -> bool:
        if self.key == other_node.key:
            return True
        return False

    def to_string(self, inv_map:Dict) -> str:
        return f'x : {self.x}, y : {self.y}, class : { inv_map[int(self.label)]}, key : {self.key}'
        
    @property
    def serialise_node(self) -> Dict:
        node_dict = {
            'x':self.x,
            'y':self.y,
            'label':self.label,
            'key':self.key
        }
        return node_dict

    def scale_coordinates(self, ratio:float):
        return Node(int(ratio*self.x), int(ratio*self.y), label=self.label, key=self.key)

    def shift(self, width_shift:int, height_shift:int) -> object:
        return Node(self.x+width_shift, self.y+height_shift, label=self.label, key=self.key)

    def change_key(self, new_key:str) -> object:
        return Node(self.x, self.y, label=self.label, key=new_key)

    def is_in(self, list_of_nodes:list) -> bool:
        for node in list_of_nodes:
            if self.is_equal_to(node):
                return True
        return False

    def intersect(self, other_node:object, background):
        if abs(other_node.y - self.y) < abs(other_node.x - self.x):
            if self.x > other_node.x:
                return plotLineLow(other_node.x, other_node.y, self.x, self.y, background)
            else:
                return plotLineLow(self.x, self.y, other_node.x, other_node.y, background)

        else:
            if self.y > other_node.y:
                return plotLineHigh(other_node.x, other_node.y, self.x, self.y, background)
            else:
                return plotLineHigh(self.x, self.y, other_node.x, other_node.y, background)

def plotLineLow(x0:int, y0:int, x1:int, y1:int, background):
    cost = 0
    dx = x1 - x0
    dy = y1 - y0
    yi = 1
    if dy < 0:
        yi = -1
        dy = -dy
    D = (2 * dy) - dx
    y = y0
    for x in range(x0,x1):
        if background[x,y]!=0 and 20<=cost:
            return True
        elif background[x,y]!=0 and cost<=20:
            cost+=1
            
        if D > 0:
            y = y + yi
            D = D + (2 * (dy - dx))
        else:
            D = D + 2*dy
    return False

def plotLineHigh(x0:int, y0:int, x1:int, y1:int, background):
    cost = 0
    dx = x1 - x0
    dy = y1 - y0
    xi = 1
    if dx < 0:
        xi = -1
        dx = -dx
    D = (2 * dx) - dy
    x = x0
    for y in range(y0,y1):
        if background[x,y]!=0 and 20<=cost:
            return True
        elif background[x,y]!=0 and cost<=20:
            cost+=1
        if D > 0:
            x = x + xi
            D = D + (2 * (dx - dy))
        else:
            D = D + 2*dx
    return False





