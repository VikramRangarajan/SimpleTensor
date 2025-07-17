from ..tensor import Tensor
from itertools import chain

# In the computation graph, parent ---> child

class OpNode:
    def __init__(self, op: str):
        self.op: str = op
        self.parents: list["TensorNode"] = []
        self.children: list["TensorNode"] = []
        self.metadata: dict = {}
        self.name: str = "" # TODO: Identifier

    def __call__(self): ...

class TensorNode:
    def __init__(self, tensor: Tensor, parent, children, requires_grad):
        self.tensor: Tensor = tensor
        self.parent: OpNode = parent
        self.children: list[OpNode] = children
        self.requires_grad: bool = requires_grad
        self.variable: TensorNode | None = None
        self.gradient: TensorNode | None = None

    def get_roots(self) -> list["TensorNode"]:
        # All TensorNode's with no children
        root_nodes: list[TensorNode] = []
        visited: set[str] = set()
        stack: list[TensorNode] = [self]
        while len(stack) > 0:
            current_node = stack.pop()
            curr_name = current_node.tensor.name
            if curr_name not in visited:
                visited.add(curr_name)

                # Children first then parents to prevent potential cycles
                all_children_tensors = chain(*(child.children for child in current_node.children))
                children_tensors = [x for x in all_children_tensors if x.tensor.name not in visited]
                stack.extend(children_tensors)

                # Check if current node is root
                if len(current_node.children) == 0:
                    root_nodes.append(current_node)

                # Handle parents
                parent_tensors = [x for x in current_node.parent.parents if x.tensor.name not in visited]
                stack.extend(parent_tensors)
        return root_nodes

    def topological_sort(self) -> list[OpNode]:
        """
        Can be run on any node in the graph
        TODO: FIX
        """
        stack = [x.parent for x in self.get_roots()]
        visited: set[str] = set()
        sorted_list: list[OpNode] = []
        while len(stack) > 0:
            current_node = stack.pop()
            curr_name = current_node.name
            if curr_name not in visited:
                visited.add(curr_name)
                sorted_list.append(current_node)
        return sorted_list
