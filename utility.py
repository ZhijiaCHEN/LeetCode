from typing import List, Union, Tuple
from data_structure import ListNode, TreeNode

def singly_list(nodes: Union[List[int], List[list]] = []) -> Union[ListNode, None]:
    nextNode = None
    for e in nodes[::-1]:
        nextNode = ListNode(e, next = nextNode)
    return nextNode

def binary_tree(array: List) -> Union[TreeNode, None]:
    """return the binary tree represented in array using preorder traversal order. For example [1, 2, 4, None, None, 5, None, None, 3, None, None] returns the following tree:
         1
        / \
       2   3
      / \
     4   5
    Args:
        array (List, optional): array representation of the output binary tree. Defaults to [].

    Returns:
        Union[TreeNode, None]: root node of the binary tree.
    """
    
    if len(array) == 0:
        return None
    root = TreeNode(val = array[0])
    path = [root]
    goLeft = [True]
    for x in array[1:]:
        currentNode = path[-1]
        if x is None:
            if goLeft[-1]:
                goLeft[-1] = False
            else:
                goLeft.pop()
                path.pop()
                while((len(path) > 0) and (not goLeft[-1]) and (path[-1].right is not None)):
                    goLeft.pop()
                    path.pop()
        else:
            if goLeft[-1]:
                currentNode.left = TreeNode(val = x)
                path.append(currentNode.left)
                goLeft[-1] = False
                goLeft.append(True)
            else:
                currentNode.right = TreeNode(val = x)
                path.append(currentNode.right)
                goLeft.append(True)
    return root

