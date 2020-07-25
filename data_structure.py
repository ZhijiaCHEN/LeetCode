from typing import List, Union, Tuple
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
    
    def print(self)->None:
        """Print the tree with preorder traversal.
        """
        print('{}, '.format(self.val), end = '')
        if self.left is None:
            print('None, ', end = '')
        else:
            self.left.print()
        if self.right is None:
            print('None, ', end = '')
        else:
            self.right.print()

class ListNode:
    def __init__(self, data: Union[int, Tuple[int, Union[int, None]], List[Union[int, None]]] = 0, next: 'ListNode' = None, random: 'ListNode' = None):
        if type(data) in [tuple, list]:
            assert len(data) == 2
            self.val = data[0]
            self.random =data[1]
        else:
            self.val = data
        self.next = next

    def print(self):
        print('{}->'.format(self.val), end = '')
        p = self.next
        while p is not None:
            print('{}->'.format(p.val), end = '')
            p = p.next
        print('NULL')