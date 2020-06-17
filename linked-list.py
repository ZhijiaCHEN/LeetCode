
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        ret = None
        prev = None
        carry = False
        while (l1 is not None) or (l2 is not None) or carry:
            if carry:
                sum = 1
                carry = False
            else:
                sum = 0
            if l1 is not None:
                sum += l1.val
                l1 = l1.next
            if l2 is not None:
                sum += l2.val
                l2 = l2.next
            
            if sum >= 10:
                sum -= 10
                carry = True
            this = ListNode(val=sum)
            if prev is None:
                ret = this
                prev = this
            else:
                prev.next = this
                prev = prev.next

        return ret