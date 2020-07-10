
from typing import List


class ListNode:
    def __init__(self, val=0, next=None):
        if type(val) == list:
            assert len(val) > 0
            self.val = val[0]
            if len(val) > 1:
                self.next = ListNode(val=val[1:])
            else:
                self.next = None
            
        else:
            self.val = val
            self.next = next
    def print(self):
        if self.next is None:
            print('{}-->'.format(self.val))
        else:
            print('{}-->'.format(self.val), end = '')
            self.next.print()


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
    
    def rotateRight(self, head: ListNode, k: int) -> ListNode:
        if head is None:
            return head
        l = 1
        oldTail = head
        while oldTail.next is not None:
            l += 1
            oldTail = oldTail.next
        
        k = k%l
        if k == 0:
            return head
        p = head
        newHead = head
        newTail = ListNode(val=None, next=head)
        lag = 0
        while p.next is not None:
            p = p.next
            if lag < k - 1:
                lag += 1
            else:
                newHead = newHead.next
                newTail = newTail.next
        newTail.next = None
        oldTail.next = head
        return newHead

if __name__ == '__main__':
    s = Solution()
    L = ListNode(val=[1, 2, 3, 4, 5])
    k = 1
    nL = s.rotateRight(L, k)
    nL.print()
