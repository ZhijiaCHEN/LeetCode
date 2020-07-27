
from heapq import heappushpop
import heapq
from typing import List, Union, Tuple
from data_structure import ListNode, TreeNode
from utility import singly_list, binary_tree

Node = ListNode
class Q2:
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

class Q3:
    def lengthOfLongestSubstring(self, s: str) -> int:
        """
        Given a string, find the length of the longest substring without repeating characters.

        Args:
            s (str): input string

        Returns:
            int: the length of the longest substring without repeating characters
        """

        # Starting from the first character, try to extend a substring without repeating characters as long as possible. Once a repeating character appears, say 'r', the longer non-reapeating substring can only appear after the first 'r'. Thus we try to find the next substring starting from the character after the first 'r' and reapeat the procedure.
        ret  = 0
        sIdx = 0 # the starting index of the non-repeating substring being extended
        c2Idx = {} # a dictionary mapping each character of current substring to its index
        for i,c in enumerate(s):
            if c in c2Idx: # check if the new character exists in the substring
                # the non-repeating substring ends here
                if ret < len(c2Idx):
                    ret = len(c2Idx)

                # we only need to remove characters appears before the firs repeating characters
                rIdx = c2Idx[c]
                for j in range(sIdx, rIdx):
                    c2Idx.pop(s[j])

                # the starting index of the new substring
                sIdx = rIdx + 1
            c2Idx[c] = i
        if ret < len(c2Idx):
            ret = len(c2Idx)
        return ret

class Q11:
    def maxArea(self, height: List[int]) -> int:
        """Given n non-negative integers a1, a2, ..., an , where each represents a point at coordinate (i, ai). n vertical lines are drawn such that the two endpoints of line i is at (i, ai) and (i, 0). Find two lines, which together with x-axis forms a container, such that the container contains the most water.

        Args:
            height (List[int]): line list

        Returns:
            int: maximum area of water
        """
        # For each line, the maximum area of water it can hold is determined by the farthest taller line.
        maxArea = 0
        L = len(height)
        for i, l1 in enumerate(height):
            minDist = int(maxArea/l1)
            if i + minDist < L:
                for j, l2 in enumerate(height[-1:i+minDist:-1]):
                    if l2 >= l1:
                        area = l1 * (L - 1 - i - j)
                        if area > maxArea:
                            maxArea = area
                        break
            if i - minDist > 0:
                for j , l2 in enumerate(height[0:i - minDist]):
                    if l2 >= l1:
                        area = l1 * (i - j)
                        if area > maxArea:
                            maxArea = area
                        break
        return maxArea

class Q19:
    def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:
        pTarget = head
        pAdvance = head
        pPrev = None
        lag = 0
        while pAdvance.next is not None:
            pAdvance = pAdvance.next
            if lag < n-1:
                lag += 1
            else:
                pPrev = pTarget
                pTarget = pTarget.next
        if pPrev is not None:
            pPrev.next = pTarget.next
            return head
        else:
            if pTarget == pAdvance:
                return None
            else:
                return head.next

class Q23:
    def min_node(self, nodes: List[ListNode]) -> ListNode:
        minIdx = 0
        for i in range(1, len(nodes)):
            if nodes[i].val < nodes[minIdx].val:
                minIdx = i
        return minIdx

    def mergeKLists(self, lists: List[ListNode]) -> ListNode:
        shifter = [x for x in lists if x is not None]
        head = ListNode()
        mergeFront = head
        while len(shifter) > 0:
            nextNodeIdx = self.min_node(shifter)
            mergeFront.next = shifter[nextNodeIdx]
            mergeFront = mergeFront.next
            if shifter[nextNodeIdx].next is None:
                del shifter[nextNodeIdx]
            else:
                shifter[nextNodeIdx] = shifter[nextNodeIdx].next
        return head.next

class Q61:
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

class Q92:
    def reverseBetween(self, head: ListNode, m: int, n: int) -> ListNode:
        if (head is None): return head
        thisNode = head
        prvNode = None
        i = 1
        while i < m:
            prvNode = thisNode
            thisNode = thisNode.next
            i += 1
        breakL = prvNode
        breakR = thisNode
        while i <= n:
            nextNode = thisNode.next
            thisNode.next = prvNode
            prvNode = thisNode
            thisNode = nextNode
            i += 1
        if breakL is None:
            head = prvNode
        else:
            breakL.next = prvNode
        breakR.next = thisNode
        return head

class Q138:
    def copyRandomList(self, head: ListNode) -> ListNode:
        if head is None: return None

        nodeDict = {id(None): None}
        ret = Node(head.val)
        nodeDict[id(head)] = ret

        srcNode = head
        dstNode = ret
        while srcNode is not None:
            if not id(srcNode.next) in nodeDict:
                nodeDict[id(srcNode.next)] = Node(srcNode.next.val)
            dstNode.next = nodeDict[id(srcNode.next)]

            if not id(srcNode.random) in nodeDict:
                nodeDict[id(srcNode.random)] = Node(srcNode.random.val)
            dstNode.random = nodeDict[id(srcNode.random)]

            srcNode = srcNode.next
            dstNode = dstNode.next
        return ret

    def copyRandomListV2(self, head: ListNode) -> ListNode:
        if head is None: return None
        
        src = head
        while src is not None:
            dst = Node(src.val)
            dst.next = src.next
            src.next = dst
            src = dst.next
        
        src = head
        dst = head.next
        while True:
            if src.random is not None:
                dst.random = src.random.next
            src = dst.next
            if src is not None:
                dst = src.next
            else:
                break

        src = head
        dst = head.next
        ret = dst
        while True:
            src.next = dst.next
            src = src.next
            if src is not None:
                dst.next = src.next
                dst = src.next
            else:
                break
        return ret

class Q206:
    def reverseList(self, head: ListNode) -> ListNode:
        thisNode = head
        prvNode = None
        while thisNode is not None:
            nextNode = thisNode.next
            thisNode.next = prvNode
            prvNode = thisNode
            thisNode = nextNode
        return prvNode

class Q572:
    def match(self, sNode: TreeNode, tNode: TreeNode) -> bool:
        if sNode is None or tNode is None: 
            return sNode is tNode
        return (sNode.val == tNode.val) and self.match(sNode.left, tNode.left) and self.match(sNode.right, tNode.right)

    def isSubtree(self, s: TreeNode, t: TreeNode) -> bool:
        if self.match(s, t):
            return True
        if s is None: 
            return False
        return self.isSubtree(s.left, t) or self.isSubtree(s.right, t)
    
    def isSubtreeV2(self, s: TreeNode, t: TreeNode) -> bool:
        from hashlib import sha256
        def hash_(x):
            S = sha256()
            S.update(x.encode('utf-8'))
            return S.hexdigest()

        def merkle(node: TreeNode):
            if node is None:
                return 'None'
            mLeft = merkle(node.left)
            mRight = merkle(node.right)
            node.merkle = hash_(mLeft + str(node.val) + mRight)
            return node.merkle
        
        def dfs_search_merkle(node: TreeNode) -> Union[TreeNode, bool]:
            # search node in s that has the same merkle hash as t using dfs
            if node is None: return False
            if node.merkle == t.merkle:
                return node
            matchLeft = dfs_search_merkle(node.left)
            if matchLeft: return matchLeft
            matchRight = dfs_search_merkle(node.right)
            if matchRight: return matchRight
            return False

        merkle(s)
        merkle(t)
        matchNode = dfs_search_merkle(s)
        if matchNode:
            return self.match(matchNode, t)
        else:
            return False

class Q692:
    def topKFrequent(self, words: List[str], k: int) -> List[str]:
        frqDict = {}
        for w in words:
            frqDict[w] = frqDict.get(w, 0) - 1
        topK = sorted(frqDict.items(), key = lambda x: (x[1], x[0]))[:k]
        return [x[0] for x in topK]

class Q42:
    def trap(self, height: List[int]) -> int:
        if len(height) == 0: return 0
        sTrap = sum(height)
        height = sorted([(i, x) for i, x in enumerate(height)], key = lambda x: x[1], reverse = True)
        rIdx = height[0][0]
        lIdx = height[0][0]
        sTotal = height[0][1]
        for i, x in height[1:]:
            if x == 0: continue
            if i > rIdx:
                sTotal += x * (i - rIdx)
                rIdx = i
            elif i < lIdx:
                sTotal += x * (lIdx - i)
                lIdx = i
        return sTotal - sTrap

class Q20:
    def isValid(self, s: str) -> bool:
        parenStack = []
        parenPair = {')': '(', '}': '{', ']': '['}
        for c in s:
            if c in ['(', '[', '{']:
                parenStack.append(c)
            elif c in [')', ']', '}']:
                if (len(parenStack) == 0) or (parenStack[-1] != parenPair[c]):
                    return False
                else:
                    parenStack.pop()

        return len(parenStack) == 0

class Q973:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        """
        Args:
            points (List[List[int]]): list of points
            K (int): top K closest points to return

        Returns:
            List[List[int]]: top K closest points
        """
        import heapq
        distHeap = []
        for x, y in points:
            dist = -(x**2 + y**2)
            if len(distHeap) < K:
                heapq.heappush(distHeap, (dist, x, y))
            else:
                heappushpop(distHeap, (dist, x, y))
        return [[d[1], d[2]] for d in distHeap]
# 49. Group Anagrams
class Q49:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        letr2Idx = {x:i for i, x in enumerate('abcdefghijklmnopqrstuvwxyz')}
        letrCntDict = {}
        for s in strs:
            letrCnt = [0]*26
            for c in s:
                letrCnt[letr2Idx[c]] += 1
            letrCnt = tuple(letrCnt)
            if letrCnt in letrCntDict:
                letrCntDict[letrCnt].append(s)
            else:
                letrCntDict[letrCnt] = [s]
        return [x[1] for x in letrCntDict.items()]

# 239. Sliding Window Maximum
class Q239:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        winMax = max(nums[:k])
        ret = [winMax]
        for i, x in enumerate(nums[k:]):
            i += k
            if nums[i-k] == winMax:
                winMax = max(nums[i-k+1: i+1])
            elif nums[i] > winMax:
                winMax = nums[i]
            ret.append(winMax)
        return ret

# 295. Find Median from Data Stream
class Q295:
    def __init__(self):
        import heapq
        self.median = None
        self.leftMaxHeap = []
        self.rightMinHeap = []
        self.even = True

    def addNum(self, num: int) -> None:
        if self.median is None:
            self.median = num
        else:
            if self.even:
                if num < -self.leftMaxHeap[0]:
                    self.median = -heapq.heappushpop(self.leftMaxHeap, -num)
                elif num > self.rightMinHeap[0]:
                    self.median = heapq.heappushpop(self.rightMinHeap, num)
                else:
                    self.median = num
            else:
                if num < self.median:
                    heapq.heappush(self.leftMaxHeap, -num)
                    heapq.heappush(self.rightMinHeap, self.median)
                else:
                    heapq.heappush(self.leftMaxHeap, -self.median)
                    heapq.heappush(self.rightMinHeap, num)
                self.median = (self.rightMinHeap[0] - self.leftMaxHeap[0])/2
        self.even = not self.even

    def findMedian(self) -> float:
        return self.median

if __name__ == '__main__':
    q = Q295()
    print(q.findMedian())
    q.addNum(1)
    q.addNum(2)
    print(q.findMedian())
    q.addNum(3)
    print(q.findMedian())



