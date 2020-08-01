
from heapq import heappushpop
import heapq
from typing import ChainMap, List, Union, Tuple
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

# 253. Meeting Rooms II
class Q253:
    def minMeetingRooms(self, intervals: List[List[int]]) -> int:
        startTime = sorted([x[0] for x in intervals])
        endTime = sorted(x[1] for x in intervals)
        rmNum = 0
        etIdx = 0
        for st in startTime:
            if st < endTime[etIdx]:
                rmNum += 1
            else:
                etIdx += 1
        return rmNum

# 380. Insert Delete GetRandom O(1)
class RandomizedSet:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        from random import randint
        self.data = dict()

    def insert(self, val: int) -> bool:
        """
        Inserts a value to the set. Returns true if the set did not already contain the specified element.
        """
        if val in self.data:
            return False
        else:
            self.data[val] = val
            return True

    def remove(self, val: int) -> bool:
        """
        Removes a value from the set. Returns true if the set contained the specified element.
        """
        if val in self.data:
            self.data.pop(val)
            return True
        else:
            return False

    def getRandom(self) -> int:
        """
        Get a random element from the set.
        """
        return self.data[list(self.data.keys())[randint(0, len(self.data)-1)]]

# 987. Vertical Order Traversal of a Binary Tree
class Q987:
    def verticalTraversal(self, root: TreeNode) -> List[List[int]]:
        from queue import PriorityQueue
        posCor = []
        negCor = []
        def visit_fun(node: TreeNode, x: int, y: int):
            if node is None: return
            if x >= 0:
                if x + 1 > len(posCor):
                    posCor.append([(y, node.val)])
                else:
                    posCor[x].append((y, node.val))
            else:
                if abs(x) > len(negCor):
                    negCor.append([(y, node.val)])
                else:
                    negCor[abs(x) - 1].append((y, node.val))

            visit_fun(node.left, x - 1, y + 1)
            visit_fun(node.right, x + 1, y + 1)

        visit_fun(root, 0, 0)
        return [[y[1] for y in sorted(x)] for x in negCor[-1::-1]] + [[y[1] for y in sorted(x)] for x in posCor]

# 937. Reorder Data in Log Files
class Q937:
    def reorderLogFiles(self, logs: List[str]) -> List[str]:
        digLog = []
        letLog = []
        for log in logs:
            spaceIdx = log.find(' ')
            if log[spaceIdx + 1] in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']:
                digLog.append(log)
            else:
                letLog.append((log[spaceIdx:], log[:spaceIdx]))
        letLog = [x[1] + x[0] for x in sorted(letLog)]
        return letLog + digLog

# 146. LRU Cache
class LRUCache:
    class CacheNode:
        def __init__(self, key: int, val: int, next = None, prev = None) -> None:
            self.key = key
            self.val = val
            self.next = next
            self.prev = prev
        def print(self):
            print('({}: {})->'.format(self.key, self.val), end = '')
            p = self.next
            while p is not None:
                print('({}: {})->'.format(p.key, p.val), end = '')
                p = p.next
            print('NULL')
    def __init__(self, capacity: int):
        
        self.cacheMap = dict()
        self.LRU = None
        self.MRU = None
        self.capacity = capacity

    def _update(self, node):
        if node == self.LRU and node.key not in self.cacheMap:
            # node evict
            if self.capacity > 1:
                self.LRU = self.LRU.prev
                self.LRU.next = None

                node.next = self.MRU
                self.MRU.prev = node
                self.MRU = node
                self.MRU.prev = None
            self.cacheMap[node.key] = node
        elif node.key in self.cacheMap:
            # node get or update
            if len(self.cacheMap) > 1:
                if node == self.LRU:
                    self.LRU = self.LRU.prev
                    self.LRU.next = None

                    node.prev = node
                    node.next = self.MRU
                    self.MRU.prev = node
                    self.MRU = node
                    self.MRU.prev = None
                elif node != self.MRU:
                    node.next.prev = node.prev
                    node.prev.next = node.next
                    node.next = self.MRU
                    self.MRU.prev = node
                    self.MRU = node
                    self.MRU.prev = None
        else:
            # new node
            if len(self.cacheMap) == 0:
                self.LRU = node
                self.MRU = node
            else:
                node.next = self.MRU
                self.MRU.prev = node
                self.MRU = node
            self.cacheMap[node.key] = node

    def get(self, key: int) -> int:
        if key in self.cacheMap:
            node = self.cacheMap[key]
            self._update(node)
            return node.val
        else:
            return -1

    def put(self, key: int, value: int) -> None:
        if key in self.cacheMap:
            # update cache
            node = self.cacheMap[key]
            node.val = value
        else:
            # new cache
            if len(self.cacheMap) == self.capacity:
                # need an eviction
                node = self.cacheMap.pop(self.LRU.key)
                node.key = key
                node.val = value
            else:
                #create a new node
                node = self.CacheNode(key, value)
        self._update(node)

def lru_cache_test(input: List[List[int]], expected: List[Union[int, None]]):
    cache = LRUCache(input[0][0])
    for i, (x, y) in enumerate(zip(input[1:], expected[1:])):
        # if i == 53:
        #     print('')
        if len(x) == 1:
            #print('get {}'.format(x[0]))
            yOut = cache.get(x[0])
        else:
            #print('put {}'.format(x))
            yOut = cache.put(x[0], x[1])
        if yOut == y or yOut is y:
            cache.MRU.print()
        else:
            print('error')

# 273. Integer to English Words
# class Q273:
#     def numberToWords(self, num: int) -> str:
#         singleDigit2Word = {'0': 'Zero', '1': 'One', '2': 'Two', '3': 'Three', '4': 'Four', '5': 'Five', '6': 'Six', '7': 'Seven', '8': 'Eight', '9': 'Nine'}
#         doubleDigit2Word = {'0': 'And', '2': 'Twenty', '3': 'Thirty', '4': 'Forty', '5': 'Fifty', '6': 'Sixty', '7': 'Seventy', '8': 'Eighty', '9': 'Ninety'}
#         teens2Word = {'0': 'teen', '1': 'Eleven', '2': 'Twelve', '3': 'Thirteen', '4': 'Fourteen', '5': 'Fifteen', '6': 'Sixteen', '7': 'Seventeen', '8': 'Eightteen', '9': 'Nineteen'}
#         units = ['', 'Thousand', 'Million', 'Billion']
#         out = []
#         num = str(num)[-1::-1]
#         i = 0
#         while i < len(num):
#             if i%3 == 0:
#                 if (i + 1) < len(num) and num[i + 1] == '1':
#                     out.append(teens2Word[num[i]])
#                     i += 1
#                 else:
#                     out.append(singleDigit2Word[num[i]])
#                 i += 1
#                 out.append(units[int(i/3)])
#             elif i%3 == 1:

# 21. Merge Two Sorted Lists
class Q21:
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        if (l1 is None) and (l2 is None):
            return None
        p1 = l1
        p2 = l2
        mp = ListNode(-1)
        ret = mp
        while(p1 or p2):
            if p1 is None:
                mp.next = p2
                p2 = p2.next
            elif p2 is None:
                mp.next = p1
                p1 = p1.next
            else:
                if p1.val < p2.val:
                    mp.next = p1
                    p1 = p1.next
                else:
                    mp.next = p2
                    p2 = p2.next
            mp = mp.next
        return ret.next

# 863. All Nodes Distance K in Binary Tree
class Q863:
    def distanceK(self, root: TreeNode, target: TreeNode, K: int) -> List[int]:
        adjacentMap = {}
        distK = []
        def tree_visit(parentNode: TreeNode, currentNode: TreeNode):
            adjacentMap[currentNode.val] = set()
            
            if parentNode is not None:
                adjacentMap[currentNode.val].add(parentNode.val)
            if currentNode.left is not None:
                adjacentMap[currentNode.val].add(currentNode.left.val)
                tree_visit(currentNode, currentNode.left)
            if currentNode.right is not None:
                adjacentMap[currentNode.val].add(currentNode.right.val)
                tree_visit(currentNode, currentNode.right)
            
        def graph_visit(nodeVal: int, distance: int):
            if distance == K:
                distK.append(nodeVal)
            else:
                for val in adjacentMap[nodeVal]:
                    adjacentMap[val].remove(nodeVal)
                    graph_visit(val, distance + 1)
        
        tree_visit(None, root)
        graph_visit(target.val, 0)

        return distK

# 269. Alien Dictionary
class Q269:
    class MyNode:
        def __init__(self, c: str) -> None:
            self.inBound = set()
            self.outBound = set()
            self.val = c

    def alienOrder(self, words: List[str]) -> str:
        letGraph = {c: self.MyNode(c) for w in words for c in w}
        for w1, w2 in zip(words, words[1:]):
            for c1, c2 in zip(w1, w2):
                if c1 != c2:
                    letGraph[c1].outBound.add(c2)
                    letGraph[c2].inBound.add(c1)
                    break

        letSorted = ""
        for x in list(letGraph.keys()):
            node = letGraph[x]
            if len(node.inBound) + len(node.outBound) == 0:
                letGraph.pop(x)

        while len(letGraph) > 0:
            cycle = True
            for x in list(letGraph.keys()):
                if len(letGraph[x].inBound) == 0:
                    cycle = False
                    node = letGraph.pop(x)
                    letSorted += x
                    for y in node.outBound:
                        letGraph[y].inBound.remove(x)
            if cycle:
                return ""
        return letSorted

# 297. Serialize and Deserialize Binary Tree
class Codec:
    def serialize(self, root):
        """Encodes a tree to a single string.
        
        :type root: TreeNode
        :rtype: str
        """
        from collections import deque
        visitQ = deque([root])
        ret = []
        while len(visitQ) > 0:
            node = visitQ.pop()
            if node is None:
                ret.append("null")
            else:
                ret.append(str(node.val))
                visitQ.appendleft(node.left)
                visitQ.appendleft(node.right)
        endIdx = len(ret) - 1
        while endIdx >= 0 and ret[endIdx] == "null":
            endIdx -= 1
        return "[{}]".format(','.join(ret[:endIdx+1]))


    def deserialize(self, data):
        """Decodes your encoded data to tree.
        
        :type data: str
        :rtype: TreeNode
        """
        from collections import deque
        def str2node(s: str):
            if s == "null":
                return None
            else:
                return TreeNode(val = int(s))
        data = data[1:-1]
        if len(data) == 0:
            return None
        else:
            data = deque(data.split(',')[-1::-1])
        root = str2node(data.pop())
        visitQ = deque([root])
        while len(visitQ) > 0:
            node = visitQ.pop()
            if len(data) > 0:
                node.left = str2node(data.pop())
            else:
                node.left = str2node("null")
            if len(data) > 0:
                node.right = str2node(data.pop())
            else:
                node.right = str2node("null")
            if node.left:
                visitQ.appendleft(node.left)
            if node.right:
                visitQ.appendleft(node.right)
        return root

if __name__ == '__main__':
    q = Codec()
    print(q.serialize(None))
    print(q.serialize(q.deserialize("[]")))
    # q.put(4,4)
    # print(q.get(1))
    # print(q.get(3))
    # print(q.get(4))



