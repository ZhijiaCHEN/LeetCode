
from collections import deque, namedtuple
from heapq import heappushpop, nsmallest
import heapq
from typing import ChainMap, Collection, List, Union, Tuple
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

# 1192. Critical Connections in a Network
class Q1192:
    def criticalConnections(self, n: int, connections: List[List[int]]) -> List[List[int]]:
        adjacentMap = {i: [] for i in range(n)}
        for c in connections:
            adjacentMap[c[0]].append(c[1])
            adjacentMap[c[1]].append(c[0])
        minRank = {x: -1 for x in adjacentMap}
        criCon = []
        def dfs(s: int, rank: int, p: int) -> int:
            #global minRank
            minRank[s] = rank
            for c in adjacentMap[s]:
                if c == p: continue
                if minRank[c] >= 0:
                    cRank = minRank[c]
                else:
                    cRank = dfs(c, rank + 1, s)
                minRank[s] = min(minRank[s], cRank)
            if minRank[s] >= rank and p >= 0:
                criCon.append([p, s])

            return minRank[s]
        dfs(0, 0, -1)
        return criCon

# 200. Number of Islands
class Q200:
    def numIslands(self, grid: List[List[str]]) -> int:
        if len(grid) == 0 or len(grid[0]) == 0:
            return 0
        rNum = len(grid)
        cNum = len(grid[0])
        def dfs(r: int, c: int):
            if r >= 0 and r < rNum and c >= 0 and c < cNum:
                if grid[r][c] == "1":
                    grid[r][c] = 0
                    dfs(r + 1, c)
                    dfs(r, c + 1)
                    dfs(r - 1, c)
                    dfs(r, c - 1)
        ret = 0
        for r in range(rNum):
            for c in range(cNum):
                if grid[r][c] == "1":
                    ret += 1
                    dfs(r, c)
        
        return ret

# 994. Rotting Oranges
class Q994:
    def orangesRotting(self, grid: List[List[int]]) -> int:
        from collections import deque
        if len(grid) == 0 or len(grid[0]) == 0:
            return 0
        rNum = len(grid)
        cNum = len(grid[0])
        freshCnt = 0
        rotten = deque()
        for r in range(rNum):
            for c in range(cNum):
                if grid[r][c] == 1:
                    freshCnt += 1
                elif grid[r][c] == 2:
                    rotten.append((r, c))
        rottenTime = 0
        
        while rotten and freshCnt > 0:
            rottenTime += 1
            for _ in range(len(rotten)):
                r, c = rotten.popleft()
                for dr, dc in [(-1, 0), (0, -1), (1, 0), (0, 1)]:
                    rr = r + dr
                    cc = c + dc
                    if rr < 0 or rr == rNum or cc < 0 or cc == cNum: continue
                    if grid[rr][cc] == 1:
                        grid[rr][cc] = 2
                        freshCnt -= 1
                        rotten.append((rr, cc))
        if freshCnt > 0:
            return -1
        else:
            return rottenTime

# 773. Sliding Puzzle
class Q773:
    class MyNode:
        def __init__(self, board: List[List[int]]) -> None:
            self.key = tuple(board[0] + board[1])
            self.child = []

    def generate_graph(self, root):
        from collections import deque
        q = deque([root])
        existingState = set([root.key])
        while len(q) > 0:
            state = q.popleft()
            zeroIdx = state.key.index(0)
            if zeroIdx > 2:
                r = 1
            else:
                r = 0
            c = zeroIdx % 3
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1 ,0)]:
                rr = r + dr
                cc = c + dc
                if rr >= 0 and rr < 2 and cc >= 0 and cc < 3:
                    nextState = [list(state.key[:3]), list(state.key[3:])]
                    nextState[r][c] = nextState[rr][cc]
                    nextState[rr][cc] = 0
                    nextState = self.MyNode(nextState)
                    if nextState.key in existingState: continue
                    existingState.add(nextState.key)
                    state.child.append(nextState)
                    q.append(nextState)

    def slidingPuzzle(self, board: List[List[int]]) -> int:
        from collections import deque
        target = (1,2,3,4,5,0)
        root = self.MyNode(board)
        if root.key == target: return 0
        self.generate_graph(root)

        q = deque([root])
        dist = -1
        while len(q) > 0:
            dist += 1
            for _ in range(len(q)):
                state = q.popleft()
                if state.key == target: return dist
                for child in state.child:
                    q.append(child)
        return -1

# 4. Median of Two Sorted Arrays
class Q4:
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        from math import ceil, floor
        if len(nums2) < len(nums1):
            tmp = nums1
            nums1 = nums2
            nums2 = tmp
        l1 = len(nums1)
        l2 = len(nums2)
        l = l1 + l2

        # if l1 == 0:
        #     if len(nums2) % 2 == 0:
        #         return (nums2[int(l2/2)] + nums2[int(l2/2)-1])/2
        #     else:
        #         return nums2[int((l2-1)/2)]

        iMin = 0
        iMax = l1
        halfL = floor((l1 + l2 + 1)/2)
        while True:
            i = floor((iMin + iMax)/2)
            j = halfL - i
            if i < l1 and nums1[i] < nums2[j - 1]:
                iMin = i + 1
            elif i > 0 and nums1[i - 1] > nums2[j]:
                iMax = i - 1
            else:
                if i == 0:
                    maxL = nums2[j - 1]
                elif j == 0:
                    maxL = nums1[i - 1] 
                else:
                    maxL = max(nums1[i - 1], nums2[j - 1])

                if l % 2 == 1:
                    return maxL
                
                if i == l1:
                    minR = nums2[j]
                elif j == l2:
                    minR = nums1[i]
                else:
                    minR = min(nums1[i], nums2[j])

                return (maxL + minR)/2

# 53. Maximum Subarray
class Q53:
    def maxSubArray(self, nums: List[int]) -> int:
        ret = nums[0]
        currentSum = 0
        for x in nums:
            currentSum += x
            if currentSum > ret:
                ret = currentSum
            if currentSum < 0:
                currentSum = 0
        return ret

# 124. Binary Tree Maximum Path Sum
class Q124:
    def maxPathSum(self, root: TreeNode) -> int:
        def dfs(node: TreeNode):
            from math import inf
            if node is None: return
            dfs(node.left)
            dfs(node.right)

            cumSumL = node.val
            cumSumR = node.val
            maxSumL = -inf
            maxSumR = -inf
            if node.left:
                cumSumL += node.left.cumSum
                maxSumL = node.left.maxSum
            if node.right:
                cumSumR += node.right.cumSum
                maxSumR = node.right.maxSum
            node.cumSum = max(cumSumL, cumSumR, 0)
            node.maxSum = max(node.val, cumSumL, cumSumR, cumSumL + cumSumR - node.val, maxSumL, maxSumR)
        
        dfs(root)
        return root.maxSum

# 33. Search in Rotated Sorted Array
class Q33:
    def search(self, nums: List[int], target: int) -> int:
        if len(nums) == 0:
            return -1
        iMin = 0
        iMax = len(nums) - 1
        iPiv = -1
        while iMax - iMin > 1:
            iMid = int((iMin+iMax+1)/2)
            if nums[iMid] > nums[0]:
                iMin = iMid
            elif nums[iMid - 1] < nums[0]:
                iMax = iMid
            else:
                iPiv = iMid
                break
        if iPiv == -1:
            if iMin > 0 and nums[iMin] < nums[0] and nums[iMin - 1] >= nums[0]:
                iPiv = iMin
            elif iMax > 0 and nums[iMax] < nums[0] and nums[iMax - 1] >= nums[0]:
                iPiv = iMax
        if iPiv > 0:
            nums = nums[iPiv:] + nums[:iPiv]
        iMin = 0
        iMax = len(nums) - 1
        iTarget = -1
        while iMax - iMin > 1:
            iMid = int((iMin+iMax+1)/2)
            if nums[iMid] > target:
                iMax = iMid
            elif nums[iMid] < target:
                iMin = iMid
            else:
                iTarget = iMid
                break

        if iTarget == -1:
            if nums[iMin] == target:
                iTarget = iMin
            elif nums[iMax] == target:
                iTarget = iMax
            else:
                return -1
        if iPiv == -1:
            return iTarget
        shiftL = len(nums) - iPiv
        if iTarget < shiftL:
            iTarget += iPiv
        else:
            iTarget -= shiftL
        return iTarget

# 981. Time Based Key-Value Store
class TimeMap:
    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.keyMap = {}
    def set(self, key: str, value: str, timestamp: int) -> None:
        if key in self.keyMap:
            self.keyMap[key].append((timestamp, value))
        else:
            self.keyMap[key] = [(timestamp, value)]

    def get(self, key: str, timestamp: int) -> str:
        if key not in self.keyMap:
            return ""
        else:
            q = self.keyMap[key]
            if q[0][0] > timestamp: return ""
            iMin = 0
            iMax = len(q) - 1
            iMid = 0
            while iMax - iMin > 1:
                iMid = int((iMin + iMax + 1)/2)
                if q[iMid][0] > timestamp:
                    iMax = iMid
                elif q[iMid][0] < timestamp:
                    iMin = iMid
            if q[iMin][0] <= timestamp and q[iMid][0] > timestamp:
                return q[iMin][1]
            if q[iMax][0] <= timestamp:
                return q[iMax][1]
            return q[iMid][1]

# 199. Binary Tree Right Side View
class Q199:
    def rightSideView(self, root: TreeNode) -> List[int]:
        ret = []
        def dfs(node: TreeNode, depth: int, nextDepth: int):
            if node is None:
                return nextDepth
            if depth == nextDepth:
                ret.append(node.val)
                nextDepth += 1
            nextDepth = dfs(node.right, depth + 1, nextDepth)
            nextDepth = dfs(node.left, depth + 1, nextDepth)
            
            return nextDepth
        dfs(root, 0, 0)
        return ret

# 212. Word Search II
class Q212:
    def findWords(self, board: List[List[str]], words: List[str]) -> List[str]:
        if len(board) == 0 or len(board[0]) == 0 or len(words) == 0:
            return []
        trie ={}
        rNum = len(board)
        cNum = len(board[0])
        EDW = "$"
        for word in words:
            node = trie
            for char in word:
                node = node.setdefault(char, {})
            node[EDW] = word
        
        ret = []
        def backtrack(r: int, c: int, parent: dict):
            char = board[r][c]
            node = parent[char]
            word = node.pop(EDW, False)
            if word:
                ret.append(word)

            board[r][c] = '#'
            for dr, dc in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
                rr = r + dr
                cc = c + dc
                if rr >= 0 and rr < rNum and cc >= 0 and cc < cNum:
                    if board[rr][cc] in node:
                        backtrack(rr, cc, node)
            board[r][c] = char
            if len(node) == 0:
                parent.pop(char)
        for r in range(rNum):
            for c in range(cNum):
                if board[r][c] in trie:
                    backtrack(r, c, trie)
        return ret

# 236. Lowest Common Ancestor of a Binary Tree
class Q236:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        pHit, qHit = False, False
        pPath = []
        qPath = []
        path = []
        def dfs(node: TreeNode):
            nonlocal pHit, qHit, path, pPath, qPath
            path.append(node)
            if node.val == p.val:
                pHit = True
                pPath = [x for x in path]
            if node.val == q.val:
                qHit = True
                qPath = [x for x in path]
            if not (pHit and qHit):
                if node.left:
                    dfs(node.left)
                if node.right:
                    dfs(node.right)
            path.pop()
        dfs(root)

        for i, (pp, qq) in enumerate(zip(pPath, qPath)):
            if pp != qq:
                return pPath[i - 1]
        if len(pPath) < len(qPath):
            return pPath[-1]
        else:
            return qPath[-1]

# 472. Concatenated Words
class Q472:
    def findAllConcatenatedWordsInADict(self, words: List[str]) -> List[str]:
        trie = {}
        words.sort(key = lambda x: len(x))
        EOW = '$'
        for word in words:
            node = trie
            for char in word:
                node = node.setdefault(char, {})
            node[EOW] = {}

        concatNum = {'': 0}
        def concat_num(word: str) -> bool:
            if word in concatNum:
                return concatNum[word]

            node = trie
            for i, char in enumerate(word):
                if char not in node:
                    return 0
                node = node[char]
                if EOW in node:
                    n = concat_num(word[i + 1:])
                    if n > 0:
                        return n + 1
            if EOW in node: 
                return 1
            else:
                return 0

        ret = []
        for word in words:
            if len(word) == 0: continue
            n = concat_num(word)
            concatNum[word] = n
            if n > 1:
                ret.append(word)
        return ret

# 126. Word Ladder II
class Q126:
    class WordNode:
        def __init__(self, word: str) -> None:
            self.word = word
            self.used = False
    def findLadders(self, beginWord: str, endWord: str, wordList: List[str]) -> List[List[str]]:
        from collections import deque
        trie = {}
        for word in [beginWord] + wordList:
            node = trie
            for char in word:
                node = node.setdefault(char, {})
        word2Node = {w: self.WordNode(w) for w in [beginWord] + wordList}
        adjacentMap = {}
        # def backtrack(charIdx: int, matched: str, parent: dict, wildAvail: bool) -> Union[str, None]:
        #     nonlocal q, matchingWord
        #     if wildAvail:
        #         for char in parent:
        #             node = parent[char]
        #             if char == matchingWord[charIdx]: continue 
        #             if len(node) == 0:
        #                 matched += char
        #                 if matched not in existingWords:
        #                     q.appendleft(matched)
        #                     adjacentMap[matchingWord].append(matched)
        #                 return
        #             else:
        #                 backtrack(charIdx + 1, matched + char, node, False)

        #         char = matchingWord[charIdx]
        #         if char in parent:
        #             node = parent[char]
        #             if len(node) == 0:
        #                 parent.pop(char)
        #             else:
        #                 backtrack(charIdx + 1, matched + char, node, True)
        #                 if len(node) == 0:
        #                     parent.pop(char)
        #     else:
        #         char = matchingWord[charIdx]
        #         matched += char
        #         if char not in parent: return
        #         node = parent[char]
        #         if len(node) == 0:
        #             if matched not in existingWords:
        #                 q.appendleft(matched)
        #                 adjacentMap[matchingWord].append(matched)
        #                 return
        #         else:
        #             backtrack(charIdx + 1, matched, node, False)
        def backtrack(charIdx: int, matched: str, parent: dict, wildAvail: bool) -> None:
            nonlocal matchingWord
            for char in parent:
                node = parent[char]
                if char == matchingWord[charIdx]:
                    if len(node) == 0:
                        if not wildAvail:
                            adjacentMap[matchingWord].append(word2Node[matched + char])
                    else:
                        backtrack(charIdx + 1, matched + char, node, wildAvail)
                else:
                    if wildAvail:
                        if len(node) == 0:
                            adjacentMap[matchingWord].append(word2Node[matched + char])
                        else:
                            backtrack(charIdx + 1, matched + char, node, False)

        # def backtrack(thisChar: str, nextCharIdx: int, matched: str, parent: dict, wildAvail: bool) -> Union[str, None]:
        #     nonlocal q, matchingWord
        #     node = parent[thisChar]
        #     if len(node) == 0:
        #         if wildAvail:
        #             # a word itself is match in the trie, remove it
        #             parent.pop(thisChar)
        #             return
        #         else:
        #             # a neighboring word is found
        #             q.appendleft(matched + thisChar)
        #             return

        #     if wildAvail:
        #         for nextChar in node:
        #             if nextChar == matchingWord[nextCharIdx]: continue
        #             backtrack(nextChar, nextCharIdx + 1, matched + thisChar, node, False)
            
        #     nextChar = matchingWord[nextCharIdx]
        #     if nextChar in node:
        #         backtrack(nextChar, nextCharIdx + 1, matched + thisChar, node, True)
            
        #     if len(node) == 0:
        #         parent.pop(thisChar)

        for matchingWord in [beginWord] + wordList:
            adjacentMap[matchingWord] = []
            backtrack(0,"", trie, True)
        
        beginNode = word2Node[beginWord]
        q = deque([[beginNode]])
        allPath = []

        while len(q) > 0:
            path = q.pop()
            thisNode = path[-1]
            thisNode.used = True
            if len(allPath) == 0 or len(path) <= len(allPath[-1]):
                if thisNode.word == endWord:
                    allPath.append([n.word for n in path])
                else:
                    for nextNode in adjacentMap[thisNode.word]:
                        if nextNode.used: continue
                        q.appendleft(path + [nextNode])

        return allPath

# 10. Regular Expression Matching
class Q10:
    def isMatch(self, s: str, p: str) -> bool:
        memo ={}
        def dp(i: int, j: int) -> bool:
            if (i, j) not in memo:
                if j == len(p):
                    ans = (i == len(s))
                else:
                    matchFirst = i < len(s) and p[j] in [s[i], '.']

                    if j + 1 < len(p) and p[j+1] == '*':
                        ans = dp(i, j + 2) or matchFirst and dp(i + 1, j)
                    else:
                        ans = matchFirst and dp(i + 1, j + 1)
                
                memo[(i, j)] = ans
            return memo[(i, j)]
        return dp(0 , 0)

# 22. Generate Parentheses
class Q22:
    def generateParenthesis(self, n: int) -> List[str]:
        if n == 0:
            return []
        ret = []
        def backtrack(openCnt: int, paren: str, lCnt: int, rCnt: int):
            if lCnt == 0 and rCnt == 0:
                ret.append(paren)
                return
            if lCnt > 0:
                backtrack(openCnt + 1, paren + '(', lCnt - 1, rCnt)
            if rCnt >0 and openCnt > 0:
                backtrack(openCnt - 1, paren + ')', lCnt, rCnt - 1)
        backtrack(0, '', n, n)
        return  ret            

# 17. Letter Combinations of a Phone Number
class Q17:
    def letterCombinations(self, digits: str) -> List[str]:
        if len(digits) == 0:
            return []
        dig2let = {'2': ['a', 'b', 'c'], '3': ['d', 'e', 'f'], '4': ['g', 'h', 'i'], '5': ['j', 'k', 'l'], '6': ['m', 'n', 'o'], '7': ['p', 'q', 'r', 's'], '8': ['t', 'u', 'v'], '9': ['w', 'x', 'y', 'z']}
        ret = []
        def backtrack(i: int, s: str):
            if i == len(digits):
                ret.append(s)
            else:
                for l in dig2let[digits[i]]:
                    backtrack(i + 1, s+l)
        backtrack(0, '')
        return ret

# 136. Single Number
class Q136:
    def singleNumber(self, nums: List[int]) -> int:
        i = 0
        for x in nums:
            i ^= x
        return i

# 137. Single Number II
class Q137:
    def singleNumber(self, nums: List[int]) -> int:
        seenOnce = seenTwice = 0
        for x in nums:
            seenOnce = ~seenTwice&(x ^ seenOnce)
            seenTwice = ~seenOnce&(x ^ seenTwice)
        return seenOnce

# 268. Missing Number
class Q268:
    def missingNumber(self, nums: List[int]) -> int:
        miss = 0 ^ len(nums)
        for i, x in enumerate(nums):
            miss ^= i
            miss ^= x
        return miss

# 329. Longest Increasing Path in a Matrix
class Q329:
    def longestIncreasingPath(self, matrix: List[List[int]]) -> int:
        if len(matrix) == 0:
            return 0
        R = len(matrix)
        C = len(matrix[0])
        memo  = [[0]*len(matrix[0]) for _ in range(R)]
        def dfs(r, c):
            nonlocal memo, R, C
            maxL = 0
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                rr = r + dr
                cc = c + dc
                if rr >= 0 and rr < R and cc >= 0 and cc < C and matrix[rr][cc] > matrix[r][c]:
                    if memo[rr][cc] == 0:
                        dfs(rr, cc)
                    maxL = max(maxL, memo[rr][cc])
            memo[r][c] = maxL + 1
        maxL = 0
        for r in range(R):
            for c in range(C):
                if memo[r][c] == 0:
                    dfs(r, c)
                maxL = max(maxL, memo[r][c])
        return maxL

# 207. Course Schedule
class Solution:
    def __init__(self) -> None:
        self.WHITE = 0
        self.GREY = 1
        self.BLACK = 2
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        adjacentMap = {i:[] for i in range(numCourses)}
        colorMap = {i: self.WHITE for i in range(numCourses)}
        for p in prerequisites:
            adjacentMap[p[0]].append(p[1])
        def dfs(c)->bool:
            nonlocal colorMap
            colorMap[c] = self.GREY
            for p in adjacentMap[c]:
                pColor = colorMap[p]
                if pColor == self.GREY: return False
                if pColor == self.BLACK: continue
                if not dfs(p): return False
            colorMap[c] = self.BLACK
            return True
        for c in range(numCourses):
            if colorMap[c] == self.WHITE:
                if not dfs(c): return False
        return True

# 210. Course Schedule II
class Q210:
    def __init__(self) -> None:
        self.WHITE = 0
        self.GREY = 1
        self.BLACK = 2
    def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        adjacentMap = {i:[] for i in range(numCourses)}
        colorMap = {i: self.WHITE for i in range(numCourses)}
        for p in prerequisites:
            adjacentMap[p[0]].append(p[1])
        order = []
        def dfs(c)->bool:
            nonlocal colorMap, order
            colorMap[c] = self.GREY
            for p in adjacentMap[c]:
                pColor = colorMap[p]
                if pColor == self.GREY: return False
                if pColor == self.BLACK: continue
                if not dfs(p): return False
            colorMap[c] = self.BLACK
            order.append(c)
            return True
        for c in range(numCourses):
            if colorMap[c] == self.WHITE:
                if not dfs(c): return []
        return order

# 85. Maximal Rectangle
class Q85:
    def maximalRectangle(self, matrix: List[List[str]]) -> int:
        if len(matrix) == 0:
            return 0
        R = len(matrix)
        C = len(matrix[0])
        leftMost1 = [0]*C
        rightMost1 = [C]*C
        height = [0]*C
        maxArea = 0
        for r in range(R):
            curLeft = 0
            curRight = C
            for c in range(C):
                if matrix[r][c] == '1':
                    # compute the height of 1
                    height[c] += 1
                    # compute left boundary of 1
                    leftMost1[c] = max(curLeft, leftMost1[c])
                else:
                    height[c] = 0
                    leftMost1[c] = 0
                    curLeft = c + 1
                cc = C - c - 1
                if matrix[r][cc] == '1':
                    rightMost1[cc] = min(curRight, rightMost1[cc])
                else:
                    rightMost1[cc] = C
                    curRight = cc
            for c in range(C):
                maxArea = max(maxArea, height[c]*(rightMost1[c]-leftMost1[c]))
        return maxArea

# 84. Largest Rectangle in Histogram
class Q84:
    def largestRectangleArea(self, heights: List[int]) -> int:
        s = [(-1, -1)]
        maxArea = 0
        for i, x in enumerate(heights):
            if x < s[-1][1]:
                iH, xH = s[-1]
                while s[-1][1] > x:
                    _, xH = s.pop()
                    maxArea = max(maxArea, xH*(iH-s[-1][0]))
            s.append((i, x))
        iH, xH = s[-1]
        while len(s) > 1:
            _, xH = s.pop()
            maxArea = max(maxArea, xH*(iH-s[-1][0]))
        return maxArea

# 121. Best Time to Buy and Sell Stock
class Q121:
    def maxProfit(self, prices: List[int]) -> int:
        profit = 0
        buyPrice = prices[0]
        for x in prices[1:]:
            if x < buyPrice:
                buyPrice = x
            else:
                profit = max(profit, x-buyPrice)
        return profit

# 139. Word Break
class Q139:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        trie = {}
        EOW = '$'
        for word in wordDict:
            node = trie
            for char in word:
                node = node.setdefault(char, {})
            node[EOW] = {}

        memo = {}
        def backtrack(s: str) -> bool:
            if s in memo:
                return memo[s]
            
            node = trie
            canBreak = False
            for i, char in enumerate(s):
                if char not in node:
                    canBreak = False
                    break
                else:
                    node = node[char]
                    if EOW in node and (i+1 == len(s) or backtrack(s[i+1:])):
                        canBreak = True
                        break
            memo[s] = canBreak
            return canBreak
        return backtrack(s)

# 140. Word Break II
class Q140:
    def wordBreak(self, s: str, wordDict: List[str]) -> List[str]:
        memo = {}
        wordSet = set(wordDict)
        def backtrack(subS: str):
            if not subS:
                return [[]]
            if subS not in memo:
                memo[subS] = []
                for endIdx in range(1, len(subS)+1):
                    word = subS[:endIdx]
                    if word in wordSet:
                        for wordList in backtrack(subS[endIdx:]):                           
                            memo[subS].append([word] + wordList)
            return memo[subS]
        backtrack(s)
        return [' '.join(wordList) for wordList in memo[s]]

# 322. Coin Change
class Q322:
    def coinChange(self, coins: List[int], amount: int) -> int:
        memo = {0:0}
        coins = sorted(coins)
        def dp(amount: int) -> int:
            if amount not in memo:
                minCoinNum = amount
                canChange = False
                rAmount = amount
                for c in coins:
                    if c > amount: break
                    for cNum in range(amount//c+1):
                        rAmount = amount % c
                        if rAmount == 0:
                            minCoinNum = min(minCoinNum, cNum)
                            canChange = True
                            continue
                        if rAmount < coins[0]: 
                            continue
                        rNum = dp(rAmount)
                        if rNum >= 0:
                            minCoinNum = min(minCoinNum, cNum + rNum)
                            canChange = True
                if canChange:
                    memo[amount] = minCoinNum
                else:
                    memo[amount] = -1
            return memo[amount]
        return dp(amount)
    
    def coinChangeV2(self, coins: List[int], amount: int) -> int:
        memo = {0:0}
        coins = sorted(coins)
        def dp(amount: int) -> int:
            if amount not in memo:
                minCoinNum = amount
                canChange = False
                for c in coins:
                    if c > amount: break
                    cNum = dp(amount - c)
                    if cNum >= 0:
                        minCoinNum = min(minCoinNum, 1 + cNum)
                        canChange = True

                if canChange:
                    memo[amount] = minCoinNum
                else:
                    memo[amount] = -1
            return memo[amount]
        return dp(amount)

# 91. Decode Ways
class Q91:
    def numDecodings(self, s: str) -> int:
        for i,x in enumerate(s):
            if x == '0' and (i == 0 or s[i-1] not in ['1', '2']):
                return 0
        memo = {'': 1}
        def dp(s) ->int:
            if s not in memo:
                if s[0] == '0':
                    return 0
                if len(s) >= 2 and int(s[:2]) <= 26:
                    memo[s] = dp(s[1:]) + dp(s[2:])
                else:
                    memo[s] = dp(s[1:])
            return memo[s]
        return dp(s)
    
    def numDecodingsV2(self, s: str) -> int:
        ret = 1
        for i, x in list(enumerate(s))[::-1]:
            if x == '0':
                if i == 0 or (s[i-1] not in ['1', '2']):
                    return 0
            elif i > 0 and int(s[i-1:i+1]) < 26:
                ret += 1
        return ret

# 127. Word Ladder
class Q127:
    def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
        from collections import deque
        trie = {}
        for word in [beginWord] + wordList:
            node = trie
            for char in word:
                node = node.setdefault(char, {})
        
        nextWord = {}
        def backtrack(matchedWord: str, parentNode: dict, wildCard: bool):
            nonlocal word, q
            thisChar = matchedWord[-1]
            if len(matchedWord) == len(word):
                if not wildCard:
                    # a neighbor word is matched
                    nextWord[word].append(matchedWord)
                    q.appendleft(matchedWord)
                parentNode.pop(thisChar)
                return -1
            else:
                thisNode = parentNode[thisChar]
                for nextChar in list(thisNode.keys()):
                    if word[len(matchedWord)] == nextChar:
                        backtrack(matchedWord+nextChar, thisNode, wildCard)
                    elif wildCard:
                        backtrack(matchedWord+nextChar, thisNode, False)
                if len(thisNode) == 0:
                    parentNode.pop(thisChar)
                return -1

        q = deque([beginWord])
        pathLen = 0
        while len(q) > 0:
            pathLen += 1
            for _ in range(len(q)):
                word = q.pop()
                if word == endWord:
                    return pathLen
                nextWord[word] = []
                for char in list(trie.keys()):
                    if word[0] == char:
                        backtrack(char, trie, True)
                    else:
                        backtrack(char, trie, False)
        return 0

# 763. Partition Labels
class Q763:
    def partitionLabels(self, S: str) -> List[int]:
        charOrder = {}
        firstLastIdx = []
        for i,x in enumerate(S):
            if x in charOrder:
                # set last appearance position of x
                firstLastIdx[charOrder[x]][1] = i
            else:
                charOrder[x] = len(firstLastIdx)
                firstLastIdx.append([i, i])
        ret = []
        overLapInterval = [0, 0]
        for interval in firstLastIdx:
            if interval[0] <= overLapInterval[1]:
                overLapInterval[1] = max(interval[1], overLapInterval[1])
            else:
                ret.append(overLapInterval[1] - overLapInterval[0] + 1)
                overLapInterval = interval
        ret.append(overLapInterval[1] - overLapInterval[0] + 1)
        return ret

# 56. Merge Intervals
class Q56:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        if len(intervals) == 0:
            return []
        intervals.sort()
        ret = []
        overLapInterval = intervals[0]
        for interval in intervals[1:]:
            if interval[0] <= overLapInterval[1]:
                overLapInterval[1] = max(interval[1], overLapInterval[1])
            else:
                ret.append(overLapInterval)
                overLapInterval = interval
        ret.append(overLapInterval)
        return ret

# 588. Design In-Memory File System
class FileSystem:

    def __init__(self):
        self.file = {'':{}}

    def parse_path(self, path):
        path = path.split('/')
        if len(path[-1]) == 0:
            path = path[:-1]
        return path

    def ls(self, path: str) -> List[str]:
        path = self.parse_path(path)
        ret = []
        node = self.file
        for p in path:
            node = node[p]
        if type(node) == str:
            ret.append(p)
        else:
            for f in node:
                ret.append(f)
        ret.sort()
        return ret

    def mkdir(self, path: str) -> None:
        path = self.parse_path(path)
        node = self.file
        for p in path:
            node = node.setdefault(p, {})
        

    def addContentToFile(self, filePath: str, content: str) -> None:
        path = self.parse_path(filePath)
        if len(path[-1]) == 0:
            path = path[:-1]
        node = self.file
        for p in path[:-1]:
            node = node.setdefault(p, {})
        if path[-1] in node:
            node[path[-1]] += content
        else:
            node[path[-1]] = content

    def readContentFromFile(self, filePath: str) -> str:
        path = self.parse_path(filePath)
        node = self.file
        for p in path:
            node = node[p]
        return node

# 224. Basic Calculator
class Q244:
    def eval(self, l: list) -> int:
        ret = 0
        neg = False
        ans = l[0]
        i = 1
        while i < len(l) - 1:
            if l[i] in ('+', '-'):
                if neg:
                    ret -= ans
                else:
                    ret += ans
                if l[i] == '-':
                    neg = True
                else:
                    neg = False
                ans = l[i+1]
            else:
                if l[i] == '*':
                    ans = ans * l[i+1]
                else:
                    ans = ans // l[i+1]
            i += 2
        if neg:
            ret -= ans
        else:
            ret += ans
        return ret


    def preporcess(self, s: str) -> List:
        s = s.replace(' ', '')
        ret = []
        numStr = ''
        for x in s:
            if x in ['+', '-', '*', '/', '(', ')']:
                if len(numStr) > 0:
                    ret.append(int(numStr))
                    numStr = ''
                ret.append(x)
            else:
                numStr += x
        if len(numStr) > 0:
            ret.append(int(numStr))
        return ret

    def calculate(self, s: str) -> int:
        s = self.preporcess(s)

        parenCnt = 0
        segS = [[]]
        for i, x in enumerate(s):
            if x == '(':
                parenCnt += 1
                segS.append([])
                continue
            if x == ')':
                segEval = self.eval(segS.pop())
                segS[-1].append(segEval)
                continue
            segS[-1].append(x)
        return self.eval(segS[0])

# 772. Basic Calculator III
Q772 = Q244

# 227. Basic Calculator II
class Q227:
    def eval(self, l: list) -> int:
        ret = 0
        neg = False
        ans = l[0]
        i = 1
        while i < len(l) - 1:
            if l[i] in ('+', '-'):
                if neg:
                    ret -= ans
                else:
                    ret += ans
                if l[i] == '-':
                    neg = True
                else:
                    neg = False
                ans = l[i+1]
            else:
                if l[i] == '*':
                    ans = ans * l[i+1]
                else:
                    ans = ans // l[i+1]
            i += 2
        if neg:
            ret -= ans
        else:
            ret += ans
        return ret


    def preporcess(self, s: str) -> List:
        s = s.replace(' ', '')
        ret = []
        numStr = ''
        for x in s:
            if x in ['+', '-', '*', '/']:
                if len(numStr) > 0:
                    ret.append(int(numStr))
                    numStr = ''
                ret.append(x)
            else:
                numStr += x
        if len(numStr) > 0:
            ret.append(int(numStr))
        return ret

    def calculate(self, s: str) -> int:
        return self.eval(self.preporcess(s))

# 238. Product of Array Except Self
class Q238:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        productL = [1]*len(nums)
        productL[1] = nums[0]
        for i in range(2, len(nums)):
            productL[i] = nums[i-1]*productL[i-1]
        productR = 1
        for i in range(len(nums)-1, -1, -1):
            productL[i] = productR*productL[i]
            productR = productR*nums[i]
        return productL

# 348. Design Tic-Tac-Toe
class TicTacToe:
    def __init__(self, n: int):
        """
        Initialize your data structure here.
        """
        self.rowCnt = {1:{i:0 for i in range(n)}, 2:{i:0 for i in range(n)}}
        self.colCnt = {1:{i:0 for i in range(n)}, 2:{i:0 for i in range(n)}}
        self.diaCnt = {1:0, 2:0}
        self.invDiaCnt = {1:0, 2:0}
        self.n = n

    def move(self, row: int, col: int, player: int) -> int:
        """
        Player {player} makes a move at ({row}, {col}).
        @param row The row of the board.
        @param col The column of the board.
        @param player The player, can be either 1 or 2.
        @return The current winning condition, can be either:
                0: No one wins.
                1: Player 1 wins.
                2: Player 2 wins.
        """
        self.rowCnt[player][row] += 1
        self.colCnt[player][col] += 1
        if row == col:
            self.diaCnt[player] += 1
        if row + col == self.n - 1:
            self.invDiaCnt[player] += 1

        if (self.rowCnt[player][row] == self.n) or (self.colCnt[player][col] == self.n) or (self.diaCnt[player] == self.n) or (self.invDiaCnt[player] == self.n):
            return player
        else:
            return 0

# 301. Remove Invalid Parentheses
class Q301:
    # def removeInvalidParentheses(self, s: str) -> List[str]:
    #     pCnt = 0
    #     sSegIdx = [[0, 0, pCnt]]
    #     rParenCnt = [0]*len(s)
    #     for i in range(0, len(s)):
    #         if s[i] == '(':
    #             if pCnt > 0:
    #                 # start a new validation segment
                    
    #                 sSegIdx[-1][1] = i
    #                 sSegIdx.append([i, i, -1])
    #                 pCnt = -1
    #                 rParenCnt = [i] = 0


    #             else:
    #                 pCnt -= 1
    #         elif s[i] == ')':
    #             pCnt += 1
    #     sSegIdx[-1][1] = len(s)
    #     sSegIdx[-1][2] = pCnt

    #     def backtrack(sIdx: int, eIdx:int, rmPCnt:int, fixed: set):
    #         if rmPCnt == (eIdx - sIdx):
    #             fixed.add()

    #     fixedSeg = []
    #     for si in sSegIdx[:-1]:
        
    def removeInvalidParentheses(self, s: str) -> List[str]:
        ret = set()
        minRm = len(s)
        def backtrack(prevS, i, pCnt, rmCnt):
            nonlocal s, minRm
            if i == len(s):
                if pCnt == 0:
                    if rmCnt <= minRm:
                        ret.add(prevS)
                        minRm = rmCnt
            else:
                if s[i] == '(':
                    if rmCnt < minRm:
                        backtrack(prevS, i+1, pCnt, rmCnt + 1) # remove
                    else:
                        # pruning, expression is impossible to be valid if no more remove can be done
                        if pCnt - 1 + (len(s)-i) < 0:
                            return
                    backtrack(prevS+'(', i+1, pCnt-1, rmCnt) # not remove
                elif s[i] == ')':
                    if rmCnt < minRm:
                        backtrack(prevS, i+1, pCnt, rmCnt + 1) # remove
                    else:
                        if pCnt + 1 > 0:
                            return
                    if pCnt < 0:
                        # we can keep ) only if we have unmatched (
                        backtrack(prevS+')', i+1, pCnt+1, rmCnt)
                else:
                    # keep other characters
                    backtrack(prevS+s[i], i+1, pCnt, rmCnt)
        backtrack('', 0, 0, 0)
        return [x for x in ret if len(x) == len(s) - minRm]

# 31. Next Permutation
class Q31:
    def nextPermutation(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        for i in range(len(nums) - 2, -1, -1):
            if nums[i] < nums[i+1]:
                for j in range(len(nums) - 1, i, -1):
                    if nums[i] < nums[j]:
                        nums[i], nums[j] = nums[j], nums[i]
                        break
                nums[i+1:] = sorted(nums[i+1:])
                return
        nums[:] = nums[-1::-1]

# 560. Subarray Sum Equals K
class Q560:
    def subarraySum(self, nums: List[int], k: int) -> int:
        cnt = 0
        sumDict = {0:1}
        csum = 0
        for x in nums:
            csum += x
            cnt += sumDict.get(csum - k, 0)
            sumDict[csum] = sumDict.get(csum, 0) + 1
        return cnt

# 215. Kth Largest Element in an Array
class Q215:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        from heapq import heappush, heappop, heappushpop, heapify
        kLargest = nums[:k]
        heapify(kLargest)
        for x in nums[k:]:
            if x > kLargest[0]:
                heappushpop(kLargest, x)
        return kLargest[0]

# 528. Random Pick with Weight
class Q528:
    from random import randint
    from bisect import bisect_left
    def __init__(self, w: List[int]):
         
        self.culmW = []
        s = 0
        for x in w:
            s += x
            self.culmW.append(s)

    def pickIndex(self) -> int:
        w = randint(1, self.culmW[-1])
        i = bisect_left(self.culmW, w)
        return i

# 986. Interval List Intersections
class Q986:
    def intervalIntersection(self, A: List[List[int]], B: List[List[int]]) -> List[List[int]]:
        ret = []
        aIdx = 0
        bIdx = 0
        while aIdx < len(A) and bIdx < len(B):
            if A[aIdx][0] <= B[bIdx][0] <= A[aIdx][1]:
                ret.append([B[bIdx][0], min(A[aIdx][1], B[bIdx][1])])
            elif B[bIdx][0] <= A[aIdx][0] <= B[bIdx][1]:
                ret.append([A[aIdx][0], min(B[bIdx][1], A[aIdx][1])])
            
            if A[aIdx][1] < B[bIdx][1]:
                aIdx += 1
            elif A[aIdx][1] > B[bIdx][1]:
                bIdx += 1
            else:
                aIdx += 1
                bIdx += 1
        return ret

# 953. Verifying an Alien Dictionary
class Q953:
    def isAlienSorted(self, words: List[str], order: str) -> bool:
        order = {x:i for i, x in enumerate(order)}
        for w1, w2 in zip(words[:-1], words[1:]):
            for i, (char1, char2) in enumerate(zip(w1, w2)):
                if char1 != char2:
                    if order[char1] > order[char2]:
                        return False
                    else:
                        break
                else:
                    if i == len(w2) - 1 < len(w1) -1:
                        return False
        return True

# 173. Binary Search Tree Iterator
class BSTIterator:
    def __init__(self, root: TreeNode):
        self.path = []
        self.finished = []
        self.pNext = root
        while self.pNext and self.pNext.left:
            self.path.append(self.pNext)
            self.finished.append(False)
            self.pNext = self.pNext.left

    def go_next(self):
        if self.pNext.right:
            self.path.append(self.pNext)
            self.finished.append(True)
            self.pNext = self.pNext.right
            while self.pNext and self.pNext.left:
                self.path.append(self.pNext)
                self.finished.append(False)
                self.pNext = self.pNext.left
        else:
            while len(self.finished) > 0 and self.finished[-1]:
                self.path.pop()
                self.finished.pop()
            if len(self.path) > 0:
                self.pNext = self.path.pop()
                self.finished.pop()
            else:
                self.pNext = None

    def next(self) -> int:
        """
        @return the next smallest number
        """
        ret = self.pNext.val
        self.go_next()
        return ret


    def hasNext(self) -> bool:
        """
        @return whether we have a next smallest number
        """
        return self.pNext is not None

# 438. Find All Anagrams in a String
class Q438:
    def findAnagrams(self, s: str, p: str) -> List[int]:
        ret = []
        pCharCnt = {}
        for pi in p:
            pCharCnt[pi] = pCharCnt.get(pi, 0) + 1
        
        winCharCnt = {pi: 0 for pi in p}
        cnt = 0
        idx = 0
        for i, si in enumerate(s):
            if si not in pCharCnt:
                cnt = 0
                idx = i+1
                winCharCnt = {pi: 0 for pi in p}
                continue

            winCharCnt[si] = winCharCnt[si] + 1

            while winCharCnt[si] > pCharCnt[si]:
                winCharCnt[s[idx]] -= 1
                cnt -= 1
                idx += 1

            cnt += 1
            if cnt == len(p):
                ret.append(idx)
                winCharCnt[s[idx]] -= 1
                cnt -= 1
                idx += 1
        return ret

# 636. Exclusive Time of Functions
class Q636:
    def exclusiveTime(self, n: int, logs: List[str]) -> List[int]:
        ret = [0]*n
        funcStack = []
        startTime = []
        waitTime = []
        for l in logs:
            l = l.split(':')
            fid = int(l[0])
            t = int(l[2])
            if l[1] == 'start':
                funcStack.append(fid)
                startTime.append(t)
                waitTime.append(0)
            else:
                tSpan = t - startTime.pop() + 1
                assert fid == funcStack.pop()
                ret[fid] += (tSpan - waitTime.pop())
                waitTime[-1] += tSpan
        return ret

# 1249. Minimum Remove to Make Valid Parentheses
class Q1249:
    def minRemoveToMakeValid(self, s: str) -> str:
        ret = ''
        openCnt = 0
        for si in s:
            if si == '(':
                openCnt += 1
                ret += '('
            elif si == ')':
                if openCnt > 0:
                    ret += ')'
                    openCnt -= 1
            else:
                ret += si
        if openCnt > 0:
            ret = ret[::-1].replace('(', '', openCnt)[::-1]
        return ret

# 34. Find First and Last Position of Element in Sorted Array
class Q34:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        minIdx = 0
        maxIdx = len(nums) - 1
        lIdx = -1
        while maxIdx > minIdx:
            midIdx = int((maxIdx+minIdx+1)/2)
            if (nums[midIdx] > target) or (midIdx > 0 and nums[midIdx - 1] == target):
                if maxIdx == midIdx:
                    if nums[minIdx] == target:
                        lIdx = minIdx
                    break
                else:
                    maxIdx = midIdx
            elif nums[midIdx] < target:
                if minIdx == midIdx:
                    if nums[maxIdx] == target:
                        lIdx = maxIdx
                    break
                else:
                    minIdx = midIdx
            else:
                lIdx = midIdx
                break

        if len(nums) == 1:
            if nums[0] == target:
                return [0, 0]

        if lIdx == -1:
            return [-1, -1]

        minIdx = lIdx
        maxIdx = len(nums) - 1
        rIdx = maxIdx
        while maxIdx > minIdx:
            midIdx = int((maxIdx+minIdx+1)/2)
            if (nums[midIdx] > target):
                if maxIdx == midIdx:
                    if nums[minIdx] == target:
                        rIdx = minIdx
                    break
                else:
                    maxIdx = midIdx
            elif (midIdx < len(nums) - 1) and (nums[midIdx+1] == target):
                if minIdx == midIdx:
                    if nums[maxIdx] == target:
                        rIdx = maxIdx
                    break
                else:
                    minIdx = midIdx
            else:
                rIdx = midIdx
                break
        return [lIdx, rIdx]

# 158. Read N Characters Given Read4 II - Call multiple times
# The read4 API is already defined for you.
inputFile = "123456789123456789123456789"
readIdx = 0
def read4(buf4: List[str]) -> int:
    global readIdx, inputFile
    if readIdx >= len(inputFile):
        return 0

    if readIdx + 4 <= len(inputFile):
        buf4[:] = inputFile[readIdx:readIdx+4]
        readIdx += 4
        return 4
    else:
        readCnt = len(inputFile)-readIdx
        buf4[:readCnt] = inputFile[readIdx:]
        readIdx += readCnt
        return readCnt


class Q158:
    def __init__(self) -> None:
        self.buff4 = ['0']*4
        self.preread()

    def preread(self):
        self.remainCnt = read4(self.buff4)
        self.buffIdx = 0

    def read(self, buf: List[str], n: int) -> int:
        buf[:] = ['']*len(buf)

        readCnt = 0
        while n > 0:
            if self.remainCnt <= n:
                buf[readCnt: readCnt+self.remainCnt] = self.buff4[self.buffIdx:self.buffIdx+self.remainCnt]
                readCnt += self.remainCnt
                n -= self.remainCnt

                self.preread()
                if self.remainCnt == 0:
                    break
            else:
                buf[readCnt: readCnt+n] = self.buff4[self.buffIdx:self.buffIdx+n]
                readCnt += n
                self.remainCnt -= n
                self.buffIdx += n
                n = 0
        return readCnt

# 415. Add Strings
class Q415:
    def addStrings(self, num1: str, num2: str) -> str:
        from itertools import zip_longest
        nums = [0, 0]
        digit2Str = {i:str(i) for i in range(10)}
        str2Digit = {str(i):i for i in range(10)}
        ret = ''
        carrry = False
        for d1, d2 in zip_longest(num1[::-1], num2[::-1]):
            if d1 and d2:
                d1 = str2Digit[d1]
                d2 = str2Digit[d2]
                s = d1 + d2
            else:
                if d1:
                    s = str2Digit[d1]
                else:
                    s = str2Digit[d2]
            if carrry:
                s += 1
                carrry = False
            if s >= 10:
                s -= 10
                carrry = True
            ret = digit2Str[s]+ret
        if carrry:
            ret = '1'+ret
        return ret

# 133. Clone Graph
# Definition for a Node.
class Node:
    def __init__(self, val = 0, neighbors = None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []


class Q133:
    def cloneGraph(self, node: 'Node') -> 'Node':
        if node is None:
            return None
        nodeDict = {}
        dstNode = nodeDict.setdefault(node.val, Node(node.val))
        nodes = set()
        def dfs(srcNode: 'Node', dstNode: 'Node'):
            nodes.add(dstNode.val)
            for srcChild in srcNode.neighbors:
                dstChild = nodeDict.setdefault(srcChild.val, Node(srcChild.val))
                dstNode.neighbors.append(dstChild)
                if dstChild.val not in nodes:
                    dfs(srcChild, dstChild)
        dfs(node, dstNode)
        return dstNode
        
# 50. Pow(x, n)
class Q50:
    def myPow(self, x: float, n: int) -> float:
        if n == 0:
            return 1.0
        power = 1
        ret = x
        powTable = {1:x}
        nextPow = 1
        while power < abs(n):
            ret *= powTable[nextPow]
            power += nextPow
            powTable[power] = ret
            nextPow *= 2
            while nextPow + power > abs(n):
                nextPow /= 2
        if n > 0:
            return ret
        else:
            return 1/ret

# 340. Longest Substring with At Most K Distinct Characters
class Q340:
    def lengthOfLongestSubstringKDistinct(self, s: str, k: int) -> int:
        if k == 0:
            return 0
        maxL = 0
        lIdx = 0
        charSet = set()
        for rIdx in range(0, len(s)):
            charSet.add(s[rIdx])
            if len(charSet) > k:
                while s[lIdx] == s[rIdx]:
                    lIdx += 1
                charSet.remove(s[lIdx])
                lIdx += 1
            else:
                maxL = max(maxL, len(charSet))
        return maxL

# 88. Merge Sorted Array
class Q88: 
    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        """
        Do not return anything, modify nums1 in-place instead.
        """
        if m == 0:
            p1 = -1
        else:
            p1 = 0
        if n == 0:
            p2 = -1
        else:
            p2 = 0
        cnt = 0
        while 0 <= p2 < n:
            if p1 >= 0 and cnt < m:
                if nums1[p1] >= nums2[p2]:
                    nums1.insert(p1, nums2[p2])
                    p2 += 1
                else:
                    cnt += 1
                p1 += 1
            else:
                if p1 < 0:
                    nums1[:n] = nums2
                else:
                    nums1[p1: p1+(n-p2)] = nums2[p2:n]
                break
        del nums1[m+n:]

# 311. Sparse Matrix Multiplication
class Q311:
    def multiply(self, A: List[List[int]], B: List[List[int]]) -> List[List[int]]:
        rowSkip = set()
        colSkip = set()
        for r, row in enumerate(A):
            if sum([abs(x) for x in row]) == 0:
                rowSkip.add(r)
        BTrans = []
        for c in range(len(B[0])):
            col = [row[c] for row in B]
            if sum([abs(x) for x in col]) == 0:
                colSkip.add(c)
            BTrans.append(col)

        AB = [[0]*len(B[0]) for _ in range(len(A))]
        for r in range(len(A)):
            if r in rowSkip:
                continue
            for c in range(len(B[0])):
                if c in colSkip:
                    continue
                AB[r][c] = sum([x*y for x, y in zip(A[r], BTrans[c])])
        return AB

# 721. Accounts Merge
class Q721:
    def accountsMerge(self, accounts: List[List[str]]) -> List[List[str]]:
        acntIdx = set()
        emailDict = {}
        for i, acnt in enumerate(accounts):
            newEmail = [email for email in acnt[1:] if email not in emailDict]
            if len(newEmail) == len(acnt) - 1:
                for email in newEmail:
                    emailDict[email] = i
                acntIdx.add(i)
            else:
                oldEmail = [email for email in acnt[1:] if email in emailDict]
                oldAcntIdxAll = set([emailDict[email] for email in oldEmail])
                oldAcntIdx = oldAcntIdxAll.pop()
                while len(oldAcntIdxAll) > 0:
                    oldAcntIdxRm = oldAcntIdxAll.pop()
                    for email in accounts[oldAcntIdxRm][1:]:
                        accounts[oldAcntIdx].append(email)
                        emailDict[email] = oldAcntIdx
                    acntIdx.remove(oldAcntIdxRm)
                for email in newEmail:
                    emailDict[email] = oldAcntIdx
                    accounts[oldAcntIdx].append(email)
        ret = []
        for idx in acntIdx:
            acnt = accounts[idx]
            acnt[1:] = sorted(list(set(acnt[1:])))
            ret.append(acnt)
        return ret

# 523. Continuous Subarray Sum
class Q523:
    def checkSubarraySum(self, nums: List[int], k: int) -> bool:
        rem2Idx = {0: -1}
        cumSum = 0
        for i, x in enumerate(nums):
            cumSum += x
            if k == 0:
                rem = cumSum
            else:
                rem = cumSum % k
            if (rem in rem2Idx):
                if (i - rem2Idx[rem] > 1):
                    return True
            else:
                rem2Idx[rem] = i

# 278. First Bad Version
# The isBadVersion API is already defined for you.
# @param version, an integer
# @return an integer
# def isBadVersion(version):

class Solution:
    def firstBadVersion(self, n):
        """
        :type n: int
        :rtype: int
        """
        minIdx = 1
        maxIdx = n
        while minIdx < maxIdx:
            midIdx = int((minIdx + maxIdx + 1)/2)
            if not isBadVersion(midIdx):
                minIdx = midIdx
            else:
                if (midIdx == 1) or (not isBadVersion(midIdx-1)):
                    return midIdx
                if maxIdx == midIdx:
                    return minIdx
                maxIdx = midIdx
        return minIdx

# 1428. Leftmost Column with at Least a One
class Q1428:
    def leftMostColumnWithOne(self, binaryMatrix: 'BinaryMatrix') -> int:
        [M, N] = BinaryMatrix.dimensions()
        cIdxMin = N
        for r in range(0, M):
            for c in range(cIdxMin - 1, -1, -1):
                if binaryMatrix.get(r, c) == 0:
                    break
                else:
                    cIdxMin = c
        return cIdxMin

# 680. Valid Palindrome II
class Q680:
    def validPalindrome(self, s: str) -> bool:
        for i in range(len(s)//2):
            if s[i] != s[~i]:
                j = len(s)-i-1
                if not (s[i:j] == s[i:j][::-1] or s[i+1:j+1] == s[i+1:j+1][::-1]):
                    return False
        return True

# 67. Add Binary
class Q67:
    def addBinary(self, a: str, b: str) -> str:
        from itertools import zip_longest
        carry = False
        s = ''
        for ai, bi in zip_longest(a[::-1], b[::-1], fillvalue = '0'):
            if ai == bi == '1':
                if carry:
                    s+='1'
                else:
                    s+='0'
                carry = True
            elif ai == bi == '0':
                if carry:
                    s+='1'
                else:
                    s+='0'
                carry = False
            else:
                if carry:
                    s+='0'
                else:
                    s+='1'
        if carry:
            s += '1'
        return s[::-1]

# 314. Binary Tree Vertical Order Traversal
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Q314:
    def verticalOrder(self, root: TreeNode) -> List[List[int]]:
        from collections import deque
        colDict = {}
        q = deque([(root, 0)])
        while len(q) > 0:
            (node, col) = q.pop()
            if col in colDict:
                colDict[col].append(node.val)
            else:
                colDict[col] = [node.val]
            if node.left:
                q.appendleft((node.left, col - 1))
            if node.right:
                q.appendleft((node.right, col + 1))

        colIdx = [k for k in colDict]
        lCol = min(colIdx)
        rCol = max(colIdx)
        return [colDict[i] for i in range(lCol, rCol+1)]

# 543. Diameter of Binary Tree
class Q543:
    def diameterOfBinaryTree(self, root: TreeNode) -> int:
        if root is None:
            return 0
        def dfs(node: TreeNode):
            lH = 0
            lMax = 0
            if node.left:
                (lH, lMax) = dfs(node.left)
            rH = 0
            rMax = 0
            if node.right:
                (rH,  rMax) = dfs(node.right)
            return (max(lH, rH) + 1, max(lMax, rMax, lH + rH + 1))
        return dfs(root) - 1

# 938. Range Sum of BST
class Q938:
    def rangeSumBST(self, root: TreeNode, L: int, R: int) -> int:
        # sum(node.val) for all node in root such that L <= node.val <= R
        if root is None:
            return 0

        def dfs(node: TreeNode):
            sL = 0
            sR = 0
            if node.val <= L:
                if node.right:
                    sR = dfs(node.right)
            elif node.val >= R:
                if node.left:
                    sL = dfs(node.left)
            else:
                if node.left:
                    sL = dfs(node.left)
                if node.right:
                    sR = dfs(node.right)
            if L <= node.val <= R:
                return sL + sR + node.val
            else:
                return sL + sR
        return dfs(root)

# 65. Valid Number
class Q65:
    def isNumber(self, s: str) -> bool:
        s = s.strip(' ')
        if len(s) == 0:
            return False
        digit = set([str(i) for i in range(10)])
        s = s.split('e')
        if len(s) > 2:
            return False
        sCheck = []

        s1 = s[0]
        if len(s1) > 0 and s1[0] in ['-', '+']:
            s1 = s1[1:]
        if len(s1) == 0:
            return False
        dotSplit = s1.split('.')
        if len(dotSplit) == 1:
            sCheck.append(dotSplit[0])
        elif len(dotSplit) == 2:
            if len(dotSplit[0]) + len(dotSplit[1]) == 0:
                return False
            sCheck.append(dotSplit[0])
            sCheck.append(dotSplit[1])
        else:
            return False

        if len(s) == 2:
            s2 = s[1]
            if len(s2) > 0 and s2[0] in ['-', '+']:
                s2 = s2[1:]
            if len(s2) == 0:
                return False
            sCheck.append(s2)

        for ss in sCheck:
            for x in ss:
                if x not in digit:
                    return False
        return True

# 29. Divide Two Integers
class Q29:
    def divide(self, dividend: int, divisor: int) -> int:
        positive = (dividend >= 0 and divisor > 0) or (dividend < 0 and divisor < 0)
        dividend = abs(dividend)
        multi = abs(divisor)
        if dividend < multi:
            return 0

        q = 1
        multiList = [(1, multi)]
        while multi + multi <= dividend:
            multi += multi
            q += q
            multiList.append((q, multi))

        while len(multiList) > 0:
            qs, ms = multiList.pop()
            while multi + ms <= dividend:
                q += qs
                multi += ms
        
        if positive:
            if q >= 2**31:
                return 2**31 - 1
            else:
                return q
        else:
            if q > 2**31:
                return 2**31 - 1
            else:
                return -q

# 282. Expression Add Operators
class Q282:
    def addOperators(self, num: str, target: int) -> List[str]:
        if len(num) == 0:
            return []
        ret = []
        def backtrack(idx: int, s: int, multStack: List[int], plusStack: bool, op: str, expr: str):
            thisChar = num[idx]
            thisNum = int(thisChar)

            if op is None:
                if multStack[-1] == 0: return # avoid number that > 0 starting with 0
                expr += thisChar
                multStack[-1] = multStack[-1]*10 + thisNum
            else:
                expr += (op + thisChar)
                if op == '*':
                    multStack.append(thisNum)
                else:
                    if len(multStack) > 0:
                        m = 1
                        for n in multStack:
                            m *= n
                        if plusStack:
                            s += m
                        else:
                            s -= m
                    multStack = [thisNum]
                    if op == '+':
                        plusStack = True
                    else:
                        plusStack = False

            if idx == len(num) - 1:
                m = 1
                for n in multStack:
                    m *= n
                if plusStack:
                    s += m
                else:
                    s -= m
                if s == target:
                    ret.append(expr[1:])
            else:
                backtrack(idx+1, s, multStack.copy(), plusStack, None, expr)
                backtrack(idx+1, s, multStack.copy(), plusStack, '+', expr)
                backtrack(idx+1, s, multStack.copy(), plusStack, '-', expr)
                backtrack(idx+1, s, multStack.copy(), plusStack, '*', expr)
        backtrack(0, 0, [], True, '+', '')
        return ret

# 249. Group Shifted Strings
class Q249:
    def groupStrings(self, strings: List[str]) -> List[List[str]]:
        shiftDict = {}
        for s in strings:
            ascii = [ord(x) for x in s]
            key = tuple(a - ascii[0] if a >= ascii[0] else a - ascii[0] + 26 for a in ascii)
            if key in shiftDict:
                shiftDict[key].append(s)
            else:
                shiftDict[key] = [s]
        return [shiftDict[k] for k in shiftDict]

# 785. Is Graph Bipartite?
class Q785:
    def isBipartite(self, graph: List[List[int]]) -> bool:
        nodes = set()
        from collections import deque
        groupMap = [0]*len(graph)
        for i in range(len(graph)):
            # if len(graph[i]) == 0: return False
            if i in nodes: continue
            q = deque([i])
            groupMap[i] = 1
            while len(q) > 0:
                node = q.pop()
                if node in nodes:
                    continue
                else:
                    nodes.add(node)
                for neighbor in graph[node]:
                    if groupMap[neighbor] > 0 and groupMap[neighbor] == groupMap[node]:
                        return False
                    else:
                        groupMap[neighbor] = 3 - groupMap[node]
                    q.appendleft(neighbor)
        return True

# 670. Maximum Swap
class Q670:
    def maximumSwap(self, num: int) -> int:
        # 1. find the largest digit maxDigit and its index maxIdx, if multiple largest digits exit, use the right most one.
        # 2. from let to right, find a digit x and its index xIdx such that:
        #   a. x < maxDigit
        #   b. xIdx < maxIdx
        # 3. if not such (xIdx, x) is found, use the next largest digit and its index, go to 2
        num = [n for n in str(num)]
        maxIdx = {}
        for i in range(len(num) - 1, -1, -1):
            maxIdx.setdefault(num[i], i)
        for d in range(9, -1, -1):
            if d not in maxIdx: continue
            for i in range(0, len(num)):
                if num[i] < d and i < maxIdx[d]:
                    num[i], num[maxIdx[d]] = num[maxIdx[d]], num[i]
                    return int(''.join(num))

# 1060. Missing Element in Sorted Array
class Q1060:
    def missingElement(self, nums: List[int], k: int) -> int:
        if k <= 0: k = 1
        p = 0
        misNum = nums[0]
        while k > 0 and p < len(nums):
            if nums[p] <= misNum:
                p += 1
            else:
                misNum += 1
                if misNum != nums[p]:
                    k -= 1
        if k == 0:
            return misNum
        else:
            return misNum + k

# 1026. Maximum Difference Between Node and Ancestor
class Q1026:
    def maxAncestorDiff(self, root: TreeNode) -> int:
        if root is None:
            return 0
        ret = 0
        def backtrack(minVal: int, maxVal: int, node: TreeNode):
            nonlocal ret
            minVal = min(node.val, minVal)
            maxVal = max(node.val, maxVal)
            ret = max(ret, maxVal - minVal)
            if node.left:
                backtrack(minVal, maxVal, node.left)
            if node.right:
                backtrack(minVal, maxVal, node.right)
        backtrack(root.val, root.val, root)
        return ret

# 689. Maximum Sum of 3 Non-Overlapping Subarrays
class Q689:
    def maxSumOfThreeSubarrays(self, nums: List[int], k: int) -> List[int]:
        s = sum(nums[:k])
        l = len(nums)
        subArrSum = [(0, s)]
        for i in range(1, l-k+1):
            s -= nums[i-1]
            s += nums[i+k-1]
            subArrSum.append((i, s))
        sl = len(subArrSum)
        #print(sl)
        lMax = [subArrSum[0]]
        for x in subArrSum[1:sl-2*k]:
            if x[1] > lMax[-1][1]:
                lMax.append(x)
            else:
                lMax.append(lMax[-1])
        rMax = [subArrSum[-1]]
        for x in subArrSum[-2:-(sl-2*k)-1:-1]:
            if x[1] >= rMax[-1][1]:
                rMax.append(x)
            else:
                rMax.append(rMax[-1])
        #print(len(lMax), len(subArrSum[k: l-2*k+1]), len(rMax))
        maxS = lMax[0][1] + subArrSum[k][1] + rMax[-1][1]
        ret = [lMax[0][0], subArrSum[k][0], rMax[-1][0]]
        for l, m, r in zip(lMax, subArrSum[k: l-2*k+1], rMax[::-1]):
            if l[1] + m[1] + r[1] > maxS:
                maxS = l[1] + m[1] + r[1]
                ret = [l[0], m[0], r[0]]
        return ret

# 339. Nested List Weight Sum
class Q339:
    def depthSum(self, nestedList: List['NestedInteger']) -> int:
        def list_sum(nestedList: List['NestedInteger'], depth: int):
            s = 0
            for n in nestedList:
                if type(n) == int:
                    s += depth*n
                else:
                    s += list_sum(n, depth + 1)
            return s
        return list_sum(nestedList, 1)

# 621. Task Scheduler
class Q621:
    def leastInterval(self, tasks: List[str], n: int) -> int:
        freq = [0]*26
        for x in tasks:
            freq[ord(x)-ord('A')] += 1
        freq.sort()
        fMax = freq.pop()
        idle = (fMax - 1)*n
        for x in freq:
            idle -= min(x, fMax - 1)
        idle = max(0, idle)
        return len(tasks) + idle

# 398. Random Pick Index
class Solution:
    import random
    def __init__(self, nums: List[int]):
        idxDict = {}
        for i, x in enumerate(nums):
            if x in idxDict:
                idxDict[x].append(i)
            else:
                idxDict[x] = [i]

    def pick(self, target: int) -> int:
        idces = self.idxDict[x]
        return idces[random.randint(0, len(idces) - 1)]

# 896. Monotonic Array
class Q896:
    def isMonotonic(self, A: List[int]) -> bool:
        ret = True
        if len(A) <= 2:
            return True
        increase = (A[-1] - A[0] > 0)
        for x, y in zip(A, A[1:]):
            if x == y: continue
            if (y - x > 0) != increase:
                return False
        return True

# 394. Decode String
class Q394:
    def decodeString(self, s: str) -> str:
        repStr = ['']
        repNum = [1]
        numSet = set([str(i) for i in range(10)])
        numStr = ''
        for x in s:
            if x in numSet:
                numStr += x
            elif x == '[':
                repStr.append('')
                repNum.append(int(numStr))
                numStr = ''
            elif x == ']':
                decode = repStr.pop()*repNum.pop()
                repStr[-1] += decode
            else:
                repStr[-1] += x
        return repStr[-1]

# 767. Reorganize String
class Q767:
    def reorganizeString(self, S: str) -> str:
        freq = [(0, chr(ord('a') + i)) for i in range(26)]
        for x in S:
            freq[ord(x) - ord('a')][0] += 1
        freq.sort()
        freqMax = freq.pop()
        slotCnt = (freqMax[0] - 1)
        slot = [[freqMax[1]] for _ in range(freqMax[0])]
        slotItr = 0
        for x in freq:
            # slotCnt -= max(freqMax - 1, x[0])
            slotCnt -= x[0]
            while x[0] > 0:
                slot[slotItr].append(x[1])
                slotItr = ((slotItr + 1) % freqMax[0])
                x[0] -= 1
        if slotCnt > 0:
            return ''
        else:
            return ''.join([''.join(x) for x in slot])

# 843. Guess the Word
# """
# This is Master's API interface.
# You should not implement it, or speculate about its implementation
# """
# class Master:
#     def guess(self, word: str) -> int:
class Q843:
    def findSecretWord(self, wordlist: List[str], master: 'Master') -> None:
        matchDict = {word:[set() for _ in range(5)] for word in wordlist}
        for i in range(0, len(wordlist) - 1):
            for j in range(i+1, len(wordlist)):
                wi = wordlist[i]
                wj = wordlist[j]
                matchCnt = sum([1 if chari == charj else 0 for chari, charj in zip(wi, wj)])
                matchDict[wi][matchCnt].add(wj)
                matchDict[wj][matchCnt].add(wi)

        guessSet = set(wordlist)
        matchCnt = 0
        while matchCnt != 6:
            guess = guessSet.pop()
            matchCnt = master.guess(guess)
            if matchCnt == 6: return
            guessSet = guessSet & guessDict[guess][guessCnt]

# 72. Edit Distance
class Q72:
    def minDistance(self, word1: str, word2: str) -> int:
        l1 = len(word1)
        l2 = len(word2)
        if not l1 or not l2:
            return l1 + l2
    
        dist = [[0]*(l2+1) for _ in range(l1+1)]
        for i in range(l1 + 1):
            dist[i][0] = i
        for j in range(l2 + 1):
            dist[0][j] = j
        
        for i in range(1, l1 + 1):
            for j in range(1, l2 + 1):
                if word1[i-1] == word2[j-1]:
                    dist[i][j] = min(dist[i-1][j-1], dist[i-1][j] + 1, dist[i][j-1] + 1)
                else:
                    dist[i][j] = min(dist[i-1][j-1], dist[i-1][j], dist[i][j-1]) + 1
        return dist[l1][l2]

# 1197. Minimum Knight Moves
class Q1197:
    def minKnightMoves(self, x: int, y: int) -> int:
        from collections import deque
        x = abs(x)
        y = abs(y)
        if x < y:
            x,y = y,x
        DAll = [(2, 1, 1), (2, -1, 1), (-2, 1, 1), (-2, -1, 1), (1, 2, 1), (1, -2, 1), (-1, 2, 1), (-1, -2, 1)]
        D = [(-2, -1, 1), (-1, -2, 1)]
        path = deque([(x, y, 0)])
        passed = set([(x, y, 0)])
        while path:
            p = path.pop()
            if p[:2] == (0, 0): return p[2]
            if p[0] >= 2 and p[1] >= 2:
                direct = D
            else:
                direct = DAll
            for d in direct:
                (xi, yi, mi) = tuple(abs(i+j) for i,j in zip(p, d))
                if xi < yi:
                    xi,yi = yi,xi
                nextP = (xi, yi, mi)
                if nextP[:2] in passed:
                    continue
                else:
                    passed.add(nextP[:2])
                    path.appendleft(nextP)

# 1048. Longest String Chain
class Q1048:
    def longestStrChain(self, words: List[str]) -> int:
        words.sort(key = lambda x:len(x))
        chainLen = {w:1 for w in words}
        ret = 1
        for i, word1 in enumerate(words):
            for j in range(i+1, len(words)):
                word2 = words[j]
                if len(word2) == len(word1):
                    continue
                if len(word2) > len(word1) + 1:
                    break
                insert = False
                match = True
                idx1 = 0
                idx2 = 0
                while idx1 < len(word1):
                    if word1[idx1] != word2[idx2]:
                        if insert:
                            match = False
                            break
                        else:
                            insert = True
                            idx2 += 1
                            continue
                    idx1 += 1
                    idx2 += 1
                if match:
                    l = max(chainLen[word2], chainLen[word1] + 1)
                    ret = max(ret, l)
                    chainLen[word2] = l
        return ret

# 1438. Longest Continuous Subarray With Absolute Diff Less Than or Equal to Limit
class Q1438:
    def longestSubarray(self, nums: List[int], limit: int) -> int:
        ret = 1
        startIdx = 0
        maxIdx = 0
        minIdx = 0
        for endIdx in range(1, len(nums)):
            if nums[endIdx] > nums[maxIdx]:
                maxIdx = endIdx
                if nums[maxIdx] - nums[minIdx] > limit:
                    ret = max(ret, endIdx - startIdx)
                    while nums[maxIdx] - nums[minIdx] > limit:
                        minIdx += 1
                        startIdx = minIdx
                        for i in range(minIdx, endIdx):
                            if nums[i] <= nums[minIdx]:
                                minIdx = i

                    
                    
            if nums[endIdx] < nums[minIdx]:
                minIdx = endIdx
                if nums[maxIdx] - nums[minIdx] > limit:
                    ret = max(ret, endIdx - startIdx)
                    while nums[maxIdx] - nums[minIdx] > limit:
                        maxIdx += 1
                        startIdx = maxIdx
                        for i in range(maxIdx, endIdx):
                            if nums[i] >= nums[maxIdx]:
                                maxIdx = i

        ret = max(ret, endIdx - startIdx + 1)
        return ret

# 642. Design Search Autocomplete System
class AutocompleteSystem:
    import heapq
    def __init__(self, sentences: List[str], times: List[int]):
        # {char1: ({char2: ({...}, [[...], [...]])}, [sentence1 index, sentence2 index],...])}
        self.sentences = sentences
        self.times = times
        self.trie = {}
        self.idxDict = {}
        for s in sentences:
            self.idxDict[s] = len(self.idxDict)
            self._trie_add_sentence(s)
        self.inputHolder = []
        self.nextNode = self.trie

    def input(self, c: str) -> List[str]:
        if c == '#':
            self._end_sentence()
            return []
        self.inputHolder.append(c)
        (self.nextNode, sIdx) = self.nextNode.get(c, [{}, []])
        topIdx = heapq.nsmallest(3, sIdx, key = lambda i: (-self.times[i], self.sentences[i]))
        topS = [self.sentences[i] for i in topIdx]
        return topS
    
    def _trie_add_sentence(self, s):
        node = self.trie
        for char in s:
            (node, sIdx) = node.setdefault(char, [{}, []])
            sIdx.append(self.idxDict[s])

    def _end_sentence(self):
        self.nextNode = self.trie
        if not self.inputHolder: return
        s = ''.join(self.inputHolder)
        self.inputHolder = []
        if s not in self.idxDict:
            self.sentences.append(s)
            self.times.append(1)
            self.idxDict[s] = len(self.idxDict)
            self._trie_add_sentence(s)
        else:
            self.times[self.idxDict[s]] += 1

# 57. Insert Interval
class Q57:
    def insert(self, intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
        if len(intervals) == 0:
            return [newInterval]
        insP = 0
        for i in range(len(intervals)):
            if newInterval[0] > intervals[i][1]: 
                continue
            elif newInterval[1] < intervals[i][0]:
                intervals.insert(i, newInterval)
            else:
                intervals[i] = [min(newInterval[0], intervals[i][0]), max(newInterval[1], intervals[i][1])]
            insP = i
            break
        if newInterval[0] > intervals[-1][1]:
            insP = len(intervals)
            intervals.append(newInterval)
        mergIdx = insP
        mergCnt = 0
        for i in range(insP + 1, len(intervals)):
            if intervals[i][0] <= intervals[insP][1]:
                mergCnt += 1
                intervals[insP][1] = max(intervals[insP][1], intervals[i][1])
            else:
                break
        for _ in range(mergCnt):
            del intervals[insP + 1]
        return intervals

# 722. Remove Comments
class Q722:
    def removeComments(self, source: List[str]) -> List[str]:
        ret = []
        ln = 0
        while ln < len(source):
            line = source[ln]
            for cn in range(len(line) -1):
                if line[cn: cn+2] == '//':
                    line = line[:cn]
                    break
                elif line[cn: cn+2] == '/*':
                    # check if the block comment ends in the same line
                    blockEnd = False
                    for cn2 in range(cn+2, len(line)-1):
                        if line[cn2: cn2+2] == '*/':
                            blockEnd = True
                            break
                    if blockEnd:
                        line = line[:cn] + line[cn2+2:]
                    else:
                        # keep skiping lines until */ is found
                        while not blockEnd and ln < len(source) - 1:
                            ln += 1
                            newLine = source[ln]
                            for cn2 in range(len(newLine) -1):
                                if newLine[cn2: cn2+2] == '*/':
                                    line = line[:cn] + newLine[cn2+2:]
                                    blockEnd = True
                                    break
                    # // may appear after the line where a block comment ends
                    source[ln] = line
                    ln -= 1
                    line = ""
                    break
                else:
                    continue
            if len(line) > 0:
                ret.append(line)
            ln += 1
        return ret

# 399. Evaluate Division
class Q399:
    def calcEquation(self, equations: List[List[str]], values: List[float], queries: List[List[str]]) -> List[float]:
        from collections import deque
        divGraph = {}
        for [x, y], v in zip (equations, values):
            divGraph.setdefault(x, {})[y] = v
            if v != 0:
                divGraph.setdefault(y, {})[x] = 1/v
        
        ret = []
        for [dividend, divisor] in queries:
            if dividend not in divGraph:
                ret.append(-1)
                continue
            quotient = -1
            q = deque([(dividend, 1)])
            path = set([dividend])
            while q:
                (x, v) = q.pop()
                if x == divisor:
                    quotient = v
                    break
                for y, vv in divGraph[x].items():
                    if y in path:
                        continue
                    path.add(y)
                    q.appendleft((y, v*vv))
            ret.append(quotient)
        
        return ret

# 1031. Maximum Sum of Two Non-Overlapping Subarrays
class Q1031:
    def maxSumTwoNoOverlap(self, A: List[int], L: int, M: int) -> int:
        Lsum = [(sum(A[:L]), 0)]
        Msum = [(sum(A[:M]), 0)]
        for i in range(1, len(A) - L + 1):
            Lsum.append((Lsum[-1][0] - A[i-1] + A[i+L-1], i))
        for i in range(1, len(A) - M + 1):
            Msum.append((Msum[-1][0] - A[i-1] + A[i+M-1], i))
        Lsum.sort(reverse=True)
        Msum.sort(reverse=True)
        ret = sum(A[:L+M])
        for li in range(len(Lsum)):
            breakFlag = True
            for mi in range(len(Msum)):
                if (Lsum[li][1] <= Msum[mi][1] and Lsum[li][1] + L > Msum[mi][1]) or (Msum[mi][1] <= Lsum[li][1] and Msum[mi][1] + M > Lsum[li][1]):
                    breakFlag = False
                    continue
                ret = max(ret, Lsum[li][0] + Msum[mi][0])
                break
            if breakFlag:
                break
        return ret

# 727. Minimum Window Subsequence
class Q727:
    def minWindow(self, S: str, T: str) -> str:
        # for any i in range(len(T)), T[:i] is a subsquence of S[curCandWinStart[endIdx]: endIdx + 1] if curCandWinStart[endIdx] is not None
        curCandWinStart = [i if s == T[0] else None for i, s in enumerate(S)]

        for t in T[1:]:
            nextCandWinStart = [None] * len(S)
            last = None
            for i, (s, startIdx) in enumerate(zip(S, curCandWinStart)):
                if last is not None and t == s:
                    nextCandWinStart[i] = last
                if startIdx is not None:
                    last = startIdx
            curCandWinStart = nextCandWinStart
        
        startIdx, endIdx = 0, len(S)
        for e, s in enumerate(curCandWinStart):
            if s is not None and e - s < endIdx - startIdx:
                startIdx = s
                endIdx = e
        if endIdx - startIdx < len(S):
            return S[startIdx: endIdx + 1]
        else:
            return ""

# 946. Validate Stack Sequences
class Q946:
    def validateStackSequences(self, pushed: List[int], popped: List[int]) -> bool:
        stack = []
        pushIdx = 0
        popIdx = 0
        while pushIdx < len(pushed):
            stack.append(pushed[pushIdx])
            while len(stack) > 0 and stack[-1] == popped[popIdx]:
                stack.pop()
                popIdx += 1
            pushedIdx += 1
        while popIdx < len(popped):
            if len(stack) == 0 or stack.pop() != popped[popIdx]:
                return False
            else:
                popIdx += 1
        if len(stack) > 0:
            return False
        else:
            return True

# 359. Logger Rate Limiter
class Logger:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.logLastTime = {}

    def shouldPrintMessage(self, timestamp: int, message: str) -> bool:
        """
        Returns true if the message should be printed in the given timestamp, otherwise returns false.
        If this method returns false, the message will not be printed.
        The timestamp is in seconds granularity.
        """

        if message not in self.logLastTime or timestamp - self.logLastTime[message] >= 10:
            self.logLastTime[message] = timestamp
            return True
        else:
            return False

# 1153. String Transforms Into Another String
class Q1153:
    def canConvert(self, str1: str, str2: str) -> bool:
        if str1 == str2: return True
        convert = {}
        for x, y in zip(str1, str2):
            if convert.setdefault(x, y) != y:
                return False
        return len(set([s for s in str2])) < 26
    
# 659. Split Array into Consecutive Subsequences
class Q659:
    def isPossible(self, nums: List[int]) -> bool:
        from collections import deque
        stack = deque([[nums[0], nums[0]]])
        stackItr = -1
        prev = nums[0]
        for i in range(1, len(nums)):
            if nums[i] - prev == 0:
                if stackItr == -1:
                    stack.append([prev, prev])
                else:
                    stack[stackItr][1] = prev
                    stackItr -= 1 
            elif nums[i] - prev == 1:

                while stack[0][1] != prev:
                    s = stack.popleft()
                    if s[1] - s[0] < 2: return False

                stack[-1][1] += 1
                stackItr = len(stack) - 2
                prev = nums[i]
            else:
                for s in stack:
                    if s[1] - s[0] < 2: return False
                    
                stack = deque([[nums[i], nums[i]]])
                stackItr = -1
                prev = nums[i]

        while stack:
            s = stack.popleft()
            if s[1] - s[0] < 2: return False
        return True

# 809. Expressive Words
class Q809:
    def expressiveWords(self, S: str, words: List[str]) -> int:
        if S == "":
            return(sum([w for w in words if w == ""]))
        ret = 0
        s = S
        for w in words:
            if w[0] != s[0]: continue
            wIdx = 0
            sIdx = 0
            prev = w[0]
            matchCnt = 0
            stretch = False
            match = True
            while wIdx < len(w):
                if sIdx >= len(s):
                    match = False
                    break
                if w[wIdx] == s[sIdx]:
                    if w[wIdx] == prev:
                        matchCnt += 1
                    else:
                        if stretch and matchCnt < 3:
                            match = False
                            break
                        stretch = False
                        matchCnt = 1
                        prev = w[wIdx]
                    wIdx += 1
                    sIdx += 1
                else:
                    if s[sIdx] == prev:
                        matchCnt += 1
                        sIdx += 1
                        stretch = True
                    else:
                        match = False
                        break
            while match and sIdx < len(s):
                if s[sIdx] != prev:
                    match = False
                    break
                stretch = True
                matchCnt += 1
                sIdx += 1
            if match and (not stretch or matchCnt >= 3):
                ret += 1
        return ret

# 1293. Shortest Path in a Grid with Obstacles Elimination
class Q1293:
    def shortestPath(self, grid: List[List[int]], k: int) -> int:
        from collections import deque
        R = len(grid)
        C = len(grid[0])
        dp = [[-1 for c in range(C)] for r in range(R)]
        dp[0][0] = k
        D = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        q = deque([(0, 0, 0, k)])
        while q:
            (r, c, l, k) = q.pop()
            if (r, c) == (R-1, C-1):
                return l
            for d in D:
                rr = r + d[0]
                cc = c + d[1]
                if 0 <= rr < R and 0 <= cc < C:
                    if grid[rr][cc] == 1:
                        if k > 0 and k > dp[rr][cc]:
                            dp[rr][cc] = k
                            q.appendleft((rr, cc, l + 1, k - 1))
                        else:
                            continue
                    else:
                        if k > dp[rr][cc]:
                            dp[rr][cc] = k
                            q.appendleft((rr, cc, l + 1, k))
                        else:
                            continue
        return -1

# 1231. Divide Chocolate
class Q1231:
    def maximizeSweetness(self, sweetness: List[int], K: int) -> int:
        def divide(target):
            s, cnt = 0, 0
            for x in sweetness:
                s += x
                if s >= target:
                    cnt += 1
                    s = 0
            return cnt
        
        lo, hi = min(sweetness), sum(sweetness)//(K+1)
        while lo < hi:
            target = (lo + hi +1)//2
            if target == hi:
                break
            if divide(target) < K+1:
                hi = target
            else:
                lo = target
        if divide(target) < K+1:
            return lo
        else:
            return hi

# 1368. Minimum Cost to Make at Least One Valid Path in a Grid
class Q1368:
    def minCost(self, grid: List[List[int]]) -> int:
        from collections import deque
        R = len(grid)
        C = len(grid[0])
        dp = [[R*C for c in range(C)] for r in range(R)]
        dp[0][0] = 0
        q = deque([(0, 0, 0)])
        D = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        dir = {1:(0, 1), 2:(0, -1), 3:(1, 0), 4:(-1, 0)}
        while q:
            (r, c, cost) = q.pop()
            for d in D:
                rr = r + d[0]
                cc = c + d[1]
                
                if 0 <= rr < R and 0 <= cc < C:
                    newCost = cost
                    if dir[grid[r][c]] != d:
                         newCost += 1
                    
                    if newCost < dp[rr][cc]:
                        dp[rr][cc] = newCost
                        q.appendleft((rr, cc, newCost))
        print(dp)
        return dp[R-1][C-1]

# 211. Design Add and Search Words Data Structure
class WordDictionary:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.trie = {}

    def addWord(self, word: str) -> None:
        """
        Adds a word into the data structure.
        """
        node = self.trie
        for char in word:
            node = node.setdefault(char, {})
        node['$'] = {}

    def search(self, word: str) -> bool:
        """
        Returns if the word is in the data structure. A word could contain the dot character '.' to represent any one letter.
        """
        def match(charIdx, parentNode):
            char = word[charIdx]
            if char == '.':
                for char in parentNode:
                    node = parentNode[char]
                    if charIdx == len(word) - 1:
                        if '$' in node:
                            return True
                    else:
                        if match(charIdx + 1, node):
                            return True
            else:
                if char in parentNode:
                    node = parentNode[char]
                    if charIdx == len(word) - 1:
                        if '$' in node:
                            return True
                    else:
                        return match(charIdx + 1, node)
            return False
        return match(0, self.trie)

# 125. Valid Palindrome
class Q125:
    def isPalindrome(self, s: str) -> bool:
        lIdx = 0
        rIdx = len(s) - 1
        validChar = set([chr(x+ord('a')) for x in range(26)]) | set([str(n) for n in range(10)])
        while lIdx < rIdx:
            lChar = s[lIdx].lower()
            rChar = s[rIdx].lower()
            if lChar not in validChar:
                lIdx += 1
                continue
            if rChar not in validChar:
                rIdx -= 1
                continue
            if lChar != rChar:
                return False
            lIdx += 1
            rIdx -= 1
        return True

# 270. Closest Binary Search Tree Value
class Q270:
    def closestValue(self, root: TreeNode, target: float) -> int:
        ret = root.val
        node = root
        while node:
            if abs(node.val - target) < abs(ret - target):
                ret = node.val
            if target > node.val:
                if node.right:
                    node = node.right
                else:
                    break
            elif target < node.val:
                if node.left:
                    node = node.left
                else:
                    break
            else:
                break
        if abs(node.val - target) < abs(ret - target):
                ret = node.val
        return ret
        
# 958. Check Completeness of a Binary Tree
class Q958:
    def isCompleteTree(self, root: TreeNode) -> bool:
        from collections import deque
        if not root: return True
        q = deque([root])
        while q:
            endFlag = False
            for _ in range(len(q)):
                node = q.pop()
                if node:
                    if endFlag:
                        return False
                    else:
                        q.appendleft(node.left)
                        q.appendleft(node.right)
                else:
                    endFlag = True
            if endFlag:
                for node in q:
                    if node: return False
                return True
        return True

# 426. Convert Binary Search Tree to Sorted Doubly Linked List
class Q426:
    def treeToDoublyList(self, root: 'Node') -> 'Node':
        if root is None: return None
        lEnd = root
        while lEnd.left:
            lEnd = lEnd.left
        rEnd = root
        while rEnd.right:
            rEnd = rEnd.right
        def dfs(node, pred, succ):
            if node.left:
                dfs(node.left, pred, node)
            else:
                if node != pred:
                    node.left = pred
                    pred.right = node
            if node.right:
                dfs(node.right, node, succ)
            else:
                if node != succ:
                    node.right = succ
                    succ.left = node
        dfs(root, lEnd, rEnd)
        lEnd.left = rEnd
        rEnd.right = lEnd
        return lEnd

# 317. Shortest Distance from All Buildings
class Q317:
    def shortestDistance(self, grid: List[List[int]]) -> int:
        from collections import deque
        R = len(grid)
        C = len(grid[0])
        BDG = []
        for r in range(R):
            for c in range(C):
                if grid[r][c] == 1:
                    BDG.append((r, c))
        COST = [[[-1 for c in range(C)] for r in range(R)] for bdg in BDG]
        D = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        for bdg, cost in zip(BDG, COST):
            q = deque([bdg + (0,)])
            while q:
                (r, c, l) = q.popleft()
                for d in D:
                    rr = r + d[0]
                    cc = c + d[1]
                    if 0 <= rr < R and 0 <= cc < C and grid[rr][cc] == 0 and cost[rr][cc] < 0:
                        cost[rr][cc] = l + 1
                        q.append((rr, cc, l + 1))
        ret = -1
        for r in range(R):
            for c in range(C):
                valid = True
                for cost in COST:
                    if cost[r][c] < 0:
                        valid = False
                if not valid: continue
                cost = sum([cost[r][c] for cost in COST])
                if ret < 0 or cost < ret:
                    ret = cost
        return ret

# 825. Friends Of Appropriate Ages
class Q825:
    def numFriendRequests(self, ages: List[int]) -> int:
        ages.sort(reverse = True)
        from collections import deque
        minAge = deque([]) # (A index, minimum friend request age)
        ret = 0
        for bIdx, b in enumerate(ages):
            while minAge and minAge[0][1]*0.5 + 7 >= b:
                aIdx, a = minAge.popleft()
                ret += (bIdx - aIdx - 1)
            minAge.append((bIdx, b))

        if len(minAge) > 0:
            bIdx, b = minAge.pop()
            while minAge:
                aIdx, a = minAge.popleft()
                ret += (bIdx - aIdx)
        print(ret)
        # check people of the same age
        i = 0
        while i < len(ages):
            if ages[i] > 14:
                k = 0
                a = ages[i]
                i += 1
                while i < len(ages) and ages[i] == a:
                    k += 1
                    i += 1
                ret += int(k*(k+1)/2)
            else:
                i += 1
        return ret

# 432. All O`one Data Structure
class AllOne:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.key2val = {}
        self.val2key = {}
        self.minVal = None
        self.maxVal = None

    def inc(self, key: str) -> None:
        """
        Inserts a new key <Key> with value 1. Or increments an existing key by 1.
        """
        val = self.key2val.get(key, 0)
        if val != 0:
            self.val2key[val].remove(key)
            if len(self.val2key[val]) == 0:
                self.val2key.pop(val)
                if self.minVal == val:
                    self.minVal = val + 1
        val += 1
        self.key2val[key] = val
        self.val2key.setdefault(val, set()).add(key)
        if self.maxVal is None or self.maxVal < val:
            self.maxVal = val
        if self.minVal is None or self.minVal > val:
            self.minVal = val

    def dec(self, key: str) -> None:
        """
        Decrements an existing key by 1. If Key's value is 1, remove it from the data structure.
        """
        val = self.key2val.get(key, 0)
        if val == 0: return
        if val == 1:
            self.key2val.pop(key)
        else:
            self.key2val[key] = val - 1
            self.val2key.setdefault(val - 1, set()).add(key)
        self.val2key[val].remove(key)
        if len(self.val2key[val]) == 0:
            self.val2key.pop(val)
            if val == self.maxVal or val == self.minVal:
                if len(self.val2key) > 0:
                    vals = list(self.val2key.keys())
                    self.maxVal = max(vals)
                    self.minVal = min(vals)
                else:
                    self.maxVal = None
                    self.minVal = None

    def getMaxKey(self) -> str:
        """
        Returns one of the keys with maximal value.
        """
        if self.maxVal:
            return next(iter(self.val2key[self.maxVal]))
        else:
            return ""

    def getMinKey(self) -> str:
        """
        Returns one of the keys with Minimal value.
        """
        if self.minVal:
            return next(iter(self.val2key[self.minVal]))
        else:
            return ""

# 1123. Lowest Common Ancestor of Deepest Leaves
class Q1123:
    def lcaDeepestLeaves(self, root: TreeNode) -> TreeNode:
        deepest = []
        def dfs(path):
            nonlocal deepest
            node = path[-1]
            if node.left is None and node.right is None:
                if len(deepest) == 0 or len(deepest[0]) == len(path):
                    deepest.append(path)
                elif len(deepest[0]) < len(path):
                    deepest = [path]
            else:
                if node.left:
                    dfs(path + [node.left])
                if node.right:
                    dfs(path + [node.right])
        dfs([root])
        path0 = deepest[0]
        for i in range(len(path0)-1, -1, -1):
            match = True
            for pathi in deepest[1:]:
                if pathi[i] != path0[i]:
                    match = False
                    break
            if match:
                return path0[i]

# 766. Toeplitz Matrix
class Q766:
    def isToeplitzMatrix(self, matrix: List[List[int]]) -> bool:
        prevRow = matrix[0]
        for thisRow in matrix[1:]:
            for x, y in zip(prevRow, thisRow[1:]):
                if x != y:
                    return False
            prevRow = thisRow
        return True

# 463. Island Perimeter
class Q463:
    def islandPerimeter(self, grid: List[List[int]]) -> int:
        R = len(grid)
        C = len(grid[0])
        D = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        ret = 0
        for r in range(R):
            for c in range(C):
                if grid[r][c] == 1:
                    ret += 4
                    for d in D:
                        rr = r + d[0]
                        cc = c + d[1]
                        if 0 <= rr < R and 0 <= cc < C and grid[rr][cc] == 1:
                            ret -= 1
        return ret

# 839. Similar String Groups
class Q839:
    def numSimilarGroups(self, A: List[str]) -> int:
        from itertools import combinations
        parent = {w:w for w in A}

        def similar(w1, w2):
            return sum([1 if char1 != char2 else 0 for char1, char2 in zip(w1, w2)]) <= 2
        
        def find(w):
            while parent[w] != w:
                # this step is for optmization, the parent of each word will be the root word of the group
                parent[w] = parent[parent[w]]

                w = parent[w]
            return w
        
        K = len(A[0])
        A = set(A)
        N = len(A)
        res = N
        if K > N:
            for w1, w2 in combinations(A, 2):
                if similar(w1, w2):
                    pw1 = find(w1)
                    pw2 = find(w2)
                    if pw1 != pw2:
                        parent[pw2] = pw1
                        res -= 1
        else:
            for w1 in A:
                pw1 = find(w1)
                for i, j in combinations(range(K), 2):
                    w2 = w1[:i] + w1[j] + w1[i+1:j] + w1[i] + w1[j+1:]
                    if w2 not in A: continue
                    if similar(w1, w2):
                        pw2 = find(w2)
                        if pw1 != pw2:
                            parent[pw2] = pw1
                            res -= 1
        return res

# 708. Insert into a Sorted Circular Linked List
class Q708:
    def insert(self, head: 'Node', insertVal: int) -> 'Node':
        newNode = Node(insertVal)
        if head is None:
            newNode.next = newNode
            return newNode
        if head.next == head:
            newNode.next = head
            head.next = newNode
            return head

        prevNode = head
        thisNode = head.next
        start = prevNode
        end = thisNode
        while 1:
            if thisNode.val < prevNode.val:
                start = thisNode
                end = prevNode
            if prevNode.val <= newNode.val <= thisNode.val:
                prevNode.next = newNode
                newNode.next = thisNode
                return head
            prevNode = thisNode
            thisNode = thisNode.next
            if prevNode == head:
                break
        end.next = newNode
        newNode.next = start
        return head

# 109. Convert Sorted List to Binary Search Tree
class Q109:
    def sortedListToBST(self, head: ListNode) -> TreeNode:
        if head is None:
            return None
        nums = [head.val]
        node = head
        while node.next:
            node = node.next
            nums.append(node.val)
    
        def list2bst(nums):
            if len(nums) == 1:
                return TreeNode(val = nums[0])
            if len(nums) == 2:
                root = TreeNode(val = nums[1])
                root.left = TreeNode(val = nums[0])
                return root
            rIdx = int(len(nums)/2)
            root = TreeNode(val = nums[rIdx])
            root.left = list2bst(nums[:rIdx])
            root.right = list2bst(nums[rIdx+1:])
            return root
        return list2bst(nums)
            
# 934. Shortest Bridge
class Q934:
    def shortestBridge(self, A: List[List[int]]) -> int:
        from collections import deque
        R = len(A)
        C = len(A[0])
        D = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        
        # find an island
        root = None
        for r in range(R):
            find = False
            for c in range(C):
                if A[r][c] == 1:
                    A[r][c] = 2
                    root = (r, c)
                    find = True
                    break
            if find:
                break
        
        # get the edges of the island
        edge = set()
        q = deque([root])
        while q:
            r, c = q.pop()
            for d in D:
                rr, cc = r + d[0], c + d[1]
                if 0 <= rr < R and 0 <= cc < C:
                    if A[rr][cc] == 1:
                        A[rr][cc] = 2
                        q.appendleft((rr, cc))
                    elif A[rr][cc] == 0:
                        edge.add((r, c))
        
        edge = deque(edge)
        dist = 0
        while edge:
            dist += 1
            for _ in range(len(edge)):
                r, c = edge.pop()
                for d in D:
                    rr, cc = r + d[0], c + d[1]
                    if 0 <= rr < R and 0 <= cc < C:
                        if A[rr][cc] == 1:
                            return dist - 1
                        elif A[rr][cc] == 0:
                            A[rr][cc] = 2
                            edge.appendleft((rr, cc))

        return -1

# 378. Kth Smallest Element in a Sorted Matrix
class Q378:
    def kthSmallest(self, matrix: List[List[int]], k: int) -> int:
        import heapq
        N = len(matrix)
        ptr = [(matrix[r][0], r, 0) for r in range(N)]
        heapq.heapify(ptr)
        cnt = 0
        ret = matrix[0][0]
        while cnt < k:
            ret, r, c = heapq.heappop(ptr)
            cnt += 1
            c += 1
            if c < N:
                heapq.heappush(ptr, (matrix[r][c], r, c))
        return ret

# 616. Add Bold Tag in String
class Q616:
    def addBoldTag(self, s: str, dict: List[str]) -> str:
        charIdx = {}
        for i, char in enumerate(s):
            charIdx.setdefault(char, []).append(i)
        
        boldRange = []
        for subs in dict:
            if len(subs) == 0 or subs[0] not in charIdx: continue
            for sIdx in charIdx[subs[0]]:
                eIdx = sIdx+len(subs)
                if s[sIdx:eIdx] == subs:
                    boldRange.append([sIdx, eIdx])
        
        if len(boldRange) == 0:
            return s
        boldRange.sort()

        mergeRange = []
        r1 = boldRange[0]
        for r2 in boldRange[1:]:
            if r2[0] <= r1[1]:
                r1[1] = max(r2[1], r1[1])
            else:
                mergeRange.append(r1)
                r1 = r2
        mergeRange.append(r1)
        
        ret = s[:mergeRange[0][0]]
        for r1, r2 in zip(mergeRange, mergeRange[1:]):
            ret += '<b>'+s[r1[0]:r1[1]]+'</b>'+s[r1[1]:r2[0]]
        
        r1 = mergeRange[-1]
        ret += '<b>'+s[r1[0]:r1[1]]+'</b>'+s[r1[1]:]
        return ret

# 658. Find K Closest Elements
class Q658:
    def findClosestElements(self, arr: List[int], k: int, x: int) -> List[int]:
        if len(arr) <= k:  return arr
        lptr, rptr = 0, k-1
        while rptr + 1 < len(arr) and ((abs(x - arr[lptr]) > abs(arr[rptr+1] - x)) or (arr[rptr+1] <= x)):
            lptr += 1
            rptr += 1
        return arr[lptr:rptr+1]

# 347. Top K Frequent Elements
class Q347:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        histo = {}
        for n in nums:
            histo[n] = histo.get(n, 0) + 1
        ret = list(histo.items())
        ret.sort(key=lambda x:x[1], reverse = True)
        return [x[0] for x in ret[:k]]

# 827. Making A Large Island
class Q823:
    def largestIsland(self, grid: List[List[int]]) -> int:
        from collections import deque
        
        R = len(grid)
        C = len(grid[0])
        D = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        islIdx = -1
        islArea = [] #
        for r in range(R):
            for c in range(C):
                if grid[r][c] != 1: continue
                grid[r][c] = islIdx
                q = deque([(r, c)]) # row, column
                area = 1
                while q:
                    x, y = q.pop()
                    for d in D:
                        xx, yy = x+d[0], y+d[1]
                        if 0 <= xx < R and 0 <= yy < C and grid[xx][yy] == 1:
                            grid[xx][yy] = islIdx
                            area += 1
                            q.appendleft((xx, yy))
                islArea.append(area)
                islIdx -= 1
        if len(islArea) == 1:
            return min(R*C, islArea[0] + 1)
        print(grid)
        maxArea = 1
        # group = set()
        for x in range(R):
            for y in range(C):
                if grid[x][y] != 0: continue
                area = 1
                isls = set()
                for d in D:
                    xx, yy = x+d[0], y+d[1]
                    if 0 <= xx < R and 0 <= yy < C and grid[xx][yy] != 0:
                        islIdx = abs(grid[xx][yy]+1)
                        if islIdx not in isls:
                            area += islArea[islIdx]
                            isls.add(islIdx)
                maxArea = max(maxArea, area)
        
        return maxArea

from queue import Queue
import threading
# 1242. Web Crawler Multithreaded
# """
# This is HtmlParser's API interface.
# You should not implement it, or speculate about its implementation
# """
#class HtmlParser(object):
#    def getUrls(self, url):
#        """
#        :type url: str
#        :rtype List[str]
#        """
import time
class HtmlParser:
    def crawl(self, startUrl: str, htmlParser: 'HtmlParser') -> List[str]:
        q = Queue()
        q.put(startUrl)
        ret = set([startUrl])
        host = startUrl[7:].split('/')[0]
        tCnt = 0
        lock = threading.Lock()
        workNum = 10
        def valid_host(url):
            return (len(url) > 7) and (url[7:].split('/')[0] == host)
        
        def get_url(url):
            nonlocal lock, tCnt
            newURL = htmlParser.getUrls(url)
            for x in newURL:
                if x in ret: continue
                if valid_host(x):
                    q.put(x)
                    ret.add(x)
            lock.acquire()
            tCnt -= 1
            lock.release()

        
        while 1:
            try:
                nextUrl = q.get(block=True, timeout=0.02)
                while 1:
                    lock.acquire()
                    if tCnt >= workNum:
                        lock.release()
                        time.sleep(0.015)
                    else:
                        tCnt += 1
                        lock.release()
                        break
                t = threading.Thread(target=get_url, args=(nextUrl,))
                t.start()
            except:
                break
                
# 1236. Web Crawler
class Q1236:
    def crawl(self, startUrl: str, htmlParser: 'HtmlParser') -> List[str]:
        host = startUrl[7:].split('/')[0]
        def valid_host(url):
            nonlocal host
            return len(url) > 7 and url[7:].split('/')[0] == host
        
        ret = set([startUrl])
        q = deque([startUrl])
        
        while q:
            url = q.pop()
            for nextUrl in htmlParser.getUrls(url):
                if nextUrl not in ret and valid_host(nextUrl):
                    ret.add(nextUrl)
                    q.appendleft(nextUrl)
        return list(ret)

# 71. Simplify Path
class Q71:
    def simplifyPath(self, path: str) -> str:
        pStack = []
        i = 0
        path = path.split('/')
        for p in path:
            if len(p) == 0:
                continue
            elif p == '.':
                continue
            elif p == '..':
                if len(pStack) > 0:
                    pStack.pop()
            else:
                pStack.append(p)
                
        return '/'+'/'.join(pStack)

# 936. Stamping The Sequence
class Q936:
    def movesToStamp(self, stamp: str, target: str) -> List[int]:
        if len(stamp) > len(target) or len(stamp) == 0: return []
        target = list(target)
        stamp = list(stamp)
        ret = []
        dp = set()
        
        def match(i):
            nonlocal target, stamp, ret
            change = False
            for tChar, sChar in zip(target[i: i + len(stamp)], stamp):
                if tChar == '?': continue
                if tChar != sChar: return False
                change = True
            if change:
                target[i: i+len(stamp)] = '?'*len(stamp)
                ret.append(i)
            dp.add(i)
            return change
        
        change = True
        while change:
            change = False
            for i in range(len(target)-len(stamp)+1):
                if i in dp: continue
                if match(i):
                    change = True
        return ret[::-1] if target == ['?']*len(target) else []

# 304. Range Sum Query 2D - Immutable
class NumMatrix:

    def __init__(self, matrix: List[List[int]]):
        self.memo = {}
        self.matrix = matrix

    def sumRegion(self, row1: int, col1: int, row2: int, col2: int) -> int:
        if row2 < row1 or col2 < col1: return 0
        if (row1, col1, row2, col2) not in self.memo:
            if row1 == 0 and col1 == 0:
                self.memo[(0, 0, row2, col2)] = sum([sum(self.matrix[r][:col2+1]) for r in range(row2+1)])
            else:
                self.memo[(row1, col1, row2, col2)] = self.sumRegion(0, 0, row2, col2) - self.sumRegion(0, 0, row1 -1, col2) - self.sumRegion(row1, 0, row2, col1-1)
        return self.memo[(row1, col1, row2, col2)]

# 76. Minimum Window Substring
class Q76:
    def minWindow(self, s: str, t: str) -> str:
        lp = 0
        rp = 0
        cntDict = {}
        tCnt = {}
        for c in t:
            tCnt[c] = tCnt.get(c, 0) + 1
        tSet = set(t)
        rmCnt = tCnt.copy()
        while len(rmCnt) > 0 and rp < len(s):
            if s[rp] in tSet:
                cntDict[s[rp]] = cntDict.get(s[rp], 0) + 1
                
                r = rmCnt.get(s[rp], 0)
                if r == 1:
                    rmCnt.pop(s[rp])
                elif r > 1:
                    rmCnt[s[rp]] -= 1
                    
            rp += 1
        
        if len(rmCnt) > 0:
            return ''
        while s[lp] not in tSet:
            lp += 1
        ret = s[lp: rp]
        while 1:
            while cntDict[s[lp]] == tCnt[s[lp]] and rp < len(s):
                if s[rp] in tSet:
                    cntDict[s[rp]] += 1
                rp += 1
            while s[lp] not in t or cntDict[s[lp]] != tCnt[s[lp]]:
                if s[lp] in tSet:
                    cntDict[s[lp]] -= 1
                lp += 1
            if rp-lp < len(ret):
                ret = s[lp:rp]
            if rp == len(s):
                break
        return ret

# 1216. Valid Palindrome III
class Q1216:
    def isValidPalindrome(self, s: str, k: int) -> bool:
        lp = 0
        rp = len(s) - 1
        memo = {} # s: [max invalid k, min valid k]
        def dp(lp, rp, k):
            if lp >= rp:
                return True
            subS = s[lp:rp+1]
            # print(lp, rp, subS)
            if subS not in memo or memo[subS][0] < k < memo[subS][1]:
                isValid = True
                while lp < rp:
                    if s[lp] != s[rp]:
                        if k == 0:
                            isValid = False
                            break
                        else:
                            isValid = dp(lp + 1, rp, k - 1) or dp(lp, rp - 1, k - 1)
                            break
                    lp += 1
                    rp -= 1
                bound = memo.setdefault(subS, [-1, len(subS) - 1])
                if isValid:
                    bound[1] = min(k, bound[1])
                else:
                    bound[0] = max(k, bound[0])
            return k >= memo[subS][1]
        return dp(lp, rp, k)

# 1424. Diagonal Traverse II
class Q1424:
    def findDiagonalOrder(self, nums: List[List[int]]) -> List[int]:
        ret = []
        R = len(nums)
        C = max([len(row) for row in nums])
        itr = deque([[] for c in range(C)])
        for row in nums:
            ret += list(itr.popleft())[::-1]
            itr.append([])
            for r, i in zip(row, itr):
                i.append(r)
        while itr:
            ret += list(itr.popleft())[::-1]
        return ret

# 921. Minimum Add to Make Parentheses Valid
class Q921:
    def minAddToMakeValid(self, S: str) -> int:
        parCnt = 0
        ret = 0
        for char in S:
            if char == '(':
                parCnt += 1
            elif char == ')':
                parCnt -= 1
            if parCnt == -1:
                ret += 1
                parCnt = 0
        ret += parCnt
        return ret

# 1233. Remove Sub-Folders from the Filesystem
class Q1233:
    def removeSubfolders(self, folder: List[str]) -> List[str]:
        trie = [None, {}] #[path of the folder, trie node]
        for rawPath in folder:
            path = rawPath.split('/')[1:]
            node = trie
            for f in path:
                node = node[1].setdefault(f, [None, {}])
            node[0] = rawPath
        
        ret = []
        q = deque([trie])
        while q:
            node = q.pop()
            if node[0]:
                ret.append(node[0])
                continue
            else:
                for f in node[1]:
                    q.appendleft(node[1][f])
        return ret

# 419. Battleships in a Board
class Q419:
    def countBattleships(self, board: List[List[str]]) -> int:
        R = len(board)
        C = len(board[0])
        cnt = 0
        for r in range(R):
            for c in range(C):
                if board[r][c] == 'X':
                    if (r+1==R or board[r+1][c] == '.') and (c+1==C or board[r][c+1]=='.'):
                        cnt += 1
        return cnt

# 1570. Dot Product of Two Sparse Vectors
class SparseVector:
    def __init__(self, nums: List[int]):
        self.non0 = {}
        for i, x in enumerate(nums):
            if x != 0:
                self.non0[i] = x

    # Return the dotProduct of two sparse vectors
    def dotProduct(self, vec: 'SparseVector') -> int:
        ret = 0
        for i in self.non0:
            if i in vec.non0:
                ret += self.non0[i]*vec.non0[i]
        return ret

# 381. Insert Delete GetRandom O(1) - Duplicates allowed
from random import randint
class RandomizedCollection:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.cnt = 0
        self.valCnt = {}

    def insert(self, val: int) -> bool:
        """
        Inserts a value to the collection. Returns true if the collection did not already contain the specified element.
        """
        self.cnt += 1
        self.valCnt[val] = self.valCnt.get(val, 0) + 1
        return self.valCnt[val] == 1
    

    def remove(self, val: int) -> bool:
        """
        Removes a value from the collection. Returns true if the collection contained the specified element.
        """
        if val in self.valCnt:
            self.valCnt[val] -= 1
            if self.valCnt[val] == 0:
                self.valCnt.pop(val)
            self.cnt -= 1
            return True
        else:
            return False
        

    def getRandom(self) -> int:
        """
        Get a random element from the collection.
        """
        pick = randint(1, self.cnt)
        s = 0
        for val, cnt in self.valCnt.items():
            s += cnt
            if s >= pick:
                return val

# 536. Construct Binary Tree from String
class Q536:
    def str2tree(self, s: str) -> TreeNode:
        if len(s) == 0:
            return None

        def dfs(i):
            nonlocal s
            j = i+1
            while j < len(s) and s[j] not in ['(', ')']:
                j += 1
            node = TreeNode(int(s[i:j]))
            i = j
            
            if i < len(s) and s[i] == '(':
                node.left, i = dfs(i+1)
                if i < len(s) and s[i] == '(':
                    node.right, i = dfs(i+1)
            return node, i+1
        
        return dfs(0)[0]

# 1269. Number of Ways to Stay in the Same Place After Some Steps
from math import comb
class Q1269:
    def numWays(self, steps: int, arrLen: int) -> int:
        # equivalent to the possible ways of moving up to steps and return to 0 index
        # number of moves to the left and right must be the same
        # so it becomes, from for step in range(0, steps//2*2+1, 2), how many ways to arrange move step/2 left and step/2 right
        # should remain in the array
        
        memo = {(1, 1, 0):1, (0, 0, 0): 1}
        def backtrack(idx, lstep, rstep):
            nonlocal arrLen, memo
            if lstep - rstep != idx:
                return 0
            if (idx, lstep, rstep) not in memo:
                cnt = 0
                if idx > 0 and lstep > 0:
                    cnt += backtrack(idx - 1, lstep - 1, rstep)   
                if idx < arrLen - 1 and rstep > 0:
                    cnt += (backtrack(idx + 1, lstep, rstep - 1))
                cnt %= (10**9 + 7)
                memo[(idx, lstep, rstep)] = cnt
            return memo[(idx, lstep, rstep)]
        cnt = 0
        for k in range(0, steps//2*2+1, 2):
            cnt += backtrack(0, k//2, k//2)*comb(steps, steps - k)
            cnt %= (10**9 + 7)
        return cnt

# 373. Find K Pairs with Smallest Sums
import heapq
class Q373:
    def kSmallestPairs(self, nums1: List[int], nums2: List[int], k: int) -> List[List[int]]:
        if len(nums1) == 0 or len(nums2) == 0:
            return []
        ret = []
        q = [(nums1[i]+nums2[0], i, 0) for i in range(len(nums1))]
        heapq.heapify(q)
        while len(ret) < k and q:
            (_, p1, p2) = heapq.heappop(q)
            ret.append([nums1[p1], nums2[p2]])
            if p2 + 1 < len(nums2):
                heapq.heappush(q, (nums1[p1] + nums2[p2+1], p1, p2+1))
        return ret

# 1439. Find the Kth Smallest Sum of a Matrix With Sorted Rows
import heapq
class Q1439:
    def kSmallestPairs(self, nums1: List[int], nums2: List[int], k) -> List[int]:
        if len(nums1) == 0 or len(nums2) == 0:
            return []
        ret = []
        q = [(nums1[i]+nums2[0], i, 0) for i in range(len(nums1))]
        heapq.heapify(q)
        while q and len(ret) < k:
            s, p1, p2 = heapq.heappop(q)
            ret.append(s)
            if p2 + 1 < len(nums2):
                heapq.heappush(q, (nums1[p1]+nums2[p2+1], p1, p2 + 1))
        return ret
    
    def kthSmallest(self, mat: List[List[int]], k: int) -> int:
        ret = mat[0][:k]
        for row in mat[1:]:
            ret = self.kSmallestPairs(ret, row, k)
        return ret[k-1]

# 43. Multiply Strings
class Solution:
    def multiply(self, num1: str, num2: str) -> str:
        char2int = {str(i):i for i in range(10)}
        if len(num1) < len(num2):
            num1, num2 = num2, num1
        subMult = []
        digit1 = 0
        for x1 in num1[::-1]:
            x1 = char2int[x1]
            carry = 0
            mult = [0]*digit1
            for x2 in num2[::-1]:
                x2 = char2int[x2]
                m = x1*x2 + carry
                mult.append(m%10)
                carry = m//10
            mult.append(carry)
            subMult.append(mult)
            digit1 += 1
        ret = []
        carry = 0
        digit = 1
        L = max([len(x) for x in subMult])

        for i in range(L):
            s = carry
            for row in subMult:
                if i < len(row):
                    s += row[i]
            
            ret .append(str(s%10))
            carry = s//10
        ret.append(str(carry))

        while len(ret) > 1:
            if ret[-1] != '0':
                break
            ret.pop()
        ret = ''.join(ret[::-1])
        return ret

# 1411. Number of Ways to Paint N × 3 Grid
class Q1411:
    def numOfWays(self, n: int) -> int:
        if n == 0:
            return 0
        a121 = 6
        a123 = 6
        for _ in range(n-1):
            a121Tmp = (a121*3 + a123*2)
            a123Tmp = (a121*2 + a123*2)
            a121 = a121Tmp%(10**9 + 7)
            a123 = a123Tmp%(10**9 + 7)
        return (a121 + a123)%(10**9 + 7)

# 759. Employee Free Time
# Definition for an Interval.
class Interval:
    def __init__(self, start: int = None, end: int = None):
        self.start = start
        self.end = end
class Q759:
    def merge(self, schedule):
        S = [i for s in schedule for i in s]
        if len(S) == 0:
            return []
        S.sort(key = lambda s:s.start)
        prevInt = S[0]
        p = 1
        merged = [prevInt]
        for thisInt in S[1:]:
            if thisInt.start > prevInt.end:
                merged.append(thisInt)
                prevInt = thisInt
            else:
                prevInt.end = max(prevInt.end, thisInt.end)
        return merged
        
    def employeeFreeTime(self, schedule):
        merged = self.merge(schedule)
        if len(merged) == 0:
            return []
        prevInt = merged[0]
        ret = []
        for thisInt in merged[1:]:
            ret.append(Interval(prevInt.end, thisInt.start))
            prevInt = thisInt
        return ret

# 480. Sliding Window Median
import bisect
class Q480:
    def medianSlidingWindow(self, nums: List[int], k: int) -> List[float]:
        win = sorted([x for x in nums[:k]])
        p = k
        midP = k//2
        isOdd = (k%2 == 1)
        if isOdd:
            ret = [win[midP]]
        else:
            ret = [(win[midP-1]+win[midP])/2]
        while p < len(nums):
            # delete the leftmost window number
            del win[bisect.bisect_left(win, nums[p-k])]
            # add the new number
            bisect.insort(win, nums[p])
            if isOdd:
                ret.append(win[midP])
            else:
                ret.append((win[midP-1]+win[midP])/2)
            p += 1
        return ret

# 995. Minimum Number of K Consecutive Bit Flips
class Solution:
    def minKBitFlips(self, A: List[int], K: int) -> int:
        hint = [False]*len(A)
        flipFlag = False
        cnt = 0
        for i in range(len(hint)):
            if (A[i] == 0 and flipFlag == hint[i]) or (A[i] == 1 and flipFlag != hint[i]):
                if i+K-1 >= len(hint): return -1
                flipFlag = not flipFlag
                cnt += 1
            if i+K < len(hint):
                hint[i+K] = flipFlag
        return cnt
    
# 308. Range Sum Query 2D - Mutable
# class NumMatrix:
    def __init__(self, matrix: List[List[int]]):
        for row in matrix:
            for c in range(1, len(row)):
                row[c] += row[c-1]
        self.matrix = matrix

    def update(self, row: int, col: int, val: int) -> None:
        if col == 0:
            diff = self.matrix[row][col] - val
        else:
            diff = self.matrix[row][col] - self.matrix[row][col-1] - val

        for c in range(col, len(self.matrix[row])):
            #print(self.matrix[row][c])
            self.matrix[row][c] -= diff
            #print(self.matrix[row][c])
        #print(self.matrix)
            

    def sumRegion(self, row1: int, col1: int, row2: int, col2: int) -> int:
        sum = 0
        for row in self.matrix[row1:row2+1]:
            sum += row[col2]
            if col1 > 0:
                sum -= row[col1-1]
        return sum

# 460. LFU Cache
import bisect
class LFUCache:

    def __init__(self, capacity: int):
        self.key2freq = {} # char -> frequency
        self.freq2key = {} # frequency -> [key]
        self.cnt = 0
        self.capacity = capacity
        self.cache = {}
        
    def key_rm(self, key):
        # remove key from freq2key
        freq = self.key2freq[key]
        keyL = self.freq2key[freq]
        if len(keyL) == 1:
            self.freq2key.pop(freq)
        else:
            rmIdx = 0
            while keyL[rmIdx] != key:
                rmIdx += 1
            #rint(key, keyL, rmIdx)
            del keyL[rmIdx]

    
    def key_add(self, key):
        # add key to freq2key
        freq = self.key2freq[key]
        self.freq2key.setdefault(freq, []).append(key)
    
    def get(self, key: int) -> int:
        #rint('before get key: {}'.format(key), self.freq2key)
        if key not in self.cache:
            ret = -1
        else:            
            self.key_rm(key)
            self.key2freq[key] += 1
            self.key_add(key)
            ret = self.cache[key]
        #rint('after get key: {}'.format(key), self.freq2key)
        return ret
            
    def put(self, key: int, value: int) -> None:
        if self.capacity == 0:
            return
        #rint('before put key: {} and value: {}'.format(key, value), self.freq2key)
        freq = self.key2freq.get(key, 0)
        if freq == 0:
            if self.cnt < self.capacity:
                self.cnt += 1
            else:
                minFreq = min(self.freq2key.keys())
                keyL = self.freq2key[minFreq]
                #print('target cache', cache)
                rmKey = keyL[0]
                self.key_rm(rmKey)
                self.key2freq.pop(rmKey)
                self.cache.pop(rmKey)
        else:
            self.key_rm(key)
        self.key2freq[key] = freq + 1
        self.key_add(key)
        self.cache[key] = value
        #rint('after put key: {} and value: {}'.format(key, value), self.freq2key)

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
"""
reverse every sublist of length k, and connect them
there are two main problems here:
1. how to reverse the k-node sublist;
2. how to connect a sublist to the previous one.
3. how to keep the last sublist that shorter than k as it is.
The two problems should be solved while running through the list.
For the problem 1:
Let's use pL and pR to anchor the left and right end node of a sublist, respectively.
Note that we need to remember where the head and tail of the sublist, so we use subHead and subTaile to remember the head and the tail of the sublist.
The subTail is the first node the the sublist, for the subHead, let's initialize it as the first node of sublist, and update it while reversing the sublist.
For the problem 2:
we should remember the tail of the reversed part of the list, so let's keep it in a variable named tail.
After we have reversed a sublist, tail.next = subHead, tail = subTail. We can initailize the tail as the first node of the list.
For the problem 3:
For each round of reversing a sublist, we first set the pL and pR. If we are not able to set the pR, i.e., the remaining number of nodes is less than k, then
we set tail.next = pL and return
"""
class Solution:
    def reverseKGroup(self, head: ListNode, k: int) -> ListNode:
        if head is None or k == 1: return head
        ret = head
        cnt = 1
        while cnt < k and ret.next:
            ret = ret.next
            cnt += 1
        if cnt < k:
            return ret

        tail = head
        
        pL = head
        pR = head
        subHead = pL
        subTail = pL
        cnt = 1

        while 1:
            while cnt < k and pR and pR.next:
                pR = pR.next
                cnt += 1
            if cnt < k:
                if ret:
                    tail.next = pL
                    return ret
                else:
                    return head

            pRNext = pR.next
            while pL != pRNext:
                pLNext = pL.next

                pL.next = subHead
                subHead = pL
                pL = pLNext

            tail.next = subHead
            tail = subTail
            pR = pL
            subHead = pL
            subTail = pL
            cnt = 1

from collections import deque
class Q752:
    def bfs(self, deadends: List[str], target: str) -> int:
        visited = set(deadends)
        if '0000' in visited:
            return -1
        q = set(['0000'])
        step = 0
        while q:
            nextQ = set()
            for cur in q:
                if cur == target:
                    return step
                visited.add(cur)
                for i in range(4):
                    nxtPlus = cur[:i] + str((int(cur[i]) + 1) % 10) + cur[i+1:]
                    if nxtPlus not in visited:
                        nextQ.add(nxtPlus)
                    nxtMinus = cur[:i] + str((int(cur[i]) - 1) % 10) + cur[i+1:]
                    if nxtMinus not in visited:
                        nextQ.add(nxtMinus)
            step += 1
            q = nextQ
        return -1
    
    def dual_bfs(self, deadends: List[str], target: str) -> int:
        visited = set(deadends)
        if '0000' in visited:
            return -1
        else:
            visited.add('0000')
        q1 = set(['0000'])
        q2 = set([target])
        step = 0
        while q1 and q2:
            temp = set()
            for cur in q1:
                if cur in q2:
                    return step
                visited.add(cur)
                for i in range(4):
                    nxtPlus = cur[:i] + str((int(cur[i]) + 1) % 10) + cur[i+1:]
                    if nxtPlus not in visited:
                        temp.add(nxtPlus)

                    nxtMinus = cur[:i] + str((int(cur[i]) - 1) % 10) + cur[i+1:]
                    if nxtMinus not in visited:
                        temp.add(nxtMinus)
            step += 1
            if len(q2) <= len(temp):
                q1, q2 = q2, temp
            else:
                q1 = temp
        return -1
    
    def openLock(self, deadends: List[str], target: str) -> int:
        return self.bfs(deadends, target)

def infix2postfix(expr):
    s = expr.split(' ')
    def convert(s):
        p = 0
        ret0 = []
        while p < len(s):
            if s[p] == '(':
                pCnt = -1
                tmp = []
                while pCnt != 0:
                    p += 1
                    if s[p] == '(':
                        pCnt -= 1
                    elif s[p] == ')':
                        pCnt += 1
                    tmp.append(s[p])
                tmp.pop()
                ret0.append(convert(tmp))
            else:
                ret0.append(s[p])
            p += 1

        p = 0
        ret1 = []
        while p < len(ret0):
            if ret0[p] in ['*', '/']:
                ret1.append(' '.join([ret1.pop(), ret0[p+1], ret0[p]]))
                p += 2
            else:
                ret1.append(ret0[p])
                p += 1

        ret2 = []
        p = 0
        while p < len(ret1):
            if ret1[p] in ['+', '-']:
                ret2.append(' '.join([ret2.pop(), ret1[p+1], ret1[p]]))
                p += 2
            else:
                ret2.append(ret1[p])
                p += 1
        assert len(ret2) == 1
        return ret2[0]
    return convert(s)



if __name__ == '__main__':
    infix = '1 + 1 * ( ( 2 - 3 ) * 4 * ( 5 - 1 ) ) / 2'
    postfix = infix2postfix(infix)
    t = eval(infix)
    p = postfix.split(' ')
    s = []
    for e in p:
        if e in ['+', '-', '*', '/']:
            op2 = s.pop()
            op1 = s.pop()
            if e == '+':
                s.append(op1 + op2)
            elif e == '-':
                s.append(op1 - op2)
            elif e == '*':
                s.append(op1 * op2)
            else:
                s.append(op1 / op2)
        else:
            s.append(int(e))
    print(t == s[0])



