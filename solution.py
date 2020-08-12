
from collections import deque
from heapq import heappushpop
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


if __name__ == '__main__':
    q = Q91()
    print(q.numDecodings("20"))


    # q.put(4,4)
    # print(q.get(1))
    # print(q.get(3))
    # print(q.get(4))



