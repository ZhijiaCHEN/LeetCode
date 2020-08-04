
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
if __name__ == '__main__':
    q = TimeMap()
    inputs = [[],["foo","bar",1],["foo",1],["foo",3],["foo","bar2",4],["foo",4],["foo",5]]
    for x in inputs[1:]:
        if len(x) == 3:
            print(q.set(*x))
        if len(x) == 2:
            print(q.get(*x))

    # q.put(4,4)
    # print(q.get(1))
    # print(q.get(3))
    # print(q.get(4))



