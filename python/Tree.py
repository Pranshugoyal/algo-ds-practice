
################################################################################
# --------------------- Tree must do list from GeeksforGeeks ----------------- #
################################################################################

from collections import deque, defaultdict

class Node:
    def __init__(self, data):
        self.data = data
        self.left = None
        self.right = None

def serializeTreeLevelOrder(root):
    if not root:
        return []

    arr = []
    q = deque()
    q.append(root)
    while len(q) > 0:
        node = q.popleft()
        if not node:
            arr.append(None)
            continue

        arr.append(node.data)

        q.append(node.left)
        q.append(node.right)

    if len(arr) > 2:
        while not arr[-1] and not arr[-2]:
            arr.pop()
    return arr

def deserializeTreeLevelOrder(arr):
    if len(arr) == 0:
        return None
    
    root = Node(arr[0])
    q = deque()
    last = root
    i = 1
    while i < len(arr)-1:
        if arr[i]:
            last.left = Node(arr[i])
            q.append(last.left)
        if arr[i+1]:
            last.right = Node(arr[i+1])
            q.append(last.right)

        last = q.popleft()
        i += 2

    return root

def serializeTreePreOrder(root):
    if not root:
        return [None]

    arr = []
    arr.append(root.data)
    arr += serializeTreePreOrder(root.left)
    arr += serializeTreePreOrder(root.right)
    return arr

def deserializeTreePreOrder(arr):
    if len(arr) == 0:
        return None

    def deserUtil(i):
        nonlocal arr

        if not arr[i]:
            return (None, i+1) 

        root = Node(arr[i])
        i += 1
        if i < len(arr):
            (root.left, i) = deserUtil(i) 
        if i < len(arr):
            (root.right, i) = deserUtil(i) 

        return (root, i)

    return deserUtil(0)[0]

def leftTreeView(root):
    if not root:
        return []
    
    res = [root.data]

    leftTree = leftTreeView(root.left)
    rightTree = leftTreeView(root.right)

    if len(leftTree) >= len(rightTree):
        return res + leftTree
    
    return res + leftTree + rightTree[len(leftTree):]

def leftTreeViewLoT(root):
    if not root:
        return []

    q = deque()
    q.append(root)
    res = []
    while len(q) > 0:
        res.append(q[0].data)
        q.append(None)
        while q[0]:
            node = q.popleft()
            if node.left:
                q.append(node.left)
            if node.right:
                q.append(node.right)
        q.popleft()
    return res

def isBST(root, min=None, max=None):
    if not root:
        return True
    
    def rangeCheck(v,min,max):
        res = True
        if min:
            res = res and v > min
        if max:
            res = res and v < max
        return res
    
    return (rangeCheck(root.data, min, max) and
            self.isBST(root.left, min=min, max=root.data) and
            self.isBST(root.right, min=root.data, max=max))

def bottomView(root):
    if not root:
        return []

    q = []
    q.append(root.data)
    min,max,zero = 0,0,0
    
    def bottomTreeViewUtil(root, pos):
        nonlocal q,min,max,zero

        if pos < min:
            q.insert(0, root.data)
            min = pos
            zero += 1
        elif pos > max:
            q.append(root.data)
            max = pos
        else:
            q[zero+pos] = root.data

    tq = deque()
    tq.append((root,0))
    node, pos = None, 0
    while len(tq) > 0:
        (node, pos) = tq.popleft()
        bottomTreeViewUtil(node, pos)
        if node.left:
            tq.append((node.left, pos-1))
        if node.right:
            tq.append((node.right, pos+1))

    return q

def verticalOrder(root):
    if not root:
        return []

    q = []
    q.append([])
    zero = 0
    
    def util(root, pos):
        nonlocal q,zero

        if pos < -zero:
            q.insert(0, [root.data])
            min = pos
            zero += 1
        elif pos > len(q)-zero-1:
            q.append([root.data])
            max = pos
        else:
            q[zero+pos].append(root.data)

    tq = deque()
    tq.append((root,0))
    node, pos = None, 0
    while len(tq) > 0:
        (node, pos) = tq.popleft()
        util(node, pos)
        if node.left:
            tq.append((node.left, pos-1))
        if node.right:
            tq.append((node.right, pos+1))

    return [item for sublist in q for item in sublist]

def connect(root):
    q = deque()
    q.append(root)
    last = None
    while len(q) > 0:
        q.append(None)
        last = None
        while q[0]:
            node = q.popleft()
            if last:
                last.nextRight = node
            last = node
            if node.left:
                q.append(node.left)
            if node.right:
                q.append(node.right)
        q.pop()

def findSpiral(root):
    if not root:
        return []

    res = []
    q = deque()
    l2r = False
    first, second = None, None

    q.append(root)
    while len(q) > 0:
        #print("Level Start", q)
        q.append(None)
        while q[0]:
            node = q.popleft()
            #print("On node:", node.data)
            res.append(node.data)
            if l2r:
                first,second = node.left, node.right
            else:
                first,second = node.right, node.left

            if first:
                q.append(first)
            if second:
                q.append(second)

        q.popleft()
        q.reverse()
        l2r = not l2r
        #print("Level end", q, "\n")
    return res

def treeSpiralTravel(root):
    if not root:
        return []

    s1 = deque()
    s2 = deque()

    res = []
    s1.append(root)
    while len(s1) > 0 or len(s2) > 0:
        while len(s1) > 0:
            node = s1.pop()
            res.append(node.data)
            if node.right:
                s2.append(node.right)
            if node.left:
                s2.append(node.left)

        while len(s2) > 0:
            node = s2.pop()
            res.append(node.data)
            if node.left:
                s1.append(node.left)
            if node.right:
                s1.append(node.right)
    return res

def treeLevelOrderTraversal(root):
    res = []
    def util(root, level):
        if not root:
            return

        nonlocal res

        if level == 1:
            res.append(root.data)
            return
        if root.left:
            util(root.left, level-1)
        if root.right:
            util(root.right, level-1)

    def height(root) -> int:
        if not root:
            return 0

        return max(height(root.left), height(root.right)) + 1

    for i in range(height(root)):
        res.append("Level: " + str(i))
        util(root, i+1)
    return res

def treeLevelOrderTraversalByQueue(root):
    if not root:
        return []
    
    res = []
    q = deque()
    q.append(root)
    while len(q) > 0:
        node = q.popleft()
        res.append(node.data)
        if node.left:
            q.append(node.left)
        if node.right:
            q.append(node.right)
    return res

def binaryTreeToDoublyLinkedList(root):
    # Returns (head, tail) of converted doubly linked list
    def util(root):
        if not root:
            return (None, None)
        elif not root.left and not root.right:
            return (root, root)

        left = util(root.left)
        right = util(root.right)
        
        head = left[0] if left[0] else root
        tail = right[1] if right[1] else root

        if left[1]:
            left[1].right = root
        root.left = left[1]

        if right[0]:
            right[0].left = root
        root.right = right[0]

        return (head, tail)

    return util(root)[0]

# return true/false denoting whether the tree is Symmetric or not
def isSymmetric(root):
    if not root:
        return True

    def symUtil(r1, r2) -> bool:
        if not r1 and not r2:
            return True
        elif (r1 and not r2) or (r2 and not r1):
            return False

        if r1.data != r2.data:
            return False

        return symUtil(r1.left, r2.right) and symUtil(r1.right, r2.left)
    
    return symUtil(root.left, root.right)

def maxPathSum(root):
    # Returns (maxPathSum, maxSumLeafToRoot)
    def mpsUtil(root, hasParent=True):
        if not root:
            return (None, 0)
        elif not root.left and not root.right:
            return (None, root.data)

        left = mpsUtil(root.left)
        right = mpsUtil(root.right)

        maxSumLeafToRoot = max(left[1], right[1]) + root.data
        maxSumLeafToLeafThroughRoot = left[1] + right[1] + root.data

        maxSumLeafToLeaf = None
        if left[0] and right[0]:
            maxSumLeafToLeaf = max(left[0], right[0], maxSumLeafToLeafThroughRoot)
        elif left[0]:
            maxSumLeafToLeaf = max(left[0], maxSumLeafToLeafThroughRoot)
        elif right[0]:
            maxSumLeafToLeaf = max(right[0], maxSumLeafToLeafThroughRoot)
        else:
            maxSumLeafToLeaf = maxSumLeafToLeafThroughRoot if not hasParent else None

        return (maxSumLeafToLeaf, maxSumLeafToRoot)

    return mpsUtil(root, hasParent=False)[0]

################################################################################
# -------------------------- Love's Sheet Binary Tree ------------------------ #
################################################################################

#https://practice.geeksforgeeks.org/problems/height-of-binary-tree/1
def getHeight(root):
    if not root:
        return 0

    return max(getHeight(root.left), getHeight(root.right)) + 1

#https://practice.geeksforgeeks.org/problems/level-order-traversal/1
def levelOrderTraversal(root):
    def traverseLevel(root, l, c=0):
        if not root:
            return []
        elif l == c:
            return [root.data]

        return traverseLevel(root.left, l, c+1) + traverseLevel(root.right, l, c+1)
    
    res = []
    for level in range(getHeight(root)):
        res += traverseLevel(root, level)
    return res

#https://practice.geeksforgeeks.org/problems/reverse-level-order-traversal/1
def reverseLevelOrder(root):
    def traverseLevel(root, l, c=0):
        if not root:
            return []
        elif l == c:
            return [root.data]

        return traverseLevel(root.left, l, c+1) + traverseLevel(root.right, l, c+1)
    
    res = []
    for level in reversed(range(getHeight(root))):
        res += traverseLevel(root, level)
    return res

#https://practice.geeksforgeeks.org/problems/diameter-of-binary-tree/1
def diameterOfBinaryTree(root):
    def diaUtil(root):
        if not root:
            return (0,0)

        left, right = diaUtil(root.left), diaUtil(root.right)
        via = left[0] + right[0] + 1
        height = max(left[0], right[0]) + 1
        return (height, max(via, left[1], right[1]))

    return diaUtil(root)[1]

#https://www.geeksforgeeks.org/create-a-mirror-tree-from-the-given-binary-tree/
#https://leetcode.com/problems/invert-binary-tree/submissions/
def mirrorBinaryTree(root):
    if not root:
        return None

    mRoot = TreeNode(root.val)
    mRoot.left = self.invertTree(root.right)
    mRoot.right = self.invertTree(root.left)
    return mRoot

#https://www.educative.io/edpresso/what-is-morris-traversal
def morrisInorderTreeTraversal(root):
    curr = root
    res = []
    while curr:
        if curr.left is None:
            res.append(curr.data)
            curr = curr.right
        else:
            lmax = curr.left
            while lmax.right and lmax.right is not curr:
                lmax = lmax.right

            if lmax.right is None:
                lmax.right = curr
                curr = curr.left
            else:
                lmax.right = None
                res.append(curr.data)
                curr = curr.right
    return res

#https://pdf.sciencedirectassets.com/271600/1-s2.0-S0167642300X00913/1-s2.0-0167642388900639/main.pdf
#NOTE: This is buggy, needs testing and fixes
def morrisStyleIterativeInorder(root):
    curr, stack = root, []
    res = []

    def rightOrPeek(node):
        return curr.right if curr.right else stack[-1] if stack else None

    while curr:
        if curr.left is None:
            res.append(curr.data)
            curr = rightOrPeek(curr)
        else:
            if stack and stack[-1] is curr:
                res.append(curr.data)
                stack.pop()
                curr = rightOrPeek(curr)
            else:
                stack.append(curr)
                curr = curr.left
    return res

class MorrisStack:
    def __init__(self, root):
        self.root = None
        self.last = None

    def push(self, node):
        p = self._loopParent(node)
        if p:
            p.right = node
            self.last = node
            if not self.root:
                self.root = node

    def peek(self):
        return self.last

    def pop(self):
        node = self.peek()
        if not node:
            return None
        self._loopParent(node).right = None

        last = node
        while last.right and last.right.left is not node:
            last = last.right
        self.last = last.right
        if self.last is None:
            self.root = None

        return node

    def _loopParent(self, node):
        if not node.left:
            return None

        lmax = node.left
        while lmax.right and lmax.right is not node:
            lmax = lmax.right

        return lmax

    def __contains__(self, node):
        p = self._loopParent(node)
        if p:
            return p.right is node
        else:
            return False

    def __bool__(self):
        return self.peek() is not None

#Is buggy, needs fixing
def inorderIterativeMorrisStack(root):
    current = root
    stack, res = MorrisStack(root), []
    while current or stack:
        if current:
            if current.left:
                stack.push(current)
                current = current.left
            else:
                res.append(current.data)
                print(current.data)
                if current.right and current.right not in stack:
                    current = current.right
                else:
                    current = None
        elif stack:
            current = stack.pop()
            res.append(current.data)
            print(current.data)
            if current.right and current.right not in stack:
                current = current.right
            else:
                current = None
    return res

#https://www.geeksforgeeks.org/inorder-tree-traversal-without-recursion/
#https://practice.geeksforgeeks.org/problems/inorder-traversal/1
def inorderIterative(root):
    current = root
    stack, res = [], []
    while current or stack:
        if current:
            stack.append(current)
            current = current.left
        elif stack:
            current = stack.pop()
            res.append(current.data)
            current = current.right
    return res

#https://practice.geeksforgeeks.org/problems/preorder-traversal/1
def preorderIterative(root):
    current = root
    stack, res = [], []
    while stack or current:
        if current:
            res.append(current.data)
            stack.append(current)
            current = current.left
        else:
            current = stack.pop().right
    return res

#https://practice.geeksforgeeks.org/problems/postorder-traversal/1
def postOrderIterative(root):
    current = root
    stack, res = [],[]
    while stack or current:
        if current:
            if current.right:
                stack.append(current.right)
            stack.append(current)
            current = current.left
        elif stack:
            current = stack.pop()
            if current.right and stack and current.right is stack[-1]:
                right = stack.pop()
                stack.append(current)
                current = right
            else:
                res.append(current.data)
                current = None
    return res

#https://practice.geeksforgeeks.org/problems/right-view-of-binary-tree/1
def rightViewOfBinaryTree(root):
    if not root:
        return []

    q, r = deque(), None
    q.append(root)
    res = []
    while q:
        q.append(None)
        while q[0]:
            r = q.popleft()
            if r.left:
                q.append(r.left)
            if r.right:
                q.append(r.right)
        q.popleft()
        res.append(r.data)
    return res

#https://practice.geeksforgeeks.org/problems/top-view-of-binary-tree/1
def topViewOfBinaryTree(root):
    if not root:
        return []

    res, zero = deque([root.data]), 0
    def topViewUtil(root, pos):
        if not root:
            return

        nonlocal zero
        i = zero + pos
        if i >= len(res):
            res.append(root.data)
        elif i < 0:
            res.appendleft(root.data)
            zero += 1

    q = deque()
    q.append((root, 0))
    while q:
        r, pos = q.popleft()
        topViewUtil(r, pos)
        if r.left:
            q.append((r.left, pos-1))
        if r.right:
            q.append((r.right, pos+1))

    return res

#https://practice.geeksforgeeks.org/problems/bottom-view-of-binary-tree/1
def bottomViewOfBinaryTree(root):
    if not root:
        return []

    res, zero = [root.data], 0
    def topViewUtil(root, pos):
        if not root:
            return

        nonlocal zero
        i = zero + pos
        if i >= len(res):
            res.append(root.data)
        elif i < 0:
            res.insert(0, root.data)
            zero += 1
        else:
            res[i] = root.data

    q = deque()
    q.append((root, 0))
    while q:
        r, pos = q.popleft()
        topViewUtil(r, pos)
        if r.left:
            q.append((r.left, pos-1))
        if r.right:
            q.append((r.right, pos+1))

    return res

#https://practice.geeksforgeeks.org/problems/diagonal-traversal-of-binary-tree/1
def diagnolViewOfTree(root):
    if not root:
        return []
    
    res, zero = [[]], 0
    def diagUtil(root, pos):
        if not root:
            return

        i = zero + pos
        if i >= len(res):
            res.append([root.data])
        elif i < 0:
            res.insert(0, [root.data])
        else:
            res[i].append(root.data)

        diagUtil(root.left, pos+1)
        diagUtil(root.right, pos)

    diagUtil(root,0)
    return [item for sublist in res for item in sublist]

#https://practice.geeksforgeeks.org/problems/transform-to-sum-tree/1
def toSumTree(root):
    def sumUtil(root):
        if not root:
            return 0

        l = sumUtil(root.left)
        r = sumUtil(root.right)
        d = root.data
        root.data = l+r
        return root.data + d

    sumUtil(root)

#https://practice.geeksforgeeks.org/problems/construct-tree-1/1
def buildtree(inorder, preorder, n):
    if n <= 0:
        return None
    elif n == 1:
        return Node(inorder[0])

    root = Node(preorder[0])
    mid = inorder.index(root.data)
    root.left = buildtree(inorder[:mid], preorder[1:mid+1], mid)
    root.right = buildtree(inorder[mid+1:], preorder[mid+1:], n-mid-1)
    return root

#https://www.geeksforgeeks.org/minimum-swap-required-convert-binary-tree-binary-search-tree
def swapsToMakeBST(root):
    def inOrder(root, a):
        if not root:
            return

        inOrder(root.left, a)
        a.add(root.data)
        inOrder(root.right, a)
        return a

    def swapsToSort(a):
        n = len(a)
        target = sorted(a.copy())
        hm = {}
        for i in range(n):
            hm[a[i]] = i

        swaps = 0
        for i in range(len(a)):
            while a[i] != target[i]:
                j = hm[target[i]]
                a[i], a[j] = a[j], a[i]
                hm[a[i]] = i
                hm[a[j]] = j
                swaps += 1
        return swaps

    return swapsToSort(inOrder(root, []))

#https://practice.geeksforgeeks.org/problems/sum-tree/1
def isSumTree(root):
    def util(r):
        if not r:
            return (True, 0)
        elif not r.left and not r.right:
            return (True, r.data)

        il, ls = util(r.left)
        ir, rs = util(r.right)

        if ls+rs == r.data:
            return (il and ir, ls+rs+r.data)
        else:
            return (False, ls+rs+r.data)

    return 1 if util(root)[0] else 0

#https://practice.geeksforgeeks.org/problems/leaf-at-same-level/1
def areLeavesOnSameLevel(root):
    leaves = set()
    def util(r, l):
        if not r:
            return
        elif len(leaves) > 1:
            return
        elif not r.left and not r.right:
            leaves.add(l)
            return
        else:
            util(r.left, l+1)
            util(r.right, l+1)

    util(root, 0)
    return len(leaves) == 1

#https://practice.geeksforgeeks.org/problems/duplicate-subtree-in-binary-tree/1
def findDuplicateSubtree(root):
    sumMap = {}

    def addToMap(s, node):
        if s not in sumMap:
            sumMap[s] = []
        sumMap[s].append(node)

    def sumUtil(root):
        if not root:
            return 0
        elif not root.left and not root.right:
            return root.data
        
        s = sumUtil(root.left) + sumUtil(root.right) + root.data
        addToMap(s, root)
        return s

    def areSimilar(r1, r2):
        if not r1 and not r2:
            return True
        elif not r1 or not r2:
            return False
        elif r1.data != r2.data:
            return False

        return areSimilar(r1.left, r2.left) and areSimilar(r1.right, r2.right)

    sumUtil(root)
    dupFound = False
    for key, nodes in sumMap.items():
        n = len(nodes)
        if n == 1:
            continue

        for i in range(n):
            for j in range(i+1, n):
                if areSimilar(nodes[i], nodes[j]):
                    dupFound = True
                    break
            if dupFound:
                break

    return 1 if dupFound else 0

#https://practice.geeksforgeeks.org/problems/check-if-subtree/1
#https://leetcode.com/problems/subtree-of-another-tree/submissions/
def isSubTree(s, T):
    def areSimilar(r1, r2):
        if not r1 and not r2:
            return True
        elif not r1 or not r2:
            return False
        elif r1.data != r2.data:
            return False
        else:
            return areSimilar(r1.left, r2.left) and areSimilar(r1.right, r2.right)
    
    def subtreeUtil(root):
        if not root:
            return False
        elif areSimilar(root, s):
            return True
        else:
            return subtreeUtil(root.left) or subtreeUtil(root.right)
    
    return subtreeUtil(T)

#https://practice.geeksforgeeks.org/problems/sum-of-the-longest-bloodline-of-a-tree/1
def sumOfLongRootToLeafPath(root):
    def sumLevelUtil(root, l):
        if not root:
            return (0, l-1)

        ls, ll = sumLevelUtil(root.left, l+1)
        rs, rl = sumLevelUtil(root.right, l+1)

        if ll > rl:
            return (ls+root.data, ll)
        elif rl > ll:
            return (rs+root.data, rl)
        else:
            return (max(rs, ls)+root.data, ll)

    return sumLevelUtil(root, 0)[0]

#https://leetcode.com/problems/most-frequent-subtree-sum/
def findFrequentTreeSum(root):
    map = {}

    def sumUtil(root):
        if not root:
            return 0

        s = sumUtil(root.left) + sumUtil(root.right) + root.data
        map[s] = map.get(s, 0) + 1
        return s

    sumUtil(root)
    ms, mf = [], 0
    for s, f in map.items():
        if f == mf:
            ms.append(s)
        elif f > mf:
            ms, mf = [s], f
    return ms

#https://www.geeksforgeeks.org/maximum-sum-nodes-binary-tree-no-two-adjacent/
def maxSumSubset(root):
    nodeIndex = {}
    count = 0
    def dfs(root):
        if not root:
            return

        nonlocal count
        nodeIndex[root] = count
        count += 1
        dfs(root.left)
        dfs(root.right)

    dfs(root)
    n = len(nodeIndex)
    maxSelected, maxUnselected = [None]*n, [None]*n
    
    def maxSubset(root, choice):
        if not root:
            return 0

        if choice and maxSelected[nodeIndex[root]] is not None:
            return maxSelected[nodeIndex[root]] 
        elif not choice and maxUnselected[nodeIndex[root]] is not None:
            return maxUnselected[nodeIndex[root]] 

        s = maxSubset(root.left, True) + maxSubset(root.right, True)
        maxUnselected[nodeIndex[root]] = s
        if choice:
            s2 = maxSubset(root.left, False) + maxSubset(root.right, False) + root.data
            s = max(s, s2)
            maxSelected[nodeIndex[root]] = s
        return s

    return maxSubset(root, True)

#https://leetcode.com/problems/path-sum-iii/
def kPathSum(root, k):
    prefixPaths = defaultdict(int)
    prefixPaths[0] = 1
    
    def dfsUtil(root, prefix):
        if not root:
            return 0

        s = prefix + root.val
        count = prefixPaths[s-k]
        prefixPaths[s] += 1

        count += dfsUtil(root.left, s)
        count += dfsUtil(root.right, s)
        prefixPaths[s] -= 1

        return count

    return dfsUtil(root, 0)

def findPathTo(root, n, path):
    if not root:
        return []

    path.append(root)
    if root.data == n:
        return path

    l = findPathTo(root.left, n, path)
    if len(l) > 0:
        return l

    r = findPathTo(root.right, n, path)
    if len(r) > 0:
        return r

    path.pop()
    return []

#https://practice.geeksforgeeks.org/problems/lowest-common-ancestor-in-a-binary-tree/1
def findlca(root, n1, n2):
    p1 = findPathTo(root, n1, [])
    p2 = findPathTo(root, n2, [])
    match = None
    for i in range(min(len(p1), len(p2))):
        if p1[i] is p2[i]:
            match = p1[i]
        else:
            break
    return match

#https://practice.geeksforgeeks.org/problems/min-distance-between-two-given-nodes-of-a-binary-tree/1#
def findDist(root,a,b):
    p1 = findPathTo(root, a, [])
    p2 = findPathTo(root, b, [])
    match = None
    for i in range(min(len(p1), len(p2))):
        if p1[i] is p2[i]:
            match = i + 1
        else:
            break

    return len(p1) + len(p2) - 2*match

#https://www.geeksforgeeks.org/kth-ancestor-node-binary-tree-set-2/
def kthAncestorOfNode(root, node, k):
    parent = {}
    def dfsUtil(root):
        if not root:
            return

        if root.data == node:
            return

        if root.left:
            parent[root.left.data] = root.data
            dfsUtil(root.left)

        if root.right:
            parent[root.right.data] = root.data
            dfsUtil(root.right)

    dfsUtil(root)
    if node not in parent:
        return -1

    ans = node
    for i in range(k):
        ans = parent.get(ans, -1)
    return ans

    path = findPathTo(root, node, [])
    if k+1 > len(path):
        return -1
    else:
        return path[-(k+1)]

#https://practice.geeksforgeeks.org/problems/duplicate-subtrees/1
def findDuplicateSubtrees(root):
    hashes, dupes = defaultdict(int), []
    def hashTree(root):
        if not root:
            return ""

        l = hashTree(root.left)
        r = hashTree(root.right)
        s = "(" + l + str(root.data) + r + ")"
        
        if hashes[s] == 1:
            dupes.append(root)
        hashes[s] += 1
        return s

    hashTree(root)
    return dupes

#https://practice.geeksforgeeks.org/problems/check-if-tree-is-isomorphic/1
def areIsomorphic(r1, r2):
    if r1 is None and r2 is None:
        return True
    elif not r1 or not r2:
        return False
    elif r1.data != r2.data:
        return False
    else:
        if areIsomorphic(r1.left, r2.left) and areIsomorphic(r1.right, r2.right):
            return True
        if areIsomorphic(r1.left, r2.right) and areIsomorphic(r1.right, r2.left):
            return True
        return False

#https://leetcode.com/problems/merge-two-binary-trees/
def mergeTrees(root1, root2):
    if not root1 and not root2:
        return None
    elif not root1 or not root2:
        r = root1 if root1 else root2
        root = TreeNode(r.val)
        root.left = mergeTrees(r.left, None)
        root.right = mergeTrees(r.right, None)
        return root
    else:
        root = TreeNode(root1.val + root2.val)
        root.left = mergeTrees(root1.left, root2.left)
        root.right = mergeTrees(root1.right, root2.right)
        return root

def replaceNodeWithSumOfRLLeafs(root):
    if not root:
        return
    elif not root.right and not root.left:
        return root.data, root.data
    elif not root.right:
        ll, _ = replaceNodeWithSumOfRLLeafs(root.left)
        rr = root.data
        root.data = ll
        return ll+rr, rr
    elif not root.left:
        _, rr = replaceNodeWithSumOfRLLeafs(root.right)
        ll = root.data
        root.data = rr
        return ll, rr+ll
    else:
        ll, lr = replaceNodeWithSumOfRLLeafs(root.left)
        rl, rr = replaceNodeWithSumOfRLLeafs(root.right)
        d = root.data
        root.data = ll + rr
        return ll+d, rr+d
