
class Node:
    def __init__(self, data):
        self.data = data
        self.left = None
        self.right = None

################################################################################
# ------------------------ Binary Search Tree Love's Sheet ------------------- #
################################################################################

def inorderRead(root, res):
    if not root:
        return res
    inorderRead(root.left, res)
    res.append(root.data)
    return inorderRead(root.right, res)

def preOrderTraverse(root):
    res = []
    def dfsPreutil(root):
        if not root:
            return

        res.append(root.data)
        dfsPreutil(root.left)
        dfsPreutil(root.right)

    dfsPreutil(root)
    return res

def printLevelOrderTree(root):
    from collections import deque
    q = deque([root])
    while q:
        q.append(None)
        l = []
        while q[0] is not None:
            n = q.popleft()
            l.append(n.data)
            if n.left:
                q.append(n.left)
            if n.right:
                q.append(n.right)
        print(l)
        l = []
        q.popleft()

#https://practice.geeksforgeeks.org/problems/check-for-bst/1
def isBST(root):
    def dfsBst(root):
        if not root:
            return (None, None, True)
        
        left = dfsBst(root.left)
        right = dfsBst(root.right)

        lb = min(root.data, left[0]) if left[0] is not None else root.data
        rb = max(root.data, right[1]) if right[1] is not None else root.data

        bst = left[2] and right[2]
        if left[0] is not None and left[1] >= root.data:
            bst = False
        if right[1] is not None and right[0] < root.data:
            bst = False

        return (lb, rb, bst)

    return dfsBst(root)[2]

#https://practice.geeksforgeeks.org/problems/delete-a-node-from-bst/1
def deleteNodeBST(root, key):
    def firstInorderNode(root):
        if not root:
            return None
        elif not root.left:
            return root
        else:
            return firstInorderNode(root.left)

    if not root:
        return None
    elif key < root.data:
        root.left = deleteNodeBST(root.left, key)
        return root
    elif key > root.data:
        root.right = deleteNodeBST(root.right, key)
        return root
    
    #key found
    if not root.left or not root.right:
        #one or both children absent
        return root.left if root.left else root.right
    else:
        #both children present
        root.data = firstInorderNode(root.right).data
        root.right = deleteNodeBST(root.right, root.data)
        return root

#https://practice.geeksforgeeks.org/problems/largest-bst/1#
def largestBSTSubtreeSize(root):
    def bstUtil(root):
        if not root:
            return (100001, 0, True, 0)

        left = bstUtil(root.left)
        right = bstUtil(root.right)

        res = ( min(left[0], right[0], root.data),
                max(left[1], right[1], root.data),
                False,
                max(left[3], right[3]))

        if not left[2] or not right[2] or left[1] >= root.data or right[0] <= root.data:
            return res
        else:
            return (res[0], res[1], True, left[3]+right[3]+1)

    return bstUtil(root)[3]

#https://www.geeksforgeeks.org/populate-inorder-successor-for-all-nodes/
def updateInorderSuccession(root, parent=None):
    if not root:
        return None

    if root.right:
        root.next = updateInorderSuccession(root.right, parent)
    else:
        root.next = parent

    if root.left:
        return updateInorderSuccession(root.left, root)
    else:
        return root

#https://www.geeksforgeeks.org/construct-bst-from-given-preorder-traversa/
#https://practice.geeksforgeeks.org/problems/preorder-to-postorder4423/1#
def bstFromPreorder(arr):
    def decodeUtil(i, lb, rb):
        if i >= len(arr) or arr[i] >= rb or arr[i] <= lb:
            return None, i-1

        root = Node(arr[i])
        root.left, i = decodeUtil(i+1, lb, root.data)
        root.right, i = decodeUtil(i+1, root.data, rb)
        return root, i
    
    return decodeUtil(0, 0, 100)[0]

#https://www.geeksforgeeks.org/construct-bst-from-given-preorder-traversal-set-2/
def bstFromPreorderIterative(arr):
    root, stack, temp = Node(arr[0]), [], None
    stack.append(root)
    for n in arr[1:]:
        temp = None
        while stack and n > stack[-1].data:
            temp = stack.pop()

        if temp:
            temp.right = Node(n)
            stack.append(temp.right)
        else:
            temp = stack[-1]
            temp.left = Node(n)
            stack.append(temp.left)
    return root

#https://practice.geeksforgeeks.org/problems/binary-tree-to-bst/1#
def btToBst(root):
    def inorderWrite(root, res):
        if not root:
            return
        inorderWrite(root.left, res)
        root.data = res.pop()
        inorderWrite(root.right, res)

    res = inorderRead(root, [])
    res.sort(reverse=True)
    inorderWrite(root, res)
    return root

#https://www.geeksforgeeks.org/sorted-array-to-balanced-bst/
#https://leetcode.com/problems/convert-sorted-array-to-binary-search-tree/submissions/
def sortedArrayToBST(arr):
    if not arr:
        return None

    mid = len(arr)//2
    root = Node(arr[mid])
    root.left = sortedArrayToBST(arr[:mid])
    root.right = sortedArrayToBST(arr[mid+1:])
    return root

#https://www.geeksforgeeks.org/convert-normal-bst-balanced-bst/
def balanceBST(root):
    return sortedArrayToBST(inorderRead(root, []))

#https://www.geeksforgeeks.org/merge-two-balanced-binary-search-trees/
def mergeBsts(r1, r2):
    def mergeSortedArrays(a1, a2):
        res = []
        i,j = 0,0
        while i < len(a1) and j < len(a2):
            if a1[i] < a2[j]:
                res.append(a1[i])
                i += 1
            else:
                res.append(a2[j])
                j += 1

        while i < len(a1):
            res.append(a1[i])
            i += 1
        while j < len(a2):
            res.append(a2[j])
            j += 1
        return res

    t1 = inorderRead(r1, [])
    t2 = inorderRead(r2, [])
    t = mergeSortedArrays(t1, t2)
    return sortedArrayToBST(t)

#https://www.geeksforgeeks.org/in-place-conversion-of-sorted-dll-to-balanced-bst/
def bstFromSortedLinkedListM1(head):
    #nLogn method as list is read in every call to find mid
    def getLLMid(head):
       lag, slow, fast = None, head, head
       while fast and fast.next:
           lag = slow
           slow = slow.next
           fast = fast.next.next 
       return slow, lag

    def dfsUtil(head):
        mid, tail = getLLMid(head)
        if not mid:
            return None
        root = Node(mid.data)
        root.right = dfsUtil(mid.next)
        if tail:
            tail.next = None
            root.left = dfsUtil(head)
            tail.next = mid
        return root

    return dfsUtil(head)

#https://www.geeksforgeeks.org/sorted-linked-list-to-balanced-bst/
#https://leetcode.com/problems/convert-sorted-list-to-binary-search-tree/submissions/
def bstFromSortedLinkedListM2(head):
    def getLen(head):
        count, p = 0, head
        while p:
            count += 1
            p = p.next
        return count

    def createBST(head, n):
        if n <= 0 or not head:
            return None, head
        elif n == 1:
            return Node(head.data), head

        lr, lt = createBST(head, n//2)
        root = Node(lt.next.data)
        root.left = lr
        rc = n - (n//2) - 1
        if lt.next.next and rc > 0:
            root.right, tail = createBST(lt.next.next, rc)
            return root, tail
        else:
            return root, lt.next

    return createBST(head, getLen(head))[0]

#https://www.geeksforgeeks.org/in-place-conversion-of-sorted-dll-to-balanced-bst/
def bstFromDLLInPlace(head):
    def getLen(head):
        count, p = 0, head
        while p:
            count += 1
            p = p.next
        return count

    def convertToBST(head, n):
        if n <= 0 or not head:
            return None, None
        elif n == 1:
            head.previous = None
            return head, head

        lr, lt = convertToBST(head, n//2)

        root = lt.next
        root.previous = lr
        lt.next = None

        rc = n - (n//2) - 1
        if not root.next or rc <= 0:
            return root, root

        root.next, rt = convertToBST(root.next, rc)
        return root, rt

    return convertToBST(head, getLen(head))[0]

#https://practice.geeksforgeeks.org/problems/kth-largest-element-in-bst/1
def kthLargestInBst(root, k):
    res = None
    def dfsUtil(root, k):
        if not root:
            return k

        k = dfsUtil(root.right, k)
        if k > 1:
            return dfsUtil(root.left, k-1)
        elif k < 1:
            return k-1

        nonlocal res
        res = root.data
        return 0

    dfsUtil(root, k)
    return res

#https://practice.geeksforgeeks.org/problems/find-k-th-smallest-element-in-bst/1
def kthSmallestInBst(root, k):
    count, node = 0, None
    def dfsInOrder(root):
        if not root:
            return

        dfsInOrder(root.left)
        nonlocal count, node
        count += 1
        if count == k:
            node = root
            return
        dfsInOrder(root.right)

    dfsInOrder(root)
    return node.data

#https://practice.geeksforgeeks.org/problems/brothers-from-different-root/1
def countPairs(root1, root2, x):
    def makeSet(root, res):
        if not root:
            return res
        res.add(root.data)
        makeSet(root.left, res)
        return makeSet(root.right, res)

    def findPairs(root, hmap, pairs=0):
        if not root:
            return pairs

        if (x-root.data) in hmap:
            pairs += 1
        pairs = findPairs(root.left, hmap, pairs)
        return findPairs(root.right, hmap, pairs)

    return findPairs(root1, makeSet(root2, set()))

#https://www.geeksforgeeks.org/find-median-bst-time-o1-space/
def medianBst(root):
    def inorderTravel(root):
        current, stack = root, []
        while stack and current:
            if current:
                stack.append(current)
                current = current.left
            elif stack:
                current = stack.pop()
                yield current
                current = current.right

    count = 0
    for _ in inorderTravel(root):
        count += 1

    target = set([count//2])
    if count%2 == 0:
        target = target.add(count//2 - 1)

    count, median = 0, None
    for node in inorderTravel(root):
        count += 1
        if count in target:
            if median is None:
                median = node.data
            else:
                median = (median+node.data)/2
    return median

#https://practice.geeksforgeeks.org/problems/count-bst-nodes-that-lie-in-a-given-range/1
def getNodeCountInRange(root, low, high):
    count = 0
    def dfs(root):
        if not root:
            return

        if root.data <= high and root.data >= low:
            nonlocal count
            count += 1
        if root.data >= low:
            dfs(root.left)
        if root.data <= high:
            dfs(root.right)

    dfs(root)
    return count

#https://www.geeksforgeeks.org/flatten-bst-to-sorted-list-increasing-order/
#https://leetcode.com/problems/flatten-binary-tree-to-linked-list/submissions/
def flattenBst(root):
    def util(root):
        if not root:
            return None, None

        lh, lt = util(root.left)
        rh, rt = util(root.right)

        root.left = None
        head, tail = root, root
        if lt:
            lt.right = root
            head = lh
        if rh:
            root.right = rh
            tail = rt
        return head, tail

    return util(root)[0]

#https://www.geeksforgeeks.org/replace-every-element-with-the-least-greater-element-on-its-right/
def leastGreaterToRight(arr):
    def insertAndGetSuccessor(root, n, parent=None):
        if not root:
            return Node(n), parent
        elif root.data > n:
            root.left, succ = insertAndGetSuccessor(root.left, n, root)
            return root, succ
        elif root.data <= n:
            root.right, succ = insertAndGetSuccessor(root.right, n, parent)
            return root, succ

    root, res = None, []
    for n in reversed(arr):
        root, succ = insertAndGetSuccessor(root, n)
        res.append(succ.data if succ else -1)
    res.reverse()
    return res

#https://www.geeksforgeeks.org/check-if-a-given-array-can-represent-preorder-traversal-of-binary-search-tree/
def validatePreorderTraversal(arr):
    root, s = None, []
    for val in arr:
        if root and val < root:
            return False
        while s and s[-1] < val:
            root = s.pop()
        s.append(val)
    return True

#https://practice.geeksforgeeks.org/problems/check-whether-bst-contains-dead-end/1
def hasDeadEnds(root):
    def deUtil(root, lo, hi):
        if not root:
            return False
        elif lo == hi:
            return True
        else:
            return deUtil(root.left, lo, root.data-1) or deUtil(root.right, root.data+1, hi)
    return deUtil(root, 0, None)
