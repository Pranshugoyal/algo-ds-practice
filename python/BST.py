
class Node:
    def __init__(self, data):
        self.data = data
        self.left = None
        self.right = None

################################################################################
# ------------------------ Binary Search Tree Love's Sheet ------------------- #
################################################################################

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
def bstFromPreorder(arr):
    def decodeUtil(i, lb, rb):
        if i >= len(arr) or arr[i] >= rb or arr[i] <= lb:
            return None, i-1

        root = Node(arr[i])
        root.left, i = decodeUtil(i+1, lb, root.data)
        root.right, i = decodeUtil(i+1, root.data, rb)
        return root, i
    
    return decodeUtil(0, 0, 100)[0]

#https://practice.geeksforgeeks.org/problems/binary-tree-to-bst/1#
def btToBst(root):
    def inorderRead(root, res):
        if not root:
            return res
        inorderRead(root.left, res)
        res.append(root.data)
        return inorderRead(root.right, res)

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
