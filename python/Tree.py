
################################################################################
# --------------------- Tree must do list from GeeksforGeeks ----------------- #
################################################################################

from collections import deque

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

