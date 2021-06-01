
import heapq
from collections import deque

class RunningMedian:

	def __init__(self):
		self.min_heap = []
		self.max_heap = []

	def _balanceHeaps(self):
		if abs(len(self.max_heap) - len(self.min_heap)) <= 1:
			return

		if len(self.max_heap) > len(self.min_heap):
			self.moveToMinHeap()
		else:
			self.moveToMaxHeap()

	def _moveToMinHeap(self):
		while len(self.max_heap) > len(self.min_heap):
			x = heapq._heappop_max(self.max_heap)
			heapq.heappush(self.min_heap, x)

	def _moveToMaxHeap(self):
		while len(self.min_heap) > len(self.max_heap):
			x = heapq.heappop(self.min_heap)
			self._heappush_max(self.max_heap, x)
        
	def getMedian(self):
		self.balanceHeaps()
		if len(self.min_heap) == len(self.max_heap):
			return (self.min_heap[0] + self.max_heap[0])//2
		elif len(self.min_heap) > len(self.max_heap):
			return self.min_heap[0]
			rerun
		else:
			return self.max_heap[0]

	def insert(self,x):
		if len(self.min_heap) == 0 or len(self.max_heap) == 0:
			heapq.heappush(self.min_heap, x)
			return

		if x >= self.min_heap[0]:
			heapq.heappush(self.min_heap, x)
		elif x <= self.max_heap[0]:
			self._heappush_max(self.max_heap, x)
		else:
			if len(self.max_heap) > len(self.min_heap):
				heapq.heappush(self.min_heap, x)
			else:
				self._heappush_max(self.max_heap, x)

	def _heappush_max(self, heap, item):
		heap.append(item)
		heapq._siftdown_max(heap, 0, len(heap)-1)

def rearrangeString(s, d):
	from collections import Counter

	c = Counter(s)
	maxKey = (0, None)
	for key, value in c.items():
		if value > maxKey[0]:
			maxKey = (value, key)
	
	n = len(s)
	spaceNeeded = (maxKey[0]-1)*d + 1
	if spaceNeeded > n:
		return ""

	pq = []
	for key, value in c.items():
		pq.append((value, key))
	heapq._heapify_max(pq)

	key = None
	value = 0
	l = ["#" for i in range(n)]
	for indexMultiple in range(d):
		for i in range(n//d + 1):
			index = (i*d) + indexMultiple

			if index >= n:
				continue

			if value == 0:
				maxKey = heapq._heappop_max(pq)
				key = maxKey[1]
				value = maxKey[0]

			l[index] = key
			value -= 1
	return "".join(l)

def kthLargest(k, arr, n):
	pq = arr[:k]
	heapq.heapify(pq)

	res = [-1 for i in range(k-1)]
	for a in arr[k-1:]:
		res.append(pq[0])
		if a >= pq[0]:
			heapq.heapreplace(pq,a)
	
	return res

def mergeKLists(arr,K):
	heap = []
	map = {}
	for l in range(len(arr)):
		if arr[l]:
			heap.append((arr[l].data, l))
			map[l] = arr[l]
	heapq.heapify(heap)
	
	head = None
	c = head
	node = None
	while len(heap) > 0:
		index = heapq.heappop(heap)[1]
		node = map[index]
		if head:
			c.next = Node(node.data)
			c = c.next
		else:
			head = Node(node.data)
			c = head

		node = node.next
		map[index] = node
		if node:
			heapq.heappush(heap,(node.data, index))
		
	return head
