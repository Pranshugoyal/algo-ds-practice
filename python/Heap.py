
import heapq
from collections import deque

class Heap:

    def __init__(self, values, min=True):
        self.values = values.copy()
        self.min = min
        self._map = {}

        n = len(values)
        for i in range(n):
            v = values[i]
            self.addToMap(v,i)

        for i in range(n//2, -1, -1):
            self.heapify(i)

    def __len__(self):
        return len(self.values)

    def __str__(self):
        return self.values.__str__()
        #return self.values.__str__() + " " + self._map.__str__()

    def __contains__(self, key):
        return key in self._map

    def peek(self):
        return self.values[0]

    def pop(self):
        return self._popIndex(0)

    def push(self, v):
        i = len(self.values)
        self.addToMap(v, i)
        self.values.append(v)
        self.moveUp(i)

    def deleteValue(self, v):
        if v not in self._map:
            return False

        i = next(iter(self._map[v]), None)
        self._popIndex(i)

    def replace(self, old, new):
        i = next(iter(self._map[old]), None)
        self.values[i] = new

        self.removeFromMap(old, i)
        self.addToMap(new, i)

        self.moveUp(i)
        self.heapify(i)

    def heapify(self, i):
        def siftDown(i) -> int:
            child = self.childrenViolation(i)
            if child is None:
                return None
            else:
                self.swap(i, child)
                return child

        while i is not None:
            i = siftDown(i)

    def moveUp(self, i):
        while self.parentViolation(i):
            self.swap(self.parentIndex(i), i)
            i = self.parentIndex(i)

    def parentViolation(self, i):
        if i == 0:
            return False
        elif self.min:
            return self.values[i] < self.values[self.parentIndex(i)]
        else:
            return self.values[i] > self.values[self.parentIndex(i)]

    def childrenViolation(self, i):
        children = self.childrenIndexes(i)
        if not children:
            return None

        child = None
        if self.min:
            child = min(children, key=lambda x: self.values[x])
        else:
            child = max(children, key=lambda x: self.values[x])

        if self.parentViolation(child):
            return child
        else:
            return None
    
    def addToMap(self, v, i):
        if v in self._map:
            self._map[v].add(i)
        else:
            self._map[v] = set([i])

    def removeFromMap(self, v,i):
        self._map[v].remove(i)
        if len(self._map[v]) == 0:
            self._map.pop(v)

    def swap(self, i ,j):
        vi, vj = self.values[i], self.values[j]
        self.values[i], self.values[j] = vj, vi
        self._map[vi].remove(i)
        self._map[vj].remove(j)
        self._map[vi].add(j)
        self._map[vj].add(i)

    def _popIndex(self, i):
        lastIndex = len(self.values)-1
        last = self.values.pop()
        self.removeFromMap(last, lastIndex)

        if i == lastIndex:
            return last

        v = self.values[i]
        self.removeFromMap(v,i)

        self.addToMap(last, i)
        self.values[i] = last

        self.moveUp(i)
        self.heapify(i)
        return v

    def parentIndex(self, i):
        if i > 0:
            return (i-1)//2
        else:
            return 0

    def childrenIndexes(self, i):
        left, right = 2*i + 1, 2*i + 2
        res = []
        if left < len(self.values):
            res.append(left)
        if right < len(self.values):
            res.append(right)
        return res

class StreamPartitioner:

    def __init__(self, nums, k):
        self.k = k
        self.lh = Heap([], min=False)
        self.rh = Heap(nums)
        self.lSum, self.rSum = 0, sum(nums)
        self.balance()

    def __str__(self):
        return self.lh.__str__() + " " + self.rh.__str__()

    def median(self):
        if len(self.lh) == len(self.rh):
            return (self.rh.peek() + self.lh.peek())/2
        elif len(self.rh) - len(self.lh) == 1:
            return self.rh.peek()
        elif len(self.lh) - len(self.rh) == 1:
            return self.lh.peek()
        else:
            raise Exception("Median not at boundry")

    def delete(self, v):
        if v in self.lh:
            self.lh.deleteValue(v)
            self.lSum -= v
        elif v in self.rh:
            self.rh.deleteValue(v)
            self.rSum -= v
        self.balance()

    def insert(self, v):
        if v >= self.rh.peek():
            self.rh.push(v)
            self.rSum += v
        else:
            self.lh.push(v)
            self.lSum += v
        self.balance()

    def balance(self):
        n = None
        while len(self.lh) < self.k:
            n = self.rh.pop()
            self.lh.push(n)
            self.rSum -= n
            self.lSum += n
        while len(self.lh) > self.k:
            n = self.lh.pop()
            self.rh.push(n)
            self.rSum += n
            self.lSum -= n

        assert len(self.lh) == self.k

    def replace(self, old, new):
        if old in self.rh:
            if new >= self.rh.peek():
                self.rh.replace(old, new)
                self.rSum += new-old
            else:
                self.rh.deleteValue(old)
                self.lh.push(new)
                self.rSum -= old
                self.lSum += new
        else:
            if new <= self.lh.peek():
                self.lh.replace(old, new)
                self.lSum += new-old
            else:
                self.lh.deleteValue(old)
                self.rh.push(new)
                self.lSum -= old
                self.rSum += new
        self.balance()

class MKAverage:

    def __init__(self, m: int, k: int):
        self.q = deque()
        self.m = m
        self.k = k
        self.ls, self.rs = None, None

    def __str__(self):
        if self.ls is not None and self.rs is not None:
            return self.q.__str__() + "\n" + "LS: " + self.ls.__str__() + "\nRS: " + self.rs.__str__()
        else:
            return self.q.__str__()

    def initiateStreams(self):
        l = list(self.q)
        self.ls = StreamPartitioner(l, self.k)
        self.rs = StreamPartitioner(l, self.m-self.k)

    def addElement(self, num: int) -> None:
        if len(self.q) == self.m:
            old = self.q.popleft()
            self.q.append(num)
            self.ls.replace(old, num)
            self.rs.replace(old, num)
        elif len(self.q) == self.m-1:
            self.q.append(num)
            self.initiateStreams()
        else:
            self.q.append(num)

    def calculateMKAverage(self) -> int:
        if len(self.q) < self.m:
            return -1

        s = self.ls.rSum - self.rs.rSum
        return s//(self.m-2*self.k)

def slidingWindowMedian(nums, k):
    slider = StreamPartitioner(nums[:k], k-k//2)
    medians = [slider.median()]

    for i, new in enumerate(nums[k:]):
        old = nums[i]
        slider.replace(old, new)
        medians.append(slider.median())
    return medians

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

#https://leetcode.com/problems/kth-smallest-element-in-a-sorted-matrix/
def kthSmallestMatrix(matrix, k):
    n = len(matrix)
    h = []
    for r in range(n):
        h.append((matrix[r][0], r, 0))
    heapq.heapify(h)

    r, c = None, None
    for _ in range(k-1):
        r, c = h[0][1], h[0][2]
        if c == n-1:
            heapq.heappop(h)
        else:
            heapq.heapreplace(h, (matrix[r][c+1], r, c+1))

    return h[0][0]

