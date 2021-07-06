#
# Stack & Queue.py

################################################################################
# ----------- Standard Stack problems from Aditya Verma Playlist ------------- #
################################################################################

# https://www.youtube.com/watch?v=J2X70jj_I1o&list=PL_z_8CaSLPWdeOezg68SKkeLN4-T_jNHd

#Nearest greater to left
def ngl(list):
    stack, res = [], []
    for i in list:
        while stack and stack[-1] <= i: stack.pop()
        res.append(stack[-1] if stack else None)
        stack.append(i) 
    return result
    
#Nearest greater to right
def ngr(list):
    stack = []
    result = [None] * len(list)

    for i in range(len(list)-1,-1,-1):
        while len(stack) > 0 and stack[-1] <= list[i]:
            stack.pop()
        
        if len(stack) == 0:
            result[i] = None
        else:
            result[i] = stack[-1]

        stack.append(list[i])   

    return result

#Nearest smaller to right
def nsr(list):
    stack = []
    result = [None] * len(list)

    for i in range(len(list)-1,-1,-1):
        while len(stack) > 0 and stack[-1] >= list[i]:
            stack.pop()
        
        if len(stack) == 0:
            result[i] = None
        else:
            result[i] = stack[-1]

        stack.append(list[i])   

    return result

#Nearest smaller to left
def nsl(list):
    stack = []
    result = [None] * len(list)

    for i in range(0, len(list)):
        while len(stack) > 0 and stack[-1] >= list[i]:
            stack.pop()
        
        if len(stack) == 0:
            result[i] = None
        else:
            result[i] = stack[-1]

        stack.append(list[i])   

    return result

# Stock Span Problem
def stockSpan(list):
    stack = []
    result = [None]*len(list)

    for i in range(0, len(list)):
        while len(stack) > 0 and list[stack[-1]] <= list[i]:
            stack.pop()

        if len(stack) == 0:
            result[i] = i + 1
        else:
            result[i] = i - stack[-1]

        stack.append(i)

    return result

def maxAreaHistogram(list):
    stack = []
    nsl = [-1] * len(list)
    for i in range(0, len(list)):
        while len(stack) > 0 and list[stack[-1]] >= list[i]:
            stack.pop()
        if len(stack) > 0:
            nsl[i] = stack[-1]
        stack.append(i)

    stack = []
    nsr = [len(list)] * len(list)   
    for i in range(len(list)-1,-1,-1):
        while len(stack) > 0 and list[stack[-1]] >= list[i]:
            stack.pop()
        if len(stack) > 0:
            nsr[i] = stack[-1]
        stack.append(i)

    result = []
    for i in range(0, len(list)):
        area = (nsr[i] - nsl[i] - 1) * list[i]
        result.append(area) 

    return max(result)  

def maxAreaBinaryMatrix(M):
    reducedMatrix = [M[0]]
    for row in M[1:]:
        currentRow = []
        for i in range(0,len(row)):
            if row[i] == 1:
                currentRow.append(reducedMatrix[-1][i] + 1)
            else:
                currentRow.append(0)
        reducedMatrix.append(currentRow)

    maxAreas = []
    for row in reducedMatrix:
        maxAreas.append(maxAreaHistogram(row))
    return max(maxAreas)

def rainWaterTapping(list):
    rml = [list[0]]
    for i in list[1:]:
        if i > rml[-1]:
            rml.append(i)
        else:
            rml.append(rml[-1])
    
    rmr = [list[-1]]
    for i in reversed(list):
        if i > rmr[-1]:
            rmr.append(i)
        else:
            rmr.append(rmr[-1])
    rmr.reverse()

    result = []
    for i in range(0, len(list)):
        height = min(rml[i], rmr[i]) - list[i]
        result.append(height)

    return sum(result)

def rainWaterTappingDP(list):
    leftMax = list[0]
    rightMax = list[-1]

    left = 0
    right = len(list) -1
    waterTapped = 0

    while left < right:
        if leftMax < rightMax:
            if list[left] < leftMax:
                waterTapped += leftMax - list[left]
            else:
                leftMax = list[left]
            left += 1
        else:
            if list[right] < rightMax:
                waterTapped += rightMax - list[right]
            else:
                rightMax = list[right]
            right -= 1
    
    return waterTapped

class TwoStack:
    
    # init method or constructor   
    def __init__(self, size):  
        self.storage = [None] * size
        self.leftIndex = 0
        self.rightIndex = size - 1

    def size(self, stack):
        if stack == 0:
            return self.leftIndex
        else:
            return len(self.storage) - self.rightIndex - 1

    def isFull(self, stack):
        return self.size(0) + self.size(1) == len(self.storage)

    def isEmpty(self, stack):
        if stack == 0:
            return self.leftIndex == 0
        else:
            return self.rightIndex == len(self.storage) - 1

    def push(self, data, stack):
        if self.isFull(stack):
            raise Exception("Stack full")

        if stack == 0:
            self.storage[self.leftIndex] = data
            self.leftIndex += 1
        else:
            self.storage[self.rightIndex] = data
            self.rightIndex -= 1

    def pop(self, stack):
        if self.isEmpty(stack):
            raise Exception("Stack Empty")

        value = None
        if stack == 0:
            value = self.storage[self.leftIndex]
            self.storage[self.leftIndex] = None
            self.leftIndex -= 1
        else:
            value = self.storage[self.righttIndex]
            self.storage[self.righttIndex] = None
            self.righttIndex += 1
        return value

    def peek(self, stack):
        if self.isEmpty(stack):
            raise Exception("Stack Empty")

        if stack == 0:
            return self.storage[self.leftIndex]
        else:
            return self.storage[self.rightIndex]

################################################################################
# ---------------- Must do list Stack & Queue GeeksForGeeks ------------------ #
################################################################################

from collections import deque

class Node:

    def __init__(self,data):
        self.data = data
        self.next = None
        self.previous = None

def nextLargerElement(arr,n):
    stack = []
    result = []
    
    for i in reversed(arr):
        print("Loop start:", i, stack)
        while len(stack) != 0 and stack[-1] <= i:
            stack.pop()
        
        print("Stack set:", i, stack)
        if len(stack) == 0:
            result.append(-1)
        else:
            result.append(stack[-1])
        
        stack.append(i)
        print("Loop exit:", i, stack)
        
    result.reverse()
    return result

class MinStack:
    def __init__(self):
        self.s=[]
        self.min = None

    def push(self,x):
        if len(self.s) == 0:
            self.s.append(x)
            self.min = x
            return

        if x < self.min:
            t = 2*x - self.min
            self.min = x
            self.s.append(t)
        else:
            self.s.append(x)

    def pop(self):
        if len(self.s) == 0:
            return -1
        elif len(self.s) == 1:
            self.min = None
            return self.s.pop()

        t = self.s.pop()
        if t < self.min:
            x = self.min
            self.min = 2*x - t
            return x
        else:
            return t
        
    def getMin(self):
        return self.min if self.min else -1

class LRUCache:
      
    #Constructor for initializing the cache capacity with the given value.  
    def __init__(self,cap):
        self.capacity = cap
        self.cache = {}
        self.head = None
        self.tail = None
        
    #Function to return value corresponding to the key.
    def get(self, key):
        if key not in self.cache:
            return -1
        
        self.moveKeyToTop(key)
        return self.cache[key][0]
        
    #Function for storing key-value pair.   
    def set(self, key, value):
        if key in self.cache:
            node = self.cache[key][1]
            self.cache[key] = (value, node)
            self.moveKeyToTop(key)
        else:
            self.addKeyToCache(key, value)
    
    def moveKeyToTop(self, key):
        node = self.cache[key][1]
        if node is self.tail:
            return

        p = node.previous
        n = node.next
        n.previous = p
        if p:
            p.next = n
        else:
            self.head = n

        self.tail.next = node
        node.previous = self.tail
        node.next = None
        self.tail = node
    
    def addKeyToCache(self, key, value):
        if len(self.cache) == self.capacity:
            lruKey = self.head.data
            self.cache.pop(lruKey)
            if self.head.next:
                self.head = self.head.next
                self.head.previous = None
            else:
                self.head = None
                self.tail = None

        self.cache[key] = (value, self.addNodeForKey(key))
    
    def addNodeForKey(self, key):
        node = Node(key)
        if self.tail is None:
            self.head = node
            self.tail = node
        else:
            self.tail.next = node
            node.previous = self.tail
            self.tail = node
        return node

def max_of_subarrays(arr,n,k):
    result = []
    q = deque()
    i,j = 0,0
    while j < n:
        while len(q) > 0 and q[0] < arr[j]:
            q.popleft()
        if len(q) > 0:
            while q[-1] < arr[j]:
                q.pop()
        q.append(arr[j])

        if j - i + 1 == k:
            result.append(q[0])
            if q[0] == arr[i]:
                q.popleft()
            i += 1
        j += 1

    return result

#https://practice.geeksforgeeks.org/problems/rotten-oranges2536/1
def orangesRotting(grid):
    def isValid(r,c) -> bool:
        nonlocal grid
        return r >=0 and r < len(grid) and c >= 0 and c < len(grid[0])

    def rotNeighbours(r,c):
        nonlocal grid
        neighbours = [(0,1),(0,-1),(1,0),(-1,0)]
        nextRotten = []
        for d in neighbours:
            cell = (r+d[0], c+d[1])
            if isValid(cell[0], cell[1]) and grid[cell[0]][cell[1]] == 1:
                grid[cell[0]][cell[1]] = 2 
                nextRotten.append(cell)
        return nextRotten

    q = deque()
    for r in range(len(grid)):
        for c in range(len(grid[0])):
            if grid[r][c] == 2:
                q.append((r,c))

    days = 0
    while len(q) > 0:
        q.append(None)
        while q[0] is not None:
            rottenIndex = q.popleft()
            rn = rotNeighbours(rottenIndex[0], rottenIndex[1])
            for n in rn:
                q.append(n)
        q.popleft()
        if len(q) > 0:
            days += 1
    
    for r in range(len(grid)):
        for c in range(len(grid[0])):
            if grid[r][c] == 1:
                days = -1
    
    return days

def tour(lis, n):
    remainingGas = 0
    i,j,len = 0,0,0
    while len < n and i < n:
        len += 1
        remainingGas += lis[j][0] - lis[j][1]
        #print("End updated:", i,j,len,remainingGas)

        while remainingGas < 0 and i < n and len > 0:
            remainingGas -= lis[i][0] - lis[i][1]
            i += 1
            len -= 1
            #print("Start updated:", i,j,len,remainingGas)

        j = (j+1)%n
    
    #print("Loop over", i,j,len,remainingGas)
    if len == n and remainingGas >= 0:
        return i
    else:
        return -1

def firstNonRepeating(A):
    countMap = {}
    q = deque()
    result = ""

    for c in A:
        if c in countMap:
            countMap[c] += 1
            while len(q) > 0 and countMap[q[0]] > 1:
                q.popleft()
        else:
            countMap[c] = 1
            q.append(c)

        result += q[0] if len(q) > 0 else "#"
    return result
