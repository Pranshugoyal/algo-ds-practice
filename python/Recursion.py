
#
# RecursionStandardProblems
# https://www.youtube.com/playlist?list=PL_z_8CaSLPWeT1ffjiImo0sYTcnLzo-wY
#

################################################################################
# -------------------- Recursion Problems/Aditya Verma ----------------------- #
################################################################################

def sortStackUsingRecursion(stack):
	if len(stack) < 2:
		return stack
	
	last = stack.pop()
	sortStackUsingRecursion(stack)
	
	larger = []
	while len(stack) > 0 and stack[-1] > last:
		larger.append(stack.pop())
	
	stack.append(last)
	for n in reversed(larger):
		stack.append(n)
	
	return stack

def kSymbolInGrammer(n,k):
	if n == 1 and k == 1:
		return 0

	last = kSymbolInGrammer(n-1,(k+1)//2)
	new = (0,1) if last == 0 else (1,0)
	return new[(k+1)%2]

def towerOfHanoi(n,s,d,h):
	if n == 1:
		print(n, s, "-->", d)
		return
	
	towerOfHanoi(n-1,s,h,d)
	print(n, s, "-->", d)
	towerOfHanoi(n-1,h,d,s)

def printAllSubsets(S, f = ""):
	if len(S) == 0:
		print(f)
		return
	
	printAllSubsets(S[1:], f)
	printAllSubsets(S[1:], f+S[0])

def generateAllBalancedParenthesis(o,c, f = ""):
	if o == 0 and c == 0:
		print(f)
		return
	elif o == 0:
		print(f+(")"*c))
		return
	elif c == 0:
		raise Exception("Imbalanced fixed: ", f)
		
	if c > o:
		generateAllBalancedParenthesis(o, c-1,f+")")
	generateAllBalancedParenthesis(o-1, c, f+"(")

def josephusProblem(n, k):
	arr = [i+1 for i in range(n)]
	def josephusProblemR(arr, i, k):
		if len(arr) == 1:
			return arr[0]

		next = (i + k)%len(arr)
		killed = arr.pop(next)
		return josephusProblemR(arr,next,k)

	return josephusProblemR(arr, 0, k-1)	

################################################################################
# --------------------------- Must Do GeeksForGeeks -------------------------- #
################################################################################

from collections import deque
import functools

def floodFill(image, sr, sc, newColor):
	
	startColor = image[sr][sc]
	def shouldPaint(r,c):
		nonlocal image, startColor, newColor

		if r < 0 or r >= len(image):
			return False

		if c < 0 or c >= len(image[0]):
			return False

		color = image[r][c]
		return color == startColor and color != newColor
	
	def getNeighbours(r, c):
		directions = [(0,1), (1,0), (0,-1), (-1,0)]
		neighbours = []
		for d in directions:
			neighbours.append((r+d[0], c+d[1]))
		return neighbours
	
	# BFS
	q = deque([(sr, sc)])
	cell = None
	while len(q) > 0:
		cell = q.popleft()
		if shouldPaint(cell[0], cell[1]):
			image[cell[0]][cell[1]] = newColor
			for n in getNeighbours(cell[0], cell[1]):
				q.append(n)
	
	# DFS
	image[sr][sc] = newColor
	for n in getValidNeighbours(sr,sc):
		self.floodFill(image, n[0], n[1], newColor)
	return image

def combinationSum(A, N, B):
	if len(A) == 0:
		return []
	elif len(A) == 1:
		return A if A[0] == B else []
	
	tail = A[-1]
	head = A[:-1]
	solutions = combinationSum(head, N-1, B)
	if tail == B:
		solutions.append([tail])
	elif tail < B:
		for s in combinationSum(head, N-1, B-tail):
			solutions.append([tail] + s)
	
	unique = []
	for s in solutions:
		if s not in unique:
			unique.append(s)
	solutions = unique

	for s in solutions:
		s.sort()
	solutions.sort()
	return solutions

def optimalKeys(N):
	@functools.lru_cache(maxsize=None)
	def keysUtil(printed, selected, clipboard, keyStrokesLeft) -> int:
		if keyStrokesLeft == 0:
			return printed

		typeA = keysUtil(printed+1, 0, clipboard, keyStrokesLeft-1)
		selectText = keysUtil(printed, printed, clipboard, keyStrokesLeft-1)
		copyText = keysUtil(printed, 0, selected, keyStrokesLeft-1)
		pasteClipboard = keysUtil(printed+clipboard, 0, clipboard, keyStrokesLeft-1)

		return max(typeA, pasteClipboard, copyText, selectText)

	@functools.lru_cache(maxsize=None)
	def keysUtilIter(n):
		if n <= 6:
			return n
		
		maxA = 0
		for i in range(n-3, -1, -1):
			maxA = max(maxA, keysUtilIter(i)*(n-i-1))
		return maxA

	return keysUtilIter(N)

def josephus(n,k):
	def nextManStanding(arr, p, k) -> int:
		count = 1
		while count < k:
			if arr[p+1]:
				count += 1
			p = (p+1)%len(arr)
		return p

	arr = [i+1 for i in range(n)]
	nextMan = 0
	for kills in range(n-1):
		kill = nextManStanding(arr,nextMan,k)
		arr[kill] = None
		nextMan = kill
		while not arr[nextMan]:
			nextMan = (nextMan + 1)%n
		#print(arr, kill, nextMan)

	return arr[nextMan]

#https://leetcode.com/problems/count-good-numbers/
def countGoodNumbers(n):
    def fastExp(b, e, M):
        if e < 3:
            return b**e
        elif e%2 == 0:
            return fastExp(b, e//2, M) ** 2 % M
        else:
            return fastExp(b, e-1, M) * b % M

    M = 10**9 + 7
    return fastExp(4, n//2, M) * fastExp(5, n - n//2, M) % M
