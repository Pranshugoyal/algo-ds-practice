
#
# RecursionStandardProblems
# https://www.youtube.com/playlist?list=PL_z_8CaSLPWeT1ffjiImo0sYTcnLzo-wY
#

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

