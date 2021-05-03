
#
# MinStack.py
#

class MinStack:
	
	def __init__(self):
		self.list = []
		self.minStack = []

	def push(self, v):
		self.list.append(v)
		if len(self.minStack) == 0 or v <= self.minStack[-1]:
			self.minStack.append(v)

	def pop(self):
		v = self.list.pop()
		if v <= self.minStack[-1]:
			self.minStack.pop()
		return v

	def getMin(self):
		return self.minStack[-1]

if __name__ == "__main__":
	test = [3,2,1,4,5,2,8]
	stack = MinStack()
	for i in test:
		stack.push(i)

	print(stack.pop())
	print(stack.pop())
	print(stack.pop())
	print("Min: ", stack.getMin())
	print(stack.pop())
	print("Min: ", stack.getMin())
	print(stack.pop())
	print("Min: ", stack.getMin())
	print(stack.pop())
	print("Min: ", stack.getMin())
	print(stack.pop())
