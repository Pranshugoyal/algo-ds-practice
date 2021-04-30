
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


if __name__ == "__main__":
	stacks = TwoStack(4)
	
	stacks.push(1, 0)
	stacks.push(2, 0)
	stacks.push(5, 1)
	stacks.push(8, 1)

	print(stacks.peek(0), stacks.peek(1))
	print(stacks.storage)
