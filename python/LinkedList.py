
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None
	
def createList(arr) -> Node:
	head = Node(arr[0])
	current = head
	for n in arr[1:]:
		current.next = Node(n)
		current = current.next
	
	return (head, current)

def findMid(head):
    # Code here
    # return the value stored in the middle node
    n = 0
    nextNode = head
    while nextNode is not None:
        n += 1
        nextNode = nextNode.next
    
    nextNode = head
    for i in range(n//2):
        nextNode = nextNode.next
    return nextNode.data

def reverseList(head):
	last = None
	current = head
	nextNode = head.next

	while current is not None:
		current.next = last
		
		last = current
		current = nextNode
		nextNode = nextNode.next if nextNode else None
	return last

def rotate(head, k):
	def getNodeAtIndex(k, head):
		node = head
		for i in range(k):
			if node:
				node = node.next
			else:
				break
		return node
	
	newLast = getNodeAtIndex(k-1,head)
	newHead = newLast.next
	if newHead is None:
		return head
	newLast.next = None

	oldLast = newHead
	while oldLast.next is not None:
		oldLast = oldLast.next
	
	oldLast.next = head
	return newHead

def rotateListInGroups(head,k):
	def getKthNode(head, k):
		n = head
		for i in range(k):
			if n:
				n = n.next
			else:
				break
		return n
	
	def reverseList(head):
		last = None
		current = head
		nextNode = head.next
	
		while current is not None:
			current.next = last
			
			last = current
			current = nextNode
			nextNode = nextNode.next if nextNode else None
		return last

	lastTail = None
	group = (head, getKthNode(head,k-1))
	nextHead = None
	newHead = None
	while group[0]:
		if group[1]:
			nextHead = group[1].next
			group[1].next = None
		else:
			nextHead = None
		if lastTail:
			lastTail.next = None
			lastTail.next = reverseList(group[0])
		else:
			reverseList(group[0])

		if newHead is None:
			newHead = group[1]

		lastTail = group[0]
		group = (nextHead, getKthNode(nextHead,k-1))
		if group[1]:
			nextHead = group[1].next
	
	return newHead

def findIntersection(head1,head2) -> int:
	def findLength(h) -> int:
		n = h
		count = 0
		while n is not None:
			count += 1
			n = n.next

		return count
	
	def getElementAtIndex(h,k):
		n = h
		for i in range(k):
			if n:
				n = n.next
			else:
				break
		return n
	
	n1 = findLength(head1)
	n2 = findLength(head2)
	if n2 < n1:
		head1, head2 = head2, head1
		n2, n1 = n1, n2

	p1 = head1
	p2 = getElementAtIndex(head2,n2-n1)
	while p1 and p2 and p1 is not p2:
		p1 = p1.next
		p2 = p2.next
	
	if p1 and p2 and p1 is p2:
		return p1.data
	else:
		return -1 

def detectLoop(head):
	slow = head
	fast = head

	while slow and fast and slow is not fast:
		slow = slow.next
		if fast.next:
			fast = fast.next.next
		else:
			fast = None
	
	if slow and fast and slow is fast:
		return True
	else:
		return False

def kthNodeFromEnd(head, k):
	slow = head
	fast = head

	while fast and fast.next:
		slow = slow.next
		fast = fast.next.next
		if slow is fast:
			return True
	
	return False

def mergeLinkedLists(h1,h2):
	p1, p2 = h1, h2
	head = Node(-1)
	p = head
	while p1 and p2:
		if p1.data < p2.data:
			p.bottom = Node(p1.data)
			p1 = p1.bottom
		else:
			p.bottom = Node(p2.data)
			p2 = p2.bottom
		p = p.bottom

	while p1:
		p.bottom = Node(p1.data)
		p1 = p1.bottom
		p = p.bottom
	while p2:
		p.bottom = Node(p2.data)
		p2 = p2.bottom
		p = p.bottom
	return head.bottom
	
def flattenList(root):
	head = None
	p = root
	while p:
		head = mergeLinkedLists(head,p)
		p = p.next
	return head

#https://practice.geeksforgeeks.org/problems/add-two-numbers-represented-by-linked-lists/1
def addTwoLists(first, second):
	def numberForList(head) -> int:
		n = 0
		h = head
		while h is not None:
			n = n*10 + h.data
			h = h.next
		return n
	
	n = numberForList(first) + numberForList(second)
	s = str(n)
	resultHead = Node(int(s[0]))
	current = resultHead
	for c in s[1:]:
		current.next = Node(int(c))
		current = current.next
	return resultHead

#https://practice.geeksforgeeks.org/problems/check-if-linked-list-is-pallindrome/1
def isListPalindrome(head):
	number = 0
	reversedNumber = 0
	digitPlaceValue = 1
	current = head
	while current:
		number = number*10 + current.data
		reversedNumber = current.data*digitPlaceValue + reversedNumber
		digitPlaceValue *= 10
		current = current.next
	return number == reversedNumber

def segregate(head):
	counts = [0,0,0]
	current = head

	while current:
		counts[current.data] += 1
		current = current.next
	
	current = head
	for i in range(len(counts)):
		for j in range(counts[i]):
			current.data = i
			current = current.next
	return head

def removeLoop(head):
	tortoise = head
	hare = head

	while hare and hare.next:
		#print("In loop:", tortoise.data, hare.data)
		tortoise = tortoise.next
		hare = hare.next.next

		if tortoise.next is head:
			#print("In loop:", tortoise.data, hare.data)
			#print("Full loop found:", tortoise.data, "-->", tortoise.next.data)
			tortoise.next = None
			return

		if tortoise is hare:
			#print("loop found:", tortoise.data, hare.data)
			break
	
	if hare is None or hare.next is None:
		#print("loop not found:", tortoise.data, hare.data)
		return
	
	hare = head
	while tortoise.next is not hare.next and hare is not tortoise:
		#print("In second loop", tortoise.data, hare.data)
		tortoise = tortoise.next
		hare = hare.next
	
	#print("Link:", tortoise.data, "-->", tortoise.next.data)
	tortoise.next = None
