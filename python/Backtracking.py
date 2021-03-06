
################################################################################
# -------------------------- Backtracking GfG Must Do ------------------------ #
################################################################################

#https://practice.geeksforgeeks.org/problems/rat-in-a-maze-problem/1
def nQueens(n):
	def invalidatedSquaresOnRow(k, r,c):
		sq = set([c])
		l = c - (k-r)
		if l >= 0:
			sq.add(l)
		r = c + (k-r)
		if r < n:
			sq.add(r)
		return sq
	
	def findSolutionForRow(k, queens):
		invalidSquares = set()
		for i in range(k):
			invalidSquares |= invalidatedSquaresOnRow(k,i,queens[i])
		validSquares = []
		for i in range(n):
			if i not in invalidSquares:
				validSquares.append(i)

		if k == n-1:
			return [queens+[c] for c in validSquares]

		solutions = []
		for c in validSquares:
			solutions += findSolutionForRow(k+1,queens+[c])
		return solutions
	
	s = findSolutionForRow(0, [])
	for solution in s:
		for i in range(len(solution)):
			solution[i] += 1
	return s

#https://practice.geeksforgeeks.org/problems/rat-in-a-maze-problem/1
def solveMazeForRat(m,n):
	directions = {
					"U": (-1,0),
					"D": (1,0),
					"L": (0,-1),
					"R": (0,1)
				}

	def nextSteps(r,c, visited):
		nextSteps = []
		for d in directions:
			row = r + directions[d][0]
			column = c + directions[d][1]

			if ((row >= 0 and row < n and column >= 0 and column < n) and
				((row, column) not in visited)):
				nextSteps.append(d)
		return sorted(nextSteps)
		
	def solveForPosition(r,c,path,visited):
		if m[r][c] != 1:
			return []

		if r == n-1 and c == n-1:
			return [path]

		nv = visited.copy()
		nv.add((r,c))
		
		solutions = []
		for s in nextSteps(r,c,nv):
			d = directions[s]
			solutions += solveForPosition(r+d[0], c+d[1], path+s, nv)
		return solutions
	
	return solveForPosition(0,0,"",set())

#https://practice.geeksforgeeks.org/problems/word-boggle4143/1#
def wordBoggle(board,dictionary):
	trie = Trie(dictionary)

	neighbours = [
					(-1,-1),
					(-1,0),
					(-1,1),
					(0,-1),
					(0,1),
					(1,-1),
					(1,0),
					(1,1),
				]
	def getNeighbours(i,j,visited):
		steps = []
		for n in neighbours:
			r = i + n[0]
			c = j + n[1]
			if r >= 0 and r < len(board) and c >= 0 and c < len(board[r]):
				if (r,c) not in visited:
					steps.append((r,c))
		return steps

	def explore(i,j,node, visited):
		if not node.hasChildren():
			return [node.path] if len(node.path) > 0 else []

		nv = visited.copy()
		nv.add((i,j))

		words = []
		if node.isComplete:
			words.append(node.path)
		for s in getNeighbours(i,j,nv):
			r,c = s[0],s[1]
			char = board[r][c]
			if char in node.children:
				words += explore(r,c,node.children[char],nv)
		return words

	words = set()
	for i in range(len(board)):
		for j in range(len(board[i])):
			if board[i][j] in trie.children:
				for word in explore(i,j, trie.children[board[i][j]], set()):
					words.add(word)
	return sorted(list(words))

#https://practice.geeksforgeeks.org/problems/generate-ip-addresses/1#
def generateAllPossibleIPAddresses(s):
    def placeDots(s, parts, remaining):
        if len(s) < remaining:
            #print("Call:", s, parts, remaining, "result:", [])
            return []

        if remaining == 0:
            if len(s) == 0:
                return [".".join(parts)]
            else:
                return [".".join(parts+[s])]
        
        addresses = []
        uniqueAddresses = set()
        for i in range(1,4):
            if int(s[:i]) < 256:
                newAddresses = placeDots(s[i:],parts+[s[:i]],remaining-1)
                for a in newAddresses:
                    if a not in uniqueAddresses:
                        uniqueAddresses.add(a)
                        addresses.append(a)
        #print("Call:", s, parts, remaining, "result:", addresses)
        return addresses

    return placeDots(s,[],4)

#https://leetcode.com/problems/path-with-maximum-gold/
def getMaxGold(grid, n, m):
    stack = set()

    def isValid(r,c):
        if 0 <= r < n and 0 <= c < m:
            if (r,c) not in stack:
                return grid[r][c] != 0
            else:
                return False
        else:
            return False

    def getValidChildren(r, c):
        d = [1,0,-1,0,1]
        children = []
        for i in range(4):
            r1, c1 = r+d[i], c+d[i+1]
            if isValid(r1,c1):
                children.append((r1, c1))
        return children

    def explore(r, c):
        stack.add((r,c))
        mg = 0
        for cr, cc in getValidChildren(r,c):
            mg = max(mg, explore(cr, cc))
        stack.remove((r, c))
        return mg + grid[r][c]

    gold = 0
    for r in range(n):
        for c in range(m):
            if grid[r][c] == 0:
                continue
            gold = max(gold, explore(r,c))

    return gold
