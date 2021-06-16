
################################################################################
# ----------------------------- GfG Graph Must Do ---------------------------- #
################################################################################

from collections import deque
import sys

#https://practice.geeksforgeeks.org/problems/bfs-traversal-of-graph/1
def graphBFS(V, adj):
	visited = set()

	def bfsForNode(node):
		q = deque()
		q.append(node)
		res = []
		while len(q) > 0:
			v = q.popleft()
			if v in visited:
				continue

			visited.add(v)
			res.append(v)

			for u in adj[v]:
				if u not in visited:
					q.append(u)
		return res
	
	result = []
	for i in range(V):
		if i not in visited:
			result += bfsForNode(i)
	return result

#https://practice.geeksforgeeks.org/problems/depth-first-traversal-for-a-graph/1
def graphDFS(V,adj):
	visited = set()
	def dfs(node):
		if node in visited:
			return []

		visited.add(node)
		res = [node]
		for u in adj[node]:
			res += dfs(u)
		return res
	return dfs(0)

#https://practice.geeksforgeeks.org/problems/topological-sort/1
def topologicalSort(V, adj):
    indegreeCount = [0 for i in range(V)]
    for v in range(V):
        for u in adj[v]:
            indegreeCount[u] += 1

    sortedOrder = []
    q = deque()
    for v in range(V):
        if indegreeCount[v] == 0:
            q.append(v)

    while len(q) > 0:
        v = q.popleft()
        sortedOrder.append(v)

        for u in adj[v]:
            indegreeCount[u] -= 1
            if indegreeCount[u] == 0:
                q.append(u)

    return sortedOrder

#https://practice.geeksforgeeks.org/problems/detect-cycle-in-an-undirected-graph/1/#
def detectCycleInUndirectedGraph(V, adj) -> bool:
    def performDFS(v, parent):
        visited.add(v)
        for u in adj[v]:
            if u not in visited:
                if performDFS(u,v):
                    return True
            elif u != parent:
                return True
        return False

    visited = set()
    for v in range(V):
        if v not in visited:
            if performDFS(v, None):
                return True
    return False

def detectCycleUndirectedGraphUnionFind(V, adj):
    from UnionFind import UnionFind
    uf = UnionFind([i for i in range(V)])

    edges = []
    for v in range(V):
        for u in adj[v]:
            if v < u:
                edges.append((v,u))
            elif v == u:
                return True

    for edge in edges:
        v,u = edge
        if uf.areDisjoint(u, v):
            uf.union(u,v)
        else:
            return True
    return False

#https://practice.geeksforgeeks.org/problems/detect-cycle-in-a-directed-graph/1
def detectCycleDirectedGraph(V, adj):
    def performDFS(v):
        visited.add(v)
        recStack.add(v)
        for u in adj[v]:
            if u not in visited:
                if performDFS(u):
                    return True
            elif u in recStack:
                return True

        recStack.remove(v)
        return False

    visited = set()
    recStack = set()
    for v in range(V):
        if v not in visited:
            if performDFS(v):
                return True
    return False

#https://practice.geeksforgeeks.org/problems/find-the-number-of-islands/1
#https://leetcode.com/problems/number-of-islands/
def countNumberOfIslands(grid) -> int:
    neighbours = [-1,-1,0,-1,1,0,1,1,-1]
    visited = []
    for i in range(len(grid)):
        visited.append([False]*len(grid[i]))

    def shouldExplore(r,c) -> bool:
        if r >= 0 and r < len(grid) and c >= 0 and c < len(grid[r]):
            return grid[r][c] == 1 and not visited[r][c]
        else:
            return False

    def explore(r,c):
        if visited[r][c]:
            return

        visited[r][c] = True

        for i in range(len(neighbours)-1):
            nr = r + neighbours[i]
            nc = c + neighbours[i+1]
            if shouldExplore(nr, nc):
                explore(nr, nc)

    islandCount = 0
    for r in range(len(grid)):
        for c in range(len(grid[r])):
            if shouldExplore(r,c):
                islandCount += 1
                explore(r,c)

    return islandCount

#https://practice.geeksforgeeks.org/problems/implementing-dijkstra-set-1-adjacency-matrix/1
def dijkstra(V, adj, S):
    visited = set()
    distance = [sys.maxsize]*V
    distance[S] = 0
    
    def nextVertex():
        minD = sys.maxsize
        vid = None
        for v in range(V):
            if v not in visited and distance[v] < minD:
                minD = distance[v]
                vid = v
        return vid

    v = S
    while v is not None:
        visited.add(v)
        for edge in adj[v]:
            u, d = edge[0], edge[1]
            if u not in visited:
                distance[u] = min(distance[u], distance[v] + d)
        v = nextVertex()

    return distance

#Bellman-Ford
#https://practice.geeksforgeeks.org/problems/negative-weight-cycle3504/1#
def isNegativeWeightCycle(n, edges):
    cost = [None]*n
    cost[0] = 0

    def updateEdgeCost(edge) -> bool:
        v, u, w = edge[0], edge[1], edge[2]

        if cost[v] is None:
            return False

        if cost[u] is None:
            cost[u] = cost[v] + w
            return True
        elif cost[u] > cost[v] + w:
            cost[u] = cost[v] + w
            return True
        else:
            return False

    def runUpdateCycle() -> bool:
        res = False
        for edge in edges:
            res = res or updateEdgeCost(edge)
        return res

    for _ in range(n-1):
        runUpdateCycle()

    return runUpdateCycle()

#https://practice.geeksforgeeks.org/problems/shortest-source-to-destination-path/0
def shortestDistance(N,M,A,X,Y):
    if A[0][0] == 0:
        return -1

    visited = []
    for i in range(N):
        visited.append([False]*M)

    def canMoveTo(r,c):
        if r >= 0 and r < N and c >= 0 and c < M:
            return A[r][c] == 1 and not visited[r][c]
        else:
            return False

    directions = [-1,0,1,0,-1]
    def getValidNeighbours(r,c):
        neighbours = []
        for i in range(len(directions)-1):
            nr,nc = r+directions[i], c+directions[i+1]
            if canMoveTo(nr,nc):
                neighbours.append((nr,nc))
        return neighbours

    q = deque()
    q.append((0,0))
    visited[0][0] = True
    steps = 0
    while len(q) > 0:
        q.append(None)
        while q[0] is not None:
            r,c = q.popleft()

            if (X,Y) == (r,c):
                return steps

            for cell in getValidNeighbours(r,c):
                q.append(cell)
                visited[cell[0]][cell[1]] = True
        q.popleft()
        steps += 1
    return -1

#https://practice.geeksforgeeks.org/problems/find-whether-path-exist5238/1
def doesPathExists(grid):
    n,m = len(grid), len(grid[0])
    source, destination = None, None

    for r in range(n):
        for c in range(m):
            if grid[r][c] == 1:
                source = (r,c)
            elif grid[r][c] == 2:
                destination = (r,c)
        if source and destination:
            break

    visited = []
    for r in range(n):
        visited.append([False for c in range(m)])

    directions = [-1,0,1,0,-1]
    def getValidNeighbours(r,c):
        neighbours = []
        for i in range(4):
            nr,nc = r+directions[i], c+directions[i+1]
            if nr >= 0 and nr < n and nc >= 0 and nc < m:
                if grid[nr][nc] != 0 and not visited[nr][nc]:
                    neighbours.append((nr,nc))
        return neighbours

    def searchTarget(s,t):
        visited[s[0]][s[1]] = True
        neighbours = getValidNeighbours(s[0], s[1])
        if t in neighbours:
            return True

        for cell in neighbours:
            if searchTarget(cell,t):
                return True
        return False

    visited[source[0]][source[1]] = True
    return searchTarget(source, destination)

#https://practice.geeksforgeeks.org/problems/minimum-cost-path3833/1
def minimumCostPath(grid):
    n,m = len(grid), len(grid[0])

    visited = []
    for r in range(n):
        visited.append([False]*m)

    directions = [-1,0,1,0,-1]
    def getNeighbours(r,c):
        neighbours = []
        for i in range(4):
            nr,nc = r+directions[i], c+directions[i+1]
            #print("Checking neighbour", (nr,nc), "of", (r,c))
            if nr >= 0 and nr < n and nc >= 0 and nc < m and not visited[nr][nc]:
                neighbours.append((nr,nc))
        return neighbours 

    cost = []
    for r in range(n):
        cost.append([sys.maxsize]*m)

    def getNearestCell():
        d = sys.maxsize
        cell = None
        for r in range(n):
            for c in range(m):
                if cost[r][c] < d and not visited[r][c]:
                    d = cost[r][c]
                    cell = (r,c)
        #print("Next nearest", cell)
        return cell

    def updateDistance(s,d):
        #print("Updating", d)
        cs = cost[s[0]][s[1]]
        cp = grid[d[0]][d[1]]
        cd = cost[d[0]][d[1]]
        cost[d[0]][d[1]] = min(cd, cs+cp)
        #print("Updated",d,cost[d[0]][d[1]])

    cost[0][0] = grid[0][0]
    cell = (0,0)
    destination = (n-1,m-1)
    while cell:
        if cell == destination:
            break
        r,c = cell
        visited[r][c] = True
        #print("Neighbours of", cell, getNeighbours(r,c))
        for nc in getNeighbours(r,c):
            updateDistance(cell, nc)
        cell = getNearestCell()

    #print("Cost grid:")
    #for row in cost:
        #print(row)
    return cost[destination[0]][destination[1]]

#https://practice.geeksforgeeks.org/problems/circle-of-strings4530/1
def areStringsInCircle(A):
    adj = []
    for i in range(len(A)):
        edges = []
        for j in range(len(A)):
            if i == j:
                continue
            if A[i][-1] == A[j][0]:
                edges.append(j)
        adj.append(edges)
    return detectCycleDirectedGraph(len(A), adj)

def floydWarshall(matrix):
    n = len(matrix)
    mc = matrix.copy()

    for i in range(n):
        for j in range(n):
            if mc[i][j] == -1:
                mc[i][j] = sys.maxsize

    for k in range(n):
        for i in range(n):
            for j in range(n):
                matrix[i][j] = min(mc[i][j], mc[i][k] + mc[k][j])
        mc = matrix

    for i in range(n):
        for j in range(n):
            if mc[i][j] == sys.maxsize:
                mc[i][j] = -1
    matrix = mc
