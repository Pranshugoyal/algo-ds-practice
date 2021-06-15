
################################################################################
# ----------------------------- GfG Graph Must Do ---------------------------- #
################################################################################

from collections import deque

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
    import sys

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
