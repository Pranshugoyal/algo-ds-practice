
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

#https://www.geeksforgeeks.org/union-find/
def detectCycleUndirectedGraphUnionFind(V, adj):
    from UnionFind import UnionFind
    uf = UnionFind([i for i in range(V)])

    #Making sure edges are only explored once
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

    for i,d in enumerate(distance):
        if d == sys.maxsize:
            distance[i] = None
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

#https://practice.geeksforgeeks.org/problems/circle-of-strings4530/1#
def isCircle(A):
    adj = []
    for i in range(26):
        adj.append([])

    for s in A:
        adj[ord(s[0])-ord('a')].append(ord(s[-1])-ord('a'))
    return eulerianCircuitInDirectedGraph(len(adj), adj)

#https://practice.geeksforgeeks.org/problems/implementing-floyd-warshall/0
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
                mc[i][j] = min(mc[i][j], mc[i][k] + mc[k][j])

    for i in range(n):
        for j in range(n):
            if mc[i][j] == sys.maxsize:
                mc[i][j] = -1
    return mc

#https://practice.geeksforgeeks.org/problems/alien-dictionary/1#
def alienDictionary(dict, K):
    adj = []
    for i in range(K):
        adj.append(set())

    for i in range(len(dict)-1):
        w1, w2 = dict[i], dict[i+1]
        k = min(len(w1), len(w2))
        for j in range(k):
            if w1[j] != w2[j]:
                adj[ord(w1[j])-ord('a')].add(ord(w2[j])-ord('a'))
                break

    s = topologicalSort(K, adj)
    print(adj, s)
    ans = ""
    for i in s:
        ans += chr(i+ord('a'))
    return ans

#https://practice.geeksforgeeks.org/problems/snake-and-ladder-problem4816/1
def snakeAndLadderMinThrows(N, arr, source=1, destination=30):
    snl = {}
    for i in range(N):
        snl[arr[2*i]] = arr[2*i+1] 

    adj = [[]]
    for i in range(1, 31):
        l = []
        if i in snl:
            adj.append(l)
            continue
        for d in range(i+1, min(i+6,30)+1):
            if d in snl:
                l.append(snl[d])
            else:
                l.append(d)
        adj.append(l)

    q = deque()
    visited = set()
    q.append(source)
    visited.add(source)
    steps = 0
    while len(q) > 0:
        q.append(None)
        while q[0] is not None:
            v = q.popleft()
            if v == destination:
                return steps

            for u in adj[v]:
                if u not in visited:
                    visited.add(u)
                    q.append(u)
        q.popleft()
        steps += 1
    return -1

################################################################################
# -------------------------------- GfG Practice ------------------------------ #
################################################################################

#https://practice.geeksforgeeks.org/problems/strongly-connected-components-kosarajus-algo/1
def kosarajuSCC(V, adj):

    #This is topological sorting in reverse order
    def getTravelSequence(v, stack):
        visited.add(v)
        for u in adj[v]:
            if u in visited:
                continue
            getTravelSequence(u, stack)
        stack.append(v)
        return stack

    def getTranspose(adj):
        tadj = [[] for i in range(V)]
        for v in range(V):
            for u in adj[v]:
                tadj[u].append(v)
        return tadj

    def getDfsTree(v, adj, comp):
        visited.add(v)
        comp.add(v)
        for u in adj[v]:
            if u in visited:
                continue
            getDfsTree(u,adj,comp)
        return comp

    visited = set()
    stack = []
    for i in range(V):
        if i not in visited:
            getTravelSequence(i, stack)
    visited = set()
    transpose = getTranspose(adj)
    components = []
    while len(stack) > 0:
        v = stack.pop()
        if v in visited:
            continue
        components.append(getDfsTree(v, transpose, set()))
    return len(components)

#https://www.geeksforgeeks.org/tarjan-algorithm-find-strongly-connected-components/
#https://practice.geeksforgeeks.org/problems/strongly-connected-component-tarjanss-algo-1587115621/1
def tarjanSCC(V, adj):
    counter = 0
    low, disc = [None]*V, [None]*V
    stack, inStack = [], set()
    scc = []

    def dfsUtil(v):
        nonlocal counter
        low[v], disc[v] = counter, counter
        counter += 1
        stack.append(v)
        inStack.add(v)

        for u in adj[v]:
            if disc[u] is None:
                dfsUtil(u)
                low[v] = min(low[v], low[u])
            elif u in inStack:
                low[v] = min(low[v], disc[u])

        #Check head
        if low[v] == disc[v]:
            i = stack.index(v)
            #Empty inStack in batch and not individually
            for u in stack[i:]:
                inStack.remove(u)
            scc.append(sorted(stack[i:]))
            stack[i:] = []

    for v in range(V):
        if disc[v] is None:
            dfsUtil(v)
    return scc

#https://practice.geeksforgeeks.org/problems/euler-circuit-and-path/1
def eulerianPathAndCircuit(V, adj):
    def dfsUtil(v, visited):
        visited.add(v)
        for u in adj[v]:
            if u not in visited:
                dfsUtil(u, visited)
        return visited

    def isConnected():
        source = None
        for v in range(V):
            if len(adj) > 0:
                source = v
                break

        if source is None:
            return True

        visited = dfsUtil(source, set())

        for v in range(V):
            if v not in visited and (adj[v]) > 0:
                return False
        return True

    if not isConnected():
        return 0

    oddDegree = 0
    for v in range(V):
        oddDegree += len(adj[v])%2

    if oddDegree == 0:
        return 2
    elif oddDegree == 2:
        return 1
    else:
        return 0

def eulerianCircuitInDirectedGraph(V, adj):
    def isStronglyConnected() -> bool:
        stack = []
        visited = set()
        def dfsTopoSort(v, sort):
            visited.add(v)
            for u in adj[v]:
                if u not in visited:
                    dfsTopoSort(u, sort)
            if sort:
                stack.append(v)

        def allVisited():
            for v in range(V):
                if len(adj[v]) > 0 and v not in visited:
                    return False
            return True

        for v in range(V):
            if len(adj[v]) > 0:
                dfsTopoSort(v, True)
                break

        if len(visited) == 0:
            return True

        if not allVisited():
            return False

        transpose = []
        for i in range(V):
            transpose.append([])
        for v in range(V):
            for u in adj[v]:
                transpose[u].append(v)

        visited = set()
        dfsTopoSort(stack[-1], False)
        return allVisited()

    if not isStronglyConnected():
        return False

    indegree = [0]*V
    for v in range(V):
        for u in adj[v]:
            indegree[u] += 1

    for v in range(V):
        if len(adj[v]) != indegree[v]:
            return False
    return True

#https://www.geeksforgeeks.org/johnsons-algorithm-for-all-pairs-shortest-paths-implementation/
def johnsonsAlgorithm(V, adj):

    def runUpdateCycle(h):
        for v in range(len(h)):
            for edge in adj[v]:
                u,w = edge[0],edge[1]
                h[u] = min(h[u], h[v]+w)

    def reweightEdges(h):
        for v in range(len(h)):
            for i, edge in enumerate(adj[v]):
                u,w = edge[0], edge[1]
                adj[v][i] = (u, w+h[v]-h[u])

    adj.append([(i,0) for i in range(V)])
    h = [0]*(V+1)

    for _ in range(V):
        runUpdateCycle(h)
    
    h.pop()
    adj.pop()
    reweightEdges(h)

    d = []
    for v in range(V):
        d.append(dijkstra(V, adj, v))
    return d

#https://practice.geeksforgeeks.org/problems/minimum-spanning-tree/1#
#https://www.geeksforgeeks.org/kruskals-minimum-spanning-tree-algorithm-greedy-algo-2/
def kruksalMST(V, adj):
    from UnionFind import UnionFind

    edges = []
    for v in range(V):
        for u,w in adj[v]:
            edges.append((v,u,w))
    edges.sort(key=lambda x:x[2])

    uf = UnionFind(list(range(V)))
    mst = []
    for edge in edges:
        v,u,_ = edge
        if uf.areDisjoint(v,u):
            mst.append(edge)
            uf.union(v,u)
        if len(mst) == V-1:
            break
    mst.sort(key=lambda x:x[0])
    return mst

#https://practice.geeksforgeeks.org/problems/minimum-spanning-tree/1#
#https://www.geeksforgeeks.org/prims-minimum-spanning-tree-mst-greedy-algo-5/
def primsMST(V, adj):
    cost = [sys.maxsize]*V
    parent = [-1]*V

    #The graph has to be undirected, this is to ensure that
    for v in range(V):
        adj[v] = set(adj[v])

    for v in range(V):
        for u,w in adj[v]:
            adj[u].add((v,w))

    def getNextVertex(mst):
        minCost = sys.maxsize
        nv = None
        for v in range(V):
            if v not in mst and cost[v] < minCost:
                minCost = cost[v]
                nv = v
        return nv

    mst = set()
    source = 0
    cost[0] = source
    v = source
    while v is not None:
        mst.add(v)
        for u,w in adj[v]:
            if u not in mst and cost[u] >= w:
                cost[u] = w
                parent[u] = v
        v = getNextVertex(mst)
    
    edges = []
    for v in mst:
        if parent[v] != -1:
            edges.append((parent[v], v, cost[v]))
    edges.sort(key=lambda x:x[0])
    return edges

def comparePrimAndKruksal():
    def mstCost(mst):
        esum = 0
        for edge in mst:
            esum += edge[2]
        return esum

    adj = [
        [(1,4), (7,8)],
        [(0,4), (2,8), (7,11)],
        [(1,8), (3,7), (5,4)],
        [(4,9), (5,14)],
        [],
        [(4,10)],
        [(5,2)],
        [(0,8), (6,1), (8,7)],
        [(2,2), (6,6)]
    ]
    kruksal = kruksalMST(len(adj), adj)
    prims = primsMST(len(adj), adj)
    print("Kruksal:", kruksal, "\nPrims:\t", prims)
    print(mstCost(kruksal), mstCost(prims))

#https://eecs.wsu.edu/~holder/courses/CptS223/spr08/slides/graphapps.pdf
#https://practice.geeksforgeeks.org/problems/biconnected-graph2528/1#
def findArticulationPoints(V, adj):
    disc, low, ap = {}, {}, set()
    parent, counter = [None]*V, 0
    def dfsAP(v):
        nonlocal counter
        disc[v] = counter
        low[v] = disc[v]
        counter += 1

        child = 0
        for u in adj[v]:
            if u not in disc:
                child += 1
                parent[u] = v
                dfsAP(u)
                if low[u] >= disc[v] and parent[v] is not None:
                    ap.add(v)
                low[v] = min(low[v], low[u])
            elif u != parent[v]:
                low[v] = min(low[v], disc[u])

        if child > 1 and parent[v] is None:
            ap.add(v)
    dfsAP(2)
    return ap

#https://practice.geeksforgeeks.org/problems/bipartite-graph/1
def isBipartiteGraph(V, adj):
    visited = {}

    def addToGroup(v, g):
        visited[v] = g
        gt = 1 if g == 0 else 0

        for u in adj[v]:
            if u not in visited:
                if not addToGroup(u, gt):
                    return False
            elif visited[u] != gt:
                return False
        return True

    return addToGroup(0, 0)

