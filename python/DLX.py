
class DLXNode:
    def __init__(self):
        self.l = None
        self.r = None
        self.u = None
        self.d = None
        self.c = None
        self.size = 0
        self.name = None

    def removeH(self):
        self.r.l = self.l
        self.l.r = self.r

    def removeV(self):
        self.u.d = self.d
        self.d.u = self.u
        self.c.size -= 1

    def addH(self):
        self.r.l = self
        self.l.r = self

    def addV(self):
        self.d.u = self
        self.u.d = self
        self.c.size += 1

class DLXMatrix:
    def __init__(self, M, names=None):
        self.names = names
        self.root, self.header = self._columnHeaders(M)
        self._setupCells(M)
        #self.header = None

    def __str__(self):
        node = self.root.r
        ans = []
        while node is not self.root:
            ans.append(node.name)
            node = node.r
        return "".join(ans)

    def _columnHeaders(self, M):
        n, m = len(M), len(M[0])
        root, header = DLXNode(), []
        last = root
        for i in range(m):
            node = DLXNode()
            last.r, node.l = node, last
            last = node
            header.append(node)
            if self.names:
                node.name = self.names[i]
            else:
                node.name = chr(ord('A') + i)

        root.l = header[-1]
        header[-1].r = root
        return root, header

    def _setupCells(self, M):
        n, m = len(M), len(M[0])
        row = self.header.copy()
        first, last = None, None
        for i in range(n):
            first, last = None, None
            for j in range(m):
                if not M[i][j]:
                    continue

                node = DLXNode()
                node.c = self.header[j]
                node.name = str(i)
                self.header[j].size += 1
                if last:
                    node.l = last
                    last.r = node
                else:
                    first = node
                last = node
                node.u = row[j]
                row[j].d = node
                row[j] = node
            if last:
                first.l = last
                last.r = first

        for i in range(m):
            row[i].d = self.header[i]
            self.header[i].u = row[i]
    
    def getRow(self, r):
        node = r.r
        s = [r.c.name]
        while node is not r:
            s.append(node.c.name)
            node = node.r
        return "".join(s)

    def solve(self):
        return self.search([])

    def chooseColumn(self):
        if self.root.r is self.root:
            return None

        node = self.root.r
        selected = node
        while node is not self.root:
            if node.size < selected.size:
                selected = node
            node = node.r
        return selected

    def cover(self, c):
        c.removeH()
        node = c.d
        while node is not c:
            next = node.r
            while next is not node:
                next.removeV()
                next = next.r
            node = node.d

    def uncover(self, c):
        node = c.u
        while node is not c:
            next = node.l
            while next is not node:
                next.addV()
                next = next.l
            node = node.u
        c.addH()

    def hideRow(self, row):
        node = row.r
        while node is not row:
            self.cover(node.c)
            node = node.r

    def unhideRow(self, row):
        node = row.l
        while node is not row:
            self.uncover(node.c)
            node = node.l

    def search(self, s):
        if self.root.r is self.root:
            return ["".join(sorted(s))]

        sol = []
        c = self.chooseColumn()
        self.cover(c)
        r = c.d
        while r is not c:
            self.hideRow(r)
            sol += self.search(s+[r.name])
            self.unhideRow(r)
            r = r.d
        self.uncover(c)
        return sol

A = [   [0,0,1,0,1,1,0],
        [1,0,0,1,0,0,1],
        [0,1,1,0,0,1,0],
        [1,0,0,1,0,0,0],
        [0,1,0,0,0,0,1],
        [0,0,0,1,1,0,1],
        [0,1,1,1,1,1,1],
        [1,0,0,0,0,0,0],
        [1,0,0,1,0,0,0],
        [1,1,0,1,0,0,1],
    ]

M = DLXMatrix(A)
print(DLXMatrix(A).solve())
