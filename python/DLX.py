
class DLXNode:
    def __init__(self):
        self.l = None
        self.r = None
        self.u = None
        self.d = None
        self.c = None
        self.size = 0
        self.name = None

class DLXMatrix:
    def __init__(self, M, names=None):
        self.names = names
        self.root, self.header = self._columnHeaders(M)
        self._setupCells(M)
        self.header = None

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
        def removeH(node):
            node.r.l = node.l
            node.l.r = node.r

        def removeV(node):
            node.u.d = node.d
            node.d.u = node.u
            node.c.size -= 1

        removeH(c)
        node = c.d
        while node is not c:
            next = node.r
            while next is not node:
                removeV(next)
                next = next.r
            node = node.d

    def uncover(self, c):
        def addH(node):
            node.r.l = node
            node.l.r = node

        def addV(node):
            node.d.u = node
            node.u.d = node
            node.c.size += 1

        node = c.u
        while node is not c:
            next = node.l
            while next is not node:
                addV(next)
                next = next.l
            node = node.u
        addH(c)

    def solve(self):
        return self.search([])

    def search(self, s):
        if self.root.r is self.root:
            return ["".join(sorted(s))]

        sol = []
        c = self.chooseColumn()
        self.cover(c)
        r = c.d
        while r is not c:
            # Wind
            j = r.r
            while j is not r:
                self.cover(j.c)
                j = j.r

            sol += self.search(s+[r.name])

            # Unwind
            j = r.l
            while j is not r:
                self.uncover(j.c)
                j = j.l
            r = r.d
        self.uncover(c)
        return sol
    
    def getRow(self, r):
        node = r.r
        s = [r.c.name]
        while node is not r:
            s.append(node.c.name)
            node = node.r
        return "".join(s)

    def __str__(self):
        node = self.root.r
        ans = []
        while node is not self.root:
            ans.append(node.name)
            node = node.r
        return "".join(ans)

