
class Trie:
    def __init__(self, words=[]):
        self.children = {}
        self.path = ""
        self.isComplete = 0
        for word in words:
            self.addWord(word)
    
    def __contains__(self, c):
        return c in self.children

    def __getitem__(self, c):
        return self.children[c]

    def __len__(self):
        return len(self.children)

    def hasChildren(self):
        return len(self.children) > 0

    def isLeaf(self):
        return len(self.children) == 0
    
    def addWord(self, word):
        node = self
        for i in range(len(word)):
            c = word[i]
            if c not in node.children:
                newNode = Trie()
                newNode.path = word[:i+1]
                node.children[c] = newNode
            node = node.children[c]
            if i == len(word)-1:
                node.isComplete += 1
    
    def searchWord(self, word, partial=False):
        node = self
        for c in word:
            if c not in node.children:
                return False
            node = node.children[c]
        return True if partial else not node.hasChildren()
    
    def getDictionary(self):
        words = [self.path]*self.isComplete
        if len(self.children) == 0:
            return words

        for c in self.children:
            words += self.children[c].getDictionary()

        return words

    def getLeafCount(self):
        if self.isLeaf():
            return self.isComplete

        count = self.isComplete
        for _, child in self.children.items():
            count += child.getLeafCount()
        return count

#https://leetcode.com/problems/path-with-maximum-gold/
class MapSum:

    def __init__(self):
        from Trie import Trie
        self.pairs = {}
        self.trie = Trie()

    def insert(self, key: str, val: int) -> None:
        if key not in self.pairs:
            self.trie.addWord(key)
        self.pairs[key] = val

    def sum(self, prefix: str) -> int:
        t = self.trie
        for c in prefix:
            if c in t.children:
                t = t.children[c]
            else:
                return 0

        s = 0
        for key in t.getDictionary():
            s += self.pairs[key]
        print(s)
        return s

from collections import defaultdict

#https://leetcode.com/problems/number-of-matching-subsequences/
def numMatchingSubseq(s: str, words) -> int:
    smap = {}
    for i in range(len(s)):
        if s[i] not in smap:
            smap[s[i]] = []
        smap[s[i]].append(i)

    def searchCeil(k, c):
        if c not in smap:
            return len(s)

        arr = smap[c]
        l,r = 0, len(arr)-1
        while l < r:
            mid = l + (r-l)//2
            if arr[mid] > k:
                r = mid
            else:
                l = mid + 1
        return len(s) if arr[l] <= k else arr[l]

    def search(trie, l):
        if l >= len(s):
            return 0

        count = 0
        if trie.isComplete > 0:
            count += trie.isComplete

        nextc = None
        for c, t in trie.children.items():
            count += search(t, searchCeil(l, c))
        return count

    trie = Trie(words)
    return search(trie, -1)

#https://leetcode.com/problems/number-of-matching-subsequences/discuss/329381/Python-Solution-With-Detailed-Explanation
def numMatchingSubseqLinear(S, words):
    word_dict = defaultdict(list)
    count = 0
    
    for word in words:
        word_dict[word[0]].append(word)            
    
    for char in S:
        words_expecting_char = word_dict[char]
        word_dict[char] = []
        for word in words_expecting_char:
            if len(word) == 1:
                # Finished subsequence! 
                count += 1
            else:
                word_dict[word[1]].append(word[1:])
    
    return count

#https://practice.geeksforgeeks.org/problems/phone-directory/0
def phoneDirectoryPrefixSearch(n, contact, s):
    res = []
    root = Trie(contact)
    for i in range(len(s)):
        if s[i] in root.children:
            root = root.children[s[i]]
            res.append(sorted(root.getDictionary()))
        else:
            res += [[0]]*(n-i)
            break
    return res

#https://leetcode.com/problems/palindrome-pairs/
#TODO: Incomplete
def palindromePairs(words):
    trie = Trie()
    for word in words:
        trie.addWord(word[::-1])

    def isPalindrome(s):
        i, j = 0, len(s)-1
        while i < j:
            if s[i] != s[j]:
                return False
            i += 1
            j -= 1
        return True

    print(trie.getDictionary())
    pairs = []
    for j, word in enumerate(words):
        node = trie
        for i, c in enumerate(word):
            print("char", c, "node:", node.path, node.children.keys())
            if c in node:
                node = node[c]
            else:
                break
        if node.isComplete and isPalindrome(word[i:]):
            pairs.append((j, words.index(node.path[::-1])))      
    return pairs

