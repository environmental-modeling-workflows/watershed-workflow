"""A very simple tree implementation.

This is derived from Aldo Cortesi's code, and includes the license to
that code in the source.

Acquired from: https://github.com/cortesi/tinytree/blob/master/tinytree.py

Changes made to introduce several new iteration patterns which are
useful in this application, and to remove a bunch of functions not
used here.

"""

#
# The MIT License
#
# Copyright (c) 2007 Aldo Cortesi
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to
# deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
# sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.

import sys, itertools, copy, unicodedata


def _isStringLike(anobj):
    try:
        # Avoid succeeding expensively if anobj is large.
        anobj[:0] + ''
    except:
        return 0
    else:
        return 1


def _isSequenceLike(anobj):
    if not hasattr(anobj, "next"):
        if _isStringLike(anobj):
            return 0
        try:
            anobj[:0]
        except:
            return 0
    return 1


class _MyList(list):
    """This is defined to extend list's API to match that of sortedcontainers.SortedList"""
    def add(self, arg):
        return self.append(arg)

    def update(self, args):
        return self.extend(args)


class Tree(object):
    """
        A simple implementation of an ordered tree
    """
    ListType = _MyList
    
    def __init__(self, children=None):
        """
            :children A nested list specifying a tree of children
        """
        self.children = self.__class__.ListType()
        if children:
            self.addChildrenFromList(children)
        self.parent = None

    def addChildrenFromList(self, children):
        """
            Add children to this node.

            :children A nested list specifying a tree of children
        """
        skip = True
        v = list(zip(itertools.chain([None], children), itertools.chain(children, [None])))
        for i in v:
            if skip:
                skip = False
                continue
            self.addChild(i[0])
            if _isSequenceLike(i[1]):
                i[0].addChildrenFromList(i[1])
                skip = True

    def addChild(self, node):
        """
            Add a child to this node.

            :child A Tree object
        """
        if not isinstance(node, Tree):
            s = "Invalid tree specification: %s is not a Tree object." % repr(node)
            raise ValueError(s)
        node.register(self) # must do this first for add() to be sorted
        self.children.add(node)

    def register(self, parent):
        """
            Called after a node has been added to a parent.

            :child A Tree object
        """
        self.parent = parent

    @property
    def parent_child_index(self):
        """
            Return the index of this node in the parent child list, based on
            object identity.
        """
        if self.parent is None:
            raise ValueError("Can not retrieve index of a node with no parent.")
        lst = [id(i) for i in self.parent.children]
        return lst.index(id(self))

    def remove(self):
        """
            Remove this node from its parent. Returns the index this node had
            in the parent child list.
        """
        idx = self.parent_child_index
        del self.parent.children[idx:idx + 1]
        self.parent = None
        return idx

    def clear(self):
        """
            Clear all the children of this node. Return a list of the removed
            children.
        """
        n = self.children[:]
        for i in n:
            i.remove()
        return n

    def replace(self, *nodes):
        """
            Replace this node with a sequence of other nodes. This is
            equivalent to deleting this node from the child list, and then
            inserting the specified sequence in its place.

            :nodes A sequence of Tree objects
        """
        parent = self.parent
        idx = self.remove()
        parent.children[idx:idx] = nodes
        for i in nodes:
            i.register(parent)

    def giveChildren(self, node):
        """
            Gives my children to node.

            :node A Tree object
        """
        children = self.clear()
        for i in children:
            node.addChild(i)
        return node

    def inject(self, node):
        """
            Inject node between self and self.children.
        """
        self.giveChildren(node)
        self.addChild(node)
    
    def reparent(self, node):
        """
            Inserts a node between the current node and its parent. Returns the
            specified parent node.

            :node A Tree object
        """
        self.replace(node)
        node.addChild(self)
        return node

    def isDescendantOf(self, node):
        """
            Returns true if this node lies on the path to the root from the
            specified node.

            :node A Tree object
        """
        return (self in node.pathToRoot())

    def isSiblingOf(self, node):
        """
            Returns true if this node is a sibling of the specified node.

            :node A Tree object
        """
        return (self in node.siblings)

    @property
    def siblings(self):
        """
            Generator yielding all siblings of this node, not including this node.
        """
        if self.parent:
            for i in self.parent.children:
                if i is not self:
                    yield i

    def preOrder(self):
        """
            Generates nodes in PreOrder.
        """
        yield self
        # Take copy to make this robust under modification
        for i in self.children[:]:
            for j in i.preOrder():
                yield j

    def postOrder(self):
        """
            Generates nodes in PostOrder.
        """
        # Take copy to make this robust under modification
        for i in self.children[:]:
            for j in i.postOrder():
                yield j
        yield self

    def prePostInBetweenOrder(self):
        """Generates a tuple, (flag,node) in pre and post order, but
            also with self in between all consective children.

            The flag indicates how many times we have touched this
            node.

        """
        count = 0
        yield count, self
        count += 1
        for i in self.children[:]:
            for j in i.prePostInBetweenOrder():
                yield j
            yield count, self
            count += 1

    def breadthFirstOrder(self):
        """
            Generates nodes in a breadth-first ordering.
        """
        import queue
        q = queue.Queue()
        q.put(self)
        for obj in q.get():
            for child in obj.children:
                q.put(child)
            yield obj

    @property
    def leaf_nodes(self):
        """Generator for all leaves of the tree."""
        for it in self.preOrder():
            if len(it.children) == 0:
                yield it
            
    def __len__(self):
        """
            Number of nodes in this tree, including the root.
        """
        return sum(1 for i in self.preOrder())

    def to_networkx(self, hash):
        import networkx
        G = networkx.DiGraph()
        for node in self.preOrder():
            for child in node.children:
                G.add_edge(hash(child), hash(node))
        return G


def constructFromList(lst):
    """
        :lst a nested list of Tree objects

        Returns a list consisting of the nodes at the base of each tree.  Trees
        are constructed "bottom-up", so all parent nodes for a particular node
        are guaranteed to exist when "addChild" is run.
    """
    heads = []
    for i, val in enumerate(lst):
        if _isSequenceLike(val):
            if i == 0 or _isSequenceLike(lst[i - 1]):
                raise ValueError("constructFromList: Invalid list.")
            lst[i - 1].addChildrenFromList(val)
        else:
            heads.append(val)
    return heads
