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


# acquired from: https://github.com/cortesi/tinytree/blob/master/tinytree.py

import sys, itertools, copy, unicodedata

def _isStringLike(anobj):
    try:
        # Avoid succeeding expensively if anobj is large.
        anobj[:0]+''
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


class Tree(object):
    """
        A simple implementation of an ordered tree
    """
    def __init__(self, children = None):
        """
            :children A nested list specifying a tree of children
        """
        self.children = []
        if children:
            self.addChildrenFromList(children)
        self.parent = None

    def addChildrenFromList(self, children):
        """
            Add children to this node.

            :children A nested list specifying a tree of children
        """
        skip = True
        v = list(zip(
            itertools.chain([None], children),
            itertools.chain(children, [None])
        ))
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
            s = "Invalid tree specification: %s is not a Tree object."%repr(node)
            raise ValueError(s)
        self.children.append(node)
        node.register(self)

    def register(self, parent):
        """
            Called after a node has been added to a parent.

            :child A Tree object
        """
        self.parent = parent

    def index(self):
        """
            Return the index of this node in the parent child list, based on
            object identity.
        """
        if not self.parent:
            raise ValueError("Can not retrieve index of a node with no parent.")
        lst = [id(i) for i in self.parent.children]
        return lst.index(id(self))

    def remove(self):
        """
            Remove this node from its parent. Returns the index this node had
            in the parent child list.
        """
        idx = self.index()
        del self.parent.children[idx:idx+1]
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

    def inject(self, node):
        """
            Inserts a node between the current node and its children. Returns the
            specified parent node.

            :node A Tree object
        """
        for i in self.children[:]:
            i.remove()
            node.addChild(i)
        self.clear()
        self.addChild(node)
        return node

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
        return (self in node.siblings())

    def siblings(self):
        """
            Generator yielding all siblings of this node, including this
            node itself.
        """
        if not self.parent:
            yield self
        else:
            for i in self.parent.children:
                yield i

    def pathToRoot(self):
        """
            Generator yielding all objects on the path from this node to the
            root of the tree, including this node itself.
        """
        itm = self
        while 1:
            yield itm
            if itm.parent is not None:
                itm = itm.parent
            else:
                break

    def pathFromRoot(self):
        """
            Generator yielding all nodes on the path to this node from the
            root of the tree, including this node itself.
        """
        l = list(self.pathToRoot())
        for i in reversed(l):
            yield i

    def getRoot(self):
        """
            Return the topmost node in the tree.
        """
        for i in self.pathToRoot():
            pass
        return i

    def preOrder(self):
        """
            Return a list of subnodes in PreOrder.
        """
        yield self
        # Take copy to make this robust under modification
        for i in self.children[:]:
            for j in i.preOrder():
                yield j

    def postOrder(self):
        """
            Return a list of the subnodes in PostOrder.
        """
        # Take copy to make this robust under modification
        for i in self.children[:]:
            for j in i.postOrder():
                yield j
        yield self

    def _find(self, itr, *func, **kwargs):
        for i in itr:
            if kwargs:
                kwpass = False
                for k, v in list(kwargs.items()):
                    if hasattr(i, k):
                        if not getattr(i, k) == v:
                            break
                    else:
                        break
                else:
                    kwpass = True
            else:
                kwpass = True
            if kwpass:
                if all([x(i) for x in func]):
                    return i
        return None

    def findChild(self, *func, **kwargs):
        """
            Find the first child matching all specified selectors in a
            pre-order traversal of this node's subnodes. Return None if no
            matching object is found.

            :func A list of selector functions, that accept a node, and return
            a boolean.

            :kwargs A dictionary of attribute selectors. Checks that matching
            attributes exist, and that their values are equal to the specified
            values.
        """
        return self._find(self.preOrder(), *func, **kwargs)

    def findParent(self, *func, **kwargs):
        """
            Find the first node matching func in a traversal to the root of the
            tree. Return None if no matching object is found.

            :func A list of selector functions, that accept a node, and return
            a boolean.

            :kwargs A dictionary of attribute selectors. Checks that matching
            attributes exist, and that their values are equal to the specified
            values.
        """
        return self._find(
            itertools.islice(self.pathToRoot(), 1, None),
            *func,
            **kwargs
        )

    def findForwards(self, *func, **kwargs):
        """
            Search forwards in a preOrder traversal of the whole tree (not this
            node's subnodes). Return None if object not found.

            :func A list of selector functions, that accept a node, and return
            a boolean.

            :kwargs A dictionary of attribute selectors. Checks that matching
            attributes exist, and that their values are equal to the specified
            values.
        """
        itr = self.getRoot().preOrder()
        for i in itr:
            if i is self:
                break
        return self._find(itr, *func, **kwargs)

    def findBackwards(self, *func, **kwargs):
        """
            Search backwards in a preOrder traversal of the whole tree (not
            this node's subnodes). Return None if object not found.

            :func A list of selector functions, that accept a node, and return
            a boolean.

            :kwargs A dictionary of attribute selectors. Checks that matching
            attributes exist, and that their values are equal to the specified
            values.
        """
        # FIXME: Dreadfully inefficient...
        lst = list(self.getRoot().preOrder())
        lst.reverse()
        myIndex = lst.index(self)
        return self._find(lst[(myIndex+1):], *func, **kwargs)

    def getPrevious(self):
        """
            Find the previous node in the preOrder traversal of the tree.
        """
        return self.findBackwards(lambda x: 1)

    def getNext(self):
        """
            Find the next node in the preOrder traversal of the tree.
        """
        return self.findForwards(lambda x: 1)

    def getDepth(self):
        """
            Return the depth of this node, i.e. the number of nodes on the path
            to the root.
        """
        return len(list(self.pathToRoot()))

    def findAttr(self, attr, default=None):
        """
            Traverses the path to the root of the tree, looking for the
            specified attribute. If it is found, return it, else return default.

            :attr A string attribute name
            :default Arbitrary default return value
        """
        for i in self.pathToRoot():
            if hasattr(i, attr):
                return getattr(i, attr)
        return default

    def attrsToRoot(self, attr):
        """
            Traverses the path from this node to the root of the tree, and
            yields a value for each attribute. Nodes that do not have the
            attribute and attribute values that test false are ignored.

            :attr A string attribute name
        """
        lst = []
        for i in self.pathToRoot():
            v = getattr(i, attr, None)
            if v:
                yield v

    @staticmethod
    def treeProp(name):
        """
            Define a property whose value should be looked up on nodes between
            this node and the root, inclusive. Returns the first matching
            attribute. Raises ValueError if no matching attribute is found.

            :name Property name
        """
        def fget(self):
            if name in self.__dict__:
                return self.__dict__[name]
            else:
                if not self.parent:
                    raise ValueError("Property %s not defined."%name)
                return getattr(self.parent, name)
        def fset(self, value):
            self.__dict__[name] = value
        return property(fget, fset)

    def dump(self, outf=sys.stdout):
        """
            Dump a formatted representation of this tree to the specified file
            descriptor.

            :outf Output file descriptor.
        """
        for i in self.preOrder():
            s = "\t"*(i.getDepth()-1)
            s += unicodedata.normalize('NFKD', unicode(i)).encode('ascii','ignore')
            outf.write(s)
            outf.write("\n")

    def count(self):
        """
            Number of nodes in this tree, including the root.
        """
        return len(list(self.preOrder()))


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
            if i == 0 or _isSequenceLike(lst[i-1]):
                raise ValueError("constructFromList: Invalid list.")
            lst[i-1].addChildrenFromList(val)
        else:
            heads.append(val)
    return heads
