import pytest
import numpy as np

import watershed_workflow.datasets as dsets


def profile(s):
    return {'name':s}


@pytest.fixture
def dc():
    return dsets.Dataset(profile('profile1'), np.array([2,3]), {'a' : np.array([2,3])})


@pytest.fixture
def s():
    dc1 = dsets.Dataset(profile('profile1'), np.array([2,3]), {'a' : np.array([2,3])})
    dc2 = dsets.Dataset(profile('profile2'), np.array([4,5]), {'b' : np.array([4,5])})

    s = dsets.State()
    s.collections.append(dc1)
    s.collections.append(dc2)
    return s
    

def test_empty():
    s = dsets.State()
    assert(len(s) == 0)

    dc = dsets.Dataset(profile('profile'), np.array([2,3]))
    assert(len(dc) == 0)

def test_collection(dc):
    assert('a' in dc)
    assert(len(dc) == 1)

    dc['b'] = np.array([4,5])
    assert('b' in dc)
    assert(len(dc) == 2)

    a = dc['a']
    assert(isinstance(a, dsets.Data))
    assert(all(a.data == [2,3]))
    b = dc['b']
    assert(all(b.data == [4,5]))

    assert(dc.can_contain(a))

    another = dsets.Data(profile('profile2'), np.array([2,3]), {'c' : np.array([6,7])})
    assert(not dc.can_contain(another))

    another2 = dsets.Data(profile('profile1'), np.array([4,5]), {'c' : np.array([6,7])})
    assert(not dc.can_contain(another2))

def test_state(s):
    assert('a' in s)
    assert('b' in s)
    assert(len(s) == 2)
    s['c'] = (profile('profile2'), np.array([4,5]), np.array([6,7]))
    assert(len(s) == 3)
    assert('c' in s)
    assert(len(s.collections) == 2)

    c2 = s['c']
    assert(c2.profile['name'] == 'profile2')
    assert(all(c2.times == np.array([4,5])))
    assert(all(c2.data == np.array([6,7])))
    del c2

    a2 = s['a']
    assert(a2.profile['name'] == 'profile1')
    assert(all(a2.times == [2,3]))
    assert(all(a2.data == [2,3]))
    del a2

    assert(len(s) == 3)

    # mutability of entities within the profile
    s['c'].profile['name'] = 'profile3'
    assert(s['b'].profile['name'] == 'profile3')
    assert(s['a'].profile['name'] == 'profile1')


def test_update(s):
    s1 = dsets.State()
    s1['a'] = (profile('profile1'), np.array([2,3]), np.array([0,1,2]))
    s1['b'] = (profile('profile2'), np.array([3,4]), np.array([0,1,2]))

    s2 = dsets.State()
    s2['c'] = (profile('profile1'), np.array([2,3]), np.array([0,1,2]))
    s2['d'] = (profile('profile3'), np.array([2,3]), np.array([0,1,2]))

    s1.update(s2)
    assert(len(s1) == 4)
    assert('c' in s1)
    assert('d' in s1)
    assert(len(s1.collections) == 3)


    
