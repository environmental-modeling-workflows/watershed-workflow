from dataclasses import dataclass, field
from typing import Any
import itertools 

@dataclass(order=True)
class PrioritizedItem:
    priority: int
    item: Any=field(compare=False)
pq = []                         # list of entries arranged in a heap
entry_finder = {}               # mapping of points to entries
REMOVED = '<removed-point>'      # placeholder for a removed point
counter = itertools.count()     # unique sequence count

def add_point(point, priority=0):
    'Add a new point or update the priority of an existing point'
    if point in entry_finder:
        remove_point(point)
    count = next(counter)
    entry = [priority, count, point]
    entry_finder[point] = entry
    heappush(pq, entry)

def remove_point(point):
    'Mark an existing point as REMOVED.  Raise KeyError if not found.'
    entry = entry_finder.pop(point)
    entry[-1] = REMOVED

def pop_point():
    'Remove and return the lowest priority point. Raise KeyError if empty.'
    while pq:
        priority, count, point = heappop(pq)
        if point is not REMOVED:
            del entry_finder[point]
            return point
    raise KeyError('pop from an empty priority queue')

#open the file, read to array 'list'
test = open('numbers.vtk', 'r')
list = {}
for line in test:
	words = line.split(" ")
	list[words[0]] = words[3]
	print(list)


#outletPoint_id: name of the point which is designated as an outlet.
#section id is not necessarially in order.
in_file = open('test.vtk', 'r')
z_value = [] #z_value is used for priority queue
z_finder = {} #maps a node to a z_value
REMOVED = '<removed-node>' #placeholder for a removed node
section = 0
counter = 0

for line in in_file:
	if 'CELLS' in line:
		print('Entering CELLS section 0')
		section = 0
		continue
	if 'POINTS' in line:
		print('Entering POINTS section 1')
		section = 1
	if section == 1:
		words = line.split(" ")
		z_value = counter, words[2]
		print(z_value)
		counter += 1


