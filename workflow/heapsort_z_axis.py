in_file = open('test.vtk', 'r')
elevation_tuples = [] #elevations are used for priority. See: z_sorted below
REMOVED = '<removed-node>' #placeholder for a removed node
section = 0 #improve?
counter = 0 #we count from 0 always
outlet = [0, -1000] #node which is user defined

for line in in_file:
	if 'CELLS' in line:
		print('Entering CELLS section 0')
		section = 0
		continue
	if 'POINTS' in line:
		print('Entering POINTS section 1')
		section = 1
		continue
	if section == 1:
		words = line.split(" ")
		elevation_tuples.append([int(counter), int(words[2])]) #groups tuples by (id, z_axis) and converts to ints
		print(elevation_tuples)
		counter += 1
		continue

z_sorted=(sorted(elevation_tuples, key=lambda x: x[1])) #save a sorted array of tuples, sorting by the second value in a pair (a z-axis vaue). The location of z-axis is set by x[1] which would be referring to a set with addresses [0,1] 

if (z_sorted[0] != outlet): #checks to see if there's a big ole pit, or if something's off
	print('WARNING: a pit appears to be below the outlet. This is not conforming.')
	print(outlet)
	print(z_sorted[0])

print('Continuing...')
