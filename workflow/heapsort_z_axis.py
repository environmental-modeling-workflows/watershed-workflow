#	A pit removal tool for a TIN dataset
#	By: Benjamin Liebersohn, Anne Berres, Ethan Coons
#	Oak Ridge National Laboratory, Oak Ridge, Tennessee, USA
#
#	MIT Open Source Lisence: Copyright: 2019, Oak Ridge National Laboratory
#	Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
#	The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

#	THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
#	Disclaimer: This software has been authored by employees of UT-Battelle, under contract DE-AC05-00OR22725 with the US Department of Energy. The authors would also like to acknowledge the financial and intellectual support for this research by the Integrated Assessment Research Program of the US Department of Energy's Office of Science, Biological and Environmental Research and by the Department of Energy Office of Policy.  The US government retains and the publisher, by accepting the article for publication, acknowledges that the US government retains a nonexclusive, paid-up, irrevocable, worldwide license to publish or reproduce the published form of this manuscript, or allow others to do so, for US government purposes. DOE will provide public access to these results of federally sponsored research in accordance with the DOE Public Access Plan (http://energy.gov/downloads/doe-public-access-plan).
#
#	input: the user-defined outlet, TIN.vtk file
#	output: an unstructured surface mesh which has filled-in pits

in_file = open('test.vtk', 'r')
section = "<none>" #default placeholder
pointDict = {}

outletID = 3 #dummmy outlet for testing

#	computeElevation() sorts a list of sets [ID, z-axis] by z-axis, in ascending order.
def computeElevation(pointDict, i):
	zSorted = {} #this list contains ID and elevations sorted by elevation (z-axis)
	zUnsorted = []

	zUnsorted.append(pointDict[i]['z'])
	try:
		zSorted = sorted(zUnsorted)
	except KeyError:
		print("Elevation not Found!")
	return zSorted

#	computeNeighbors() computes the connectivity for a given point ID in the network
def computeNeighbors(myPoint, myCell):
	"""
	Computes the neighbors of a point, given the cell, and adds them to the pointDict entry for this point.
	"""
	for cellPoint in myCell:
		if cellPoint != myPoint and cellPoint not in pointDict[myPoint]["neighbors"]:
			pointDict[myPoint]["neighbors"].append(cellPoint)

#	getNeighbors() returns neighboring points for a given point ID
def getNeighbors(myPoint):
	return pointDict[myPoint]["neighbors"]

#	checkSectionType() determines if we are in a valid section in a .vtk file
def checkSectionType(mySection, line, in_file): 
	if 'POINTS' in line:

		mySection = "POINTS"

	elif 'CELLS' in line: #exiting Cells section
		mySection = "CELLS"
	elif "CELL_TYPE" in line:
		mySection = "<none>"			
	else:
		pass
	return(mySection)

#	readSection() reads a line and matches keywords that indicate the section
def readSection(mySection,line):
	if mySection == "POINTS":
		readPoint(line)
	elif mySection == "CELLS":
		readCell(line)
	else: 
		pass

#	readPoint() Assigns a point ID, reads coordinates stores z-axis value for each point. 
def readPoint(line): 
	pointID = len(pointDict)
	point = line.split(" ")
	for i in range(len(point)):
		point[i] = int(point[i])
	pointDict[pointID] = {"neighbors": [], "x": point[0], "y": point[1], "z": point[2]}

#	readCell() reads from CELLS, which is a set of n point IDs [0,1,2,3,4,5,[...],n]
def readCell(line):
	cell = line.split(" ")
	cell = cell[1:4]
	for i in range(len(cell)):
		cell[i] = int(cell[i])
	for pointID in cell:
		computeNeighbors(pointID, cell)

#	getNodeDict returns a list which is a copy of z-axis values from pointDict
def getNodeDict(pointDict, point):
	elevationDict[point] = pointDict[point]["z"]
	return #elevationDict test no return, then test return?

#	getNeighborElevation returns the z-axis values from neighboring nodes
def getNeighborElevation(neighbor):
	neighborElevation = sortedList[neighbor]
	return neighborElevation




#	read in file by section: CELLS or POINTS
for line in in_file:
	newSection = checkSectionType(section, line, in_file)
	if newSection != section:
		section = newSection
		continue
	readSection(newSection, line)
	
elevationDict = {}
for point in pointDict: #could this loop be merged with line 100? currently
	getNodeDict(pointDict, point)

sortedList = (sorted(elevationDict.items(), key=lambda x:x[1])) #how can this be filled if we merged lines 95 and line 114?

sortedElevation = sorted(elevationDict.items(), key=lambda x:x[1])

for point in pointDict: #print neighbors and their elevations sorted by elevation. can this be merged with line 95? see line 101 for issues
	neighborDict = pointDict[point]["neighbors"]
	nodeElevation = getNodeDict(pointDict, point)

	print("node elevation:",nodeElevation)

	print("node", point,"neighbors")
	for item in range(len(neighborDict)):
		myNeighbor = (getNeighborElevation(neighborDict[item]))
		myNeighborID = myNeighbor[0]
		myNeighborElevation = myNeighbor[1]

		#if neighborElevation[1] < nodeElevation:
		print("myNeighborID:",myNeighborID,", myNeighborElevation:",myNeighborElevation) #testing comparisons
	continue

print("complete")
