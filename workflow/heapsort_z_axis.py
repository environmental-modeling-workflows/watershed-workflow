in_file = open('test.vtk', 'r')
section = "<none>" #default placeholder
pointDict = {}
outletID = 3
counter = 0
visited = [outletID] #starts with outlet as visited
pits = []

 #dummmy outlet for testing

#    computeNeighbors() computes the connectivity for a given point ID in the network
def computeNeighbors(myPoint, myCell):
    """
    Computes the neighbors of a point, given the cell, and adds them to the pointDict entry for this point.
    """
    for cellPoint in myCell:
        if cellPoint != myPoint and cellPoint not in pointDict[myPoint]["neighbors"]:
            pointDict[myPoint]["neighbors"].append(cellPoint)

#    getNeighbors() returns neighboring points for a given point ID
def getNeighbors(myPoint):
    return pointDict[myPoint]["neighbors"]

#    checkSectionType() determines if we are in a valid section in a .vtk file
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

#    readSection() reads a line and matches keywords that indicate the section
def readSection(mySection,line):
    if mySection == "POINTS":
        readPoint(line)
    elif mySection == "CELLS":
        readCell(line)
    else: 
        pass

#    readPoint() Assigns a point ID, reads coordinates stores z-axis value for each point. 
def readPoint(line): 
    pointID = len(pointDict)
    point = line.split(" ")
    for i in range(len(point)):
        point[i] = int(point[i])
    pointDict[pointID] = {"neighbors": [], "x": point[0], "y": point[1], "z": point[2]}

#    readCell() reads from CELLS, which is a set of n point IDs [0,1,2,3,4,5,[...],n]
def readCell(line):
    cell = line.split(" ")
    cell = cell[1:4]
    for i in range(len(cell)):
        cell[i] = int(cell[i])
    for pointID in cell:
        computeNeighbors(pointID, cell)
    
#    getPointDict returns a list which is a copy of z-axis values from pointDict
def getPointDict(pointDict, point):
    elevationDict[point] = pointDict[point]["z"]
    return #this is a template: Consider working this function back in

#    getNeighborElevation returns the z-axis values from neighboring points
def getNeighborElevation(neighbor):
    neighborElevation = sortedList[neighbor]
    return neighborElevation

def traverse(counter, outlet, sortedElevation, pointDict): #needs work!
    neighbors = pointDict[outlet]["neighbors"]
    next = neighbors[counter+1]
    for item in neighbors:
        i=0
    #    counter = counter+1
        print("neighbors: ", neighbors, "item", item)
        nextPointID = sortedElevation[next][0] #this is the traversal
        if outlet not in visited:
            counter = counter+1
            print("outlet", outlet, "not in visited")
            print("visited: ",visited)
            print("sortedElevation: ",sortedElevation)
            pits.append(outlet)
            print("nextpointID",nextPointID)
            traverse(next,nextPointID, sortedElevation, pointDict) #recursion!!! wowie!
        else:
            print("outlet is in visited:", visited)
            for item in neighbors:
                i += 1
                visited.append(neighbors[i])
            traverse(counter, nextPointID, sortedElevation, pointDict) 
          #  counter = counter+1

      #  for neighborID in neighbors:
       #     elevation = pointDict[neighborID]["z"]
          #  print("neighbors: ",neighborID, elevation)
        #    continue

#    read in file by section: CELLS or POINTS
for line in in_file:
    newSection = checkSectionType(section, line, in_file)
    if newSection != section:
        section = newSection
        continue
    readSection(newSection, line)
    
elevationDict = {}
for point in pointDict: #could this loop be merged with line 100? currently
    getPointDict(pointDict, point)

sortedList = (sorted(elevationDict.items(), key=lambda x:x[1])) #how can this be filled if we merged lines 95 and line 114?

sortedElevation = sorted(elevationDict.items(), key=lambda x:x[1]) #are sortedList and sortedElevation merge-able?
print (sortedElevation)

for point in pointDict:
    traverse(point, outletID, sortedElevation, pointDict)
    continue


print("complete")
