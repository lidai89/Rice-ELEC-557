import re, util
import heapq, collections, re, sys, time, os, random

############################################################
# Problem 1a: UCS test case

# Return an instance of util.SearchProblem.
# You might find it convenient to use
# util.createSearchProblemFromString.
def createUCSTestCase(n):
    # BEGIN_YOUR_CODE (around 5 lines of code expected)
    space=" "
    description=""
    for x in range(0, n):
        description+="A"+space+"B"+str(x)+space+"1"+"\n"
        description+="B"+str(x)+space+"C"+space+"20"+"\n"
    description+="A"+space+"B"+str(n)+space+"2"+"\n"
    description+="B"+str(n)+space+"C"+space+"2"+"\n"
    smallProblem = util.createSearchProblemFromString("A", "C",description)
    return smallProblem
    raise Exception("Not implemented yet")
    # END_YOUR_CODE

############################################################
# Problem 1b: A-star search

# Takes the SearchProblem |problem| you're trying to solve and a |heuristic|
# (which is a function that maps a state to an estimate of the cost to the
# goal).  Returns another search problem |newProblem| such that running uniform
# cost search on |newProblem| is equivalent to running A* on |problem| with
# |heuristic|.
def astarReduction(problem, heuristic):
    start=problem.startState()
    class NewSearchProblem(util.SearchProblem):
#         Please refer to util.SearchProblem to see the functions you need to
#         overried.
#         BEGIN_YOUR_CODE (around 9 lines of code expected)

#        
        # END_YOUR_CODE
        def startState(self): return problem.startState()
        def isGoal(self, state): return problem.isGoal(state)
#        def succAndCost(self, state): return [('West', state-1, 1), ('East', state+1, 2)]
        def succAndCost(self, state): 
            newitems=[(items[0],items[1],items[2]+heuristic(items[1])-heuristic(state)) for items in problem.succAndCost(state)]
            return newitems
    newProblem = NewSearchProblem()

    return newProblem
    raise Exception("Not implemented yet")

# Implements A-star search by doing a reduction.
class AStarSearch(util.SearchAlgorithm):
    def __init__(self, heuristic):
        self.heuristic = heuristic

    def solve(self, problem):
        # Reduce the |problem| to |newProblem|, which is solved by UCS.
        newProblem = astarReduction(problem, self.heuristic)
        algorithm = util.UniformCostSearch()
        algorithm.solve(newProblem)

        # Copy solution back
        self.actions = algorithm.actions
        if algorithm.totalCost != None:
            self.totalCost = algorithm.totalCost + self.heuristic(problem.startState())
        else:
            self.totalCost = None
        self.numStatesExplored = algorithm.numStatesExplored

############################################################
# Problem 2b: Delivery

class DeliveryProblem(util.SearchProblem):
    # |scenario|: delivery specification.
    def __init__(self, scenario):
        self.scenario = scenario

    # Return the start state.
    def startState(self):
        # BEGIN_YOUR_CODE (around 1 line of code expected)
        return (self.scenario.truckLocation,tuple(['ready'] * self.scenario.numPackages))
        raise Exception("Not implemented yet")
        # END_YOUR_CODE

    # Return whether |state| is a goal state or not.
    def isGoal(self, state):
        # BEGIN_YOUR_CODE (around 2 lines of code expected)
        return state[1]==tuple(['done'] * self.scenario.numPackages) and state[0]==self.startState()[0]
        raise Exception("Not implemented yet")
        # END_YOUR_CODE

    # Return a list of (action, newState, cost) tuples corresponding to edges
    # coming out of |state|.
    def succAndCost(self, state):
        # Hint: Call self.scenario.getNeighbors((x,y)) to get the valid neighbors
        # at that location. In order for the simulation code to work, please use
        # the exact strings 'Pickup' and 'Dropoff' for those two actions.
        # BEGIN_YOUR_CODE (around 18 lines of code expected)
        neighbors=self.scenario.getNeighbors(state[0])
        r,c=state[0]
        itemstatus=state[1]
        moveaction=[]
        for neighborpair in neighbors:
            moveaction.append((neighborpair[0],(neighborpair[1],itemstatus),state[1].count('transit')+1))
        for package, loc in enumerate(self.scenario.pickupLocations):
            if itemstatus[package] =='ready' and loc==(r,c):
                newitemstatus=list(itemstatus)
                newitemstatus[self.scenario.pickupLocations.index((r,c))] ='transit'
                newitemstatus=tuple(newitemstatus)
                moveaction.append(('Pickup',((r,c),newitemstatus),0))
        for package, loc in enumerate(self.scenario.dropoffLocations):
            if itemstatus[package] =='transit'and loc==(r,c):
                neweritemstatus=list(itemstatus)            
                neweritemstatus[self.scenario.dropoffLocations.index((r,c))] ='done'
                neweritemstatus=tuple(neweritemstatus)
                moveaction.append(('Dropoff',((r,c),neweritemstatus),0))
            
        return moveaction
        raise Exception("Not implemented yet")
        # END_YOUR_CODE

############################################################
# Problem 2c: heuristic 1


# Return a heuristic corresponding to solving a relaxed problem
# where you can ignore all barriers and not do any deliveries,
# you just need to go home
def createHeuristic1(scenario):
    def heuristic(state):
        # BEGIN_YOUR_CODE (around 2 lines of code expected)
        harvestine=abs(scenario.truckLocation[0]-state[0][0])+abs(scenario.truckLocation[1]-state[0][1])
        return harvestine
        raise Exception("Not implemented yet")
        # END_YOUR_CODE
    return heuristic

############################################################
# Problem 2d: heuristic 2

# Return a heuristic corresponding to solving a relaxed problem
# where you can ignore all barriers, but
# you'll need to deliver the given |package|, and then go home
def createHeuristic2(scenario, package):
    def heuristic(state):
        pick_drop_dis=abs(scenario.pickupLocations[package][0]-scenario.dropoffLocations[package][0])+abs(scenario.pickupLocations[package][1]-scenario.dropoffLocations[package][1])
        drop_return_dis=abs(scenario.truckLocation[0]-scenario.dropoffLocations[package][0])+abs(scenario.truckLocation[1]-scenario.dropoffLocations[package][1])        
        result=0
        if state[1][package]=='done':
            func=createHeuristic1(scenario)
            result=(state[1].count('transit')+1)*func(state)
        if state[1][package]=='ready':
            result=(state[1].count('transit')+2)*pick_drop_dis+drop_return_dis+(state[1].count('transit')+1)*abs(scenario.pickupLocations[package][0]-state[0][0])+(state[1].count('transit')+1)*abs(scenario.pickupLocations[package][1]-state[0][1])
        if state[1][package]=='transit':
            result=drop_return_dis+(state[1].count('transit')+1)*abs(scenario.dropoffLocations[package][0]-state[0][0])+(state[1].count('transit')+1)*abs(scenario.dropoffLocations[package][1]-state[0][1])
        # BEGIN_YOUR_CODE (around 11 lines of code expected)
        return result
        raise Exception("Not implemented yet")
        # END_YOUR_CODE
    return heuristic

############################################################
# Problem 2e: heuristic 3

# Return a heuristic corresponding to solving a relaxed problem
# where you will delivery the worst(i.e. most costly) |package|,
# you can ignore all barriers.
# Hint: you might find it useful to call
# createHeuristic2.
def createHeuristic3(scenario):
    # BEGIN_YOUR_CODE (around 5 lines of code expected)
#    distance=[]
#    r1=0
#    r2=0
#    c1=0
#    c2=0
#    for loc_pickup in scenario.pickupLocations:
#        r1,c1=loc_pickup
#        r2,c2=scenario.dropoffLocations[scenario.pickupLocations.index((r1,c1))]
#        distance.append((r1-r2)**2+(c1-c2)**2)
#    def heuristic(state):
#        maxdistance=0
#        maxindex=0
#        func=createHeuristic1(scenario)
#        result=(state[1].count('transit')+1)*func(state)
#        if state[1]!=tuple(['done']*scenario.numPackages):
#            for package in range(0,scenario.numPackages):
#                if state[1][package]!='done' and distance[package]>=maxdistance:
#                    maxindex=package
#                    maxdistance=distance[package]
#                    func2=createHeuristic2(scenario,maxindex);
#                    result=func2(state);
    
    def heuristic(state):
        maxindex=0
        maxheuristic=0
        for package in range(0,scenario.numPackages):
            func=createHeuristic2(scenario,package)
            if maxheuristic<func(state) and state[1][package]=='ready':
                maxindex=package
                maxheuristic=func(state)
            if state[1][package]=='transit':
                maxheuristic=func(state)
        return maxheuristic
    
    # END_YOUR_CODE
    return heuristic
    raise Exception("Not implemented yet")
