# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"
    #print (problem.getStartState())
    #print (problem.getSuccessors(problem.getStartState()))
    DFS_fringe_stack = util.Stack()     #import the data structrue stack from the util.py, it will store the expand path of the BFS search until the pacman reach the goal state
    #load the start state of the problem
    #a search node has position,actions,cost
    DFS_fringe_stack.push((problem.getStartState(),[],0))  #def push(self,item):the DFS algorithms will return a action list, so we create a list here
    #since we use graph search, we need to create a closed set to avoid the repetitive state
    closed = set() # we can use for x in set to quickly find whether the current state is in the closed
    while (DFS_fringe_stack.isEmpty() != True):
        state,actions,cost = DFS_fringe_stack.pop()  #Pop the most recently pushed search node from the stack
        #first check the current node is the goal
        if problem.isGoalState(state):
            #print (actions)
            return actions
        else:# else check the state if in the closed
            if state in closed:  #first check whether the current state is in the closed, if in, skip the current node 
                continue
            else:
                closed.add(state) #add the new fringe into the stack top
        next = problem.getSuccessors(state) #next is a list 
        i=0
        while i < len(next):
            next_successor = next[i][0]
            next_direction = next[i][1]
            new_actions = actions + [next_direction]
            new_cost  = cost + 1
            if next_successor not in closed :
                DFS_fringe_stack.push((next_successor,new_actions,new_cost)) #push the successor search node to the stack top
            i = i+1
    util.raiseNotDefined()

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    BFS_fringe_queue = util.Queue()     #import the data structrue queue from the util.py, it will store the expand path of the BFS search until the pacman reach the goal state
    #load the start state of the problem
    #a search node has position,actions,cost
    BFS_fringe_queue.push((problem.getStartState(),[],0))  #def push(self,item):the BFS algorithms will return a action list, so we create a list here
    #since we use graph search, we need to create a closed set to avoid the repetitive state
    closed = set() # we can use for x in set to quickly find whether the current state is in the closed
    while (BFS_fringe_queue.isEmpty() != True):
        state,actions,cost = BFS_fringe_queue.pop()  #Pop the most recently pushed search node from the queue
        #first check the current node is the goal
        if problem.isGoalState(state):
            return actions
        else:# else check the state if in the closed
            if state in closed:  #first check whether the current state is in the closed, if in, skip the current node 
                continue
            else:
                closed.add(state) #add the new fringe into the queue head
        next = problem.getSuccessors(state) #next is a list 
        i=0
        while i < len(next):
            next_successor = next[i][0]
            next_direction = next[i][1]
            new_actions = actions + [next_direction]
            new_cost  = cost + 1
            if next_successor not in closed :
                BFS_fringe_queue.push((next_successor,new_actions,new_cost)) #push the successor search node to the stack top
            i = i+1
   
    util.raiseNotDefined()

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    #in DFS and BFS, the cost is regarded as 1 per action to complete the algorithm
    #but in the Uniform cost search, we need to find the cost from the 'stepCost'
    #so we can use a data structure priority queue to pop the min cost search node first
    #let the new cost as the priority of the current search node
    UCS_fringe_priority_queue = util.PriorityQueue()     #import the data structrue priority_queue from the util.py, it will store the expand path of the UCS search until the pacman reach the goal state
    #load the start state of the problem
    #a search node has position,actions,cost
    #since the priority queue has :def push(self, item, priority), so we let cost as priority
    UCS_fringe_priority_queue.push((problem.getStartState(),[]),0)  #for a initial priority queue , the priority is default 0
    #since we use graph search, we need to create a closed set to avoid the repetitive state
    closed = set() # we can use for x in set to quickly find whether the current state is in the closed
    while (UCS_fringe_priority_queue.isEmpty() != True):
        state,actions= UCS_fringe_priority_queue.pop()  #Pop the most recently pushed search node from the priority queue
        #first check the current node is the goal
        if problem.isGoalState(state):
            return actions
        else:# else check the state if in the closed
            if state in closed:  #first check whether the current state is in the closed, if in, skip the current node 
                continue
            else:
                closed.add(state) #add the new fringe into the queue head
        next = problem.getSuccessors(state) #next is a list 
        i=0
        while i < len(next):
            next_successor = next[i][0]
            next_direction = next[i][1]
            next_cost = next[i][2]
            new_actions = actions + [next_direction]
            new_priority  = next_cost + problem.getCostOfActions(actions) #getCostOfActions returns the total cost of a particular sequence of actions.
            if next_successor not in closed :
                UCS_fringe_priority_queue.push((next_successor,new_actions),new_priority) #push the successor search node to the stack top
            i = i+1
    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    #A* still need the min cost, wo we still choose the priority queue data structure
    #in this algorithm ,we add the heuristic on the base of the UCS
    #let the heuristic + getCost as the new priority
    astar_fringe_priority_queue = util.PriorityQueue()     #import the data structrue priority_queue from the util.py, it will store the expand path of the UCS search until the pacman reach the goal state
    #load the start state of the problem
    #a search node has position,actions,cost
    #since the priority queue has :def push(self, item, priority), so we let cost as priority
    astar_fringe_priority_queue.push((problem.getStartState(),[]),0)  #for a initial priority queue , the priority is default 0
    #since we use graph search, we need to create a closed set to avoid the repetitive state
    closed = set() # we can use for x in set to quickly find whether the current state is in the closed
    while (astar_fringe_priority_queue.isEmpty() != True):
        state,actions= astar_fringe_priority_queue.pop()  #Pop the most recently pushed search node from the priority queue
        #first check the current node is the goal
        if problem.isGoalState(state):
            return actions
        else:# else check the state if in the closed
            if state in closed:  #first check whether the current state is in the closed, if in, skip the current node 
                continue
            else:
                closed.add(state) #add the new fringe into the queue head
        next = problem.getSuccessors(state) #next is a list 
        i=0
        while i < len(next):
            next_successor = next[i][0]
            next_direction = next[i][1]
            next_cost = next[i][2]
            new_actions = actions + [next_direction]
            new_priority  = next_cost + problem.getCostOfActions(actions)+heuristic(next_successor,problem = problem) #heuristic function estimates the cost from the current state to the nearest goal
            if next_successor not in closed :
                astar_fringe_priority_queue.push((next_successor,new_actions),new_priority) #push the successor search node to the priorityqueue 
            i = i+1
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
