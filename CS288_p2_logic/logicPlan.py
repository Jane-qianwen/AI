# logicPlan.py
# ------------
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
In logicPlan.py, you will implement logic planning methods which are called by
Pacman agents (in logicAgents.py).
"""

import util
import sys
import logic
import game


pacman_str = 'P'
ghost_pos_str = 'G'
ghost_east_str = 'GE'
pacman_alive_str = 'PA'

class PlanningProblem:
    """
    This class outlines the structure of a planning problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the planning problem.
        """
        util.raiseNotDefined()

    def getGhostStartStates(self):
        """
        Returns a list containing the start state for each ghost.
        Only used in problems that use ghosts (FoodGhostPlanningProblem)
        """
        util.raiseNotDefined()
        
    def getGoalState(self):
        """
        Returns goal state for problem. Note only defined for problems that have
        a unique goal state such as PositionPlanningProblem
        """
        util.raiseNotDefined()

def tinyMazePlan(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def sentence1():
    """Returns a logic.Expr instance that encodes that the following expressions are all true.
    
    A or B
    (not A) if and only if ((not B) or C)
    (not A) or (not B) or C
    """
    "*** YOUR CODE HERE ***"
    A = logic.Expr('A')
    B = logic.Expr('B')
    C = logic.Expr('C')
    A_or_B = logic.disjoin(A,B)
    Two = ~A % logic.disjoin(~B,C)
    Three = logic.disjoin(~A,~B,C)
    return logic.conjoin(A_or_B,Two,Three)
    util.raiseNotDefined()

def sentence2():
    """Returns a logic.Expr instance that encodes that the following expressions are all true.
    
    C if and only if (B or D)
    A implies ((not B) and (not D))
    (not (B and (not C))) implies A
    (not D) implies C
    """
    "*** YOUR CODE HERE ***"
    A = logic.Expr('A')
    B = logic.Expr('B')
    C = logic.Expr('C')
    D = logic.Expr('D')
    B_and_notC = logic.conjoin(B,~C)
    One = C % logic.disjoin(B,D)
    Two = A >> logic.conjoin(~B,~D)
    Three = ~B_and_notC >> A
    Four = ~D >> C
    return logic.conjoin(One,Two,Three,Four)
    util.raiseNotDefined()

def sentence3():
    """Using the symbols WumpusAlive[1], WumpusAlive[0], WumpusBorn[0], and WumpusKilled[0],
    created using the logic.PropSymbolExpr constructor, return a logic.PropSymbolExpr
    instance that encodes the following English sentences (in this order):

    The Wumpus is alive at time 1 if and only if the Wumpus was alive at time 0 and it was
    not killed at time 0 or it was not alive and time 0 and it was born at time 0.

    The Wumpus cannot both be alive at time 0 and be born at time 0.

    The Wumpus is born at time 0.
    """
    "*** YOUR CODE HERE ***"
    #know the use of the PropSymbolExpr from logic.py
    Alive_1 = logic.PropSymbolExpr("WumpusAlive",1)
    #print (Alive_1)
    Alive_0 = logic.PropSymbolExpr("WumpusAlive",0)
    Born_0 = logic.PropSymbolExpr("WumpusBorn",0)
    Killed_0 = logic.PropSymbolExpr("WumpusKilled",0)
    One = Alive_1 % logic.disjoin(logic.conjoin(Alive_0,~Killed_0),logic.conjoin(~Alive_0,Born_0))
    Two = ~logic.conjoin(Alive_0,Born_0)
    Three = Born_0
    return logic.conjoin(One,Two,Three)
    util.raiseNotDefined()

def findModel(sentence):
    """Given a propositional logic sentence (i.e. a logic.Expr instance), returns a satisfying
    model if one exists. Otherwise, returns False.
    """
    "*** YOUR CODE HERE ***"
    #print sentence
    cnf_form = logic.to_cnf(sentence) #convert the input to cnf form
    if_have_model = logic.pycoSAT(cnf_form) #return a model, if not have a model ,return false
    return if_have_model
    util.raiseNotDefined()

def atLeastOne(literals) :
    """
    Given a list of logic.Expr literals (i.e. in the form A or ~A), return a single 
    logic.Expr instance in CNF (conjunctive normal form) that represents the logic 
    that at least one of the literals in the list is true.
    >>> A = logic.PropSymbolExpr('A');
    >>> B = logic.PropSymbolExpr('B');
    >>> symbols = [A, B]
    >>> atleast1 = atLeastOne(symbols)
    >>> model1 = {A:False, B:False}
    >>> print logic.pl_true(atleast1,model1)
    False
    >>> model2 = {A:False, B:True}
    >>> print logic.pl_true(atleast1,model2)
    True
    >>> model3 = {A:True, B:True}
    >>> print logic.pl_true(atleast1,model2)
    True
    """
    "*** YOUR CODE HERE ***"
    #print (literals),check the input,[A,B,C,D]
    #at least one means in the result of disjoin is True or False in CNF
    return logic.disjoin(literals)
    util.raiseNotDefined()


def atMostOne(literals) :
    """
    Given a list of logic.Expr literals, return a single logic.Expr instance in 
    CNF (conjunctive normal form) that represents the logic that at most one of 
    the expressions in the list is true.
    """
    "*** YOUR CODE HERE ***"
    #at most one is true means conjoin any two literal always return false
    #so get each literal's not , disjoin any two, the conjoin should be true
    #print (literals)
    #print (len(literals))
    result  = []
    len_of_literal = len(literals)
    for i in range(len_of_literal):
        j = i + 1
        while j<len_of_literal :
            a = ~literals[i]
            b = ~literals[j]
            current_conjoin = logic.disjoin(a,b)
            result.append(current_conjoin)
            j = j + 1 #######do not forget to add this line
    return logic.conjoin(result)
    util.raiseNotDefined()


def exactlyOne(literals) :
    """
    Given a list of logic.Expr literals, return a single logic.Expr instance in 
    CNF (conjunctive normal form)that represents the logic that exactly one of 
    the expressions in the list is true.
    """
    "*** YOUR CODE HERE ***"
    #we can use the atMostOne and atLeastOne to get exactlyOne
    #when atMostOne and asLeastOne all true , then it means exactlyOne is true
    #the conjoin of two CNF is still a CNF form
    at_most_one = atMostOne(literals)
    at_least_one = atLeastOne(literals)
    return logic.conjoin(at_most_one,at_least_one)
    util.raiseNotDefined()


def extractActionSequence(model, actions):
    """
    Convert a model in to an ordered list of actions.
    model: Propositional logic model stored as a dictionary with keys being
    the symbol strings and values being Boolean: True or False
    Example:
    >>> model = {"North[3]":True, "P[3,4,1]":True, "P[3,3,1]":False, "West[1]":True, "GhostScary":True, "West[3]":False, "South[2]":True, "East[1]":False}
    >>> actions = ['North', 'South', 'East', 'West']
    >>> plan = extractActionSequence(model, actions)
    >>> print plan
    ['West', 'South', 'North']
    """
    "*** YOUR CODE HERE ***"
    #print (model)
    #print (actions)
    #model is stored as a dictionary, we can visit it by key and value
    #we should find the action by time and action should be in the [S,N,W,E]
    plan = []
    unsorted_plan = []
    for pl in model.keys() :
        #print (logic.PropSymbolExpr.parseExpr(pl))
        parse_pl = logic.PropSymbolExpr.parseExpr(pl) # eg:('West','0')
        action = parse_pl[0]
        #print (model[pl])
        if action in actions and model[pl] == True :
            time  = parse_pl[1]
            unsorted_plan.append((action,int(time))) #important.time is stored as string in the model
    #print (unsorted_plan)
    #print (len(unsorted_plan))
    for i in range(len(unsorted_plan)) :
        for j in range(len(unsorted_plan)) :
            if unsorted_plan[j][1] == i :
                plan.append(unsorted_plan[j][0])
                break
            else :
                continue
    return plan
    util.raiseNotDefined()


def pacmanSuccessorStateAxioms(x, y, t, walls_grid):
    """
    Successor state axiom for state (x,y,t) (from t-1), given the board (as a 
    grid representing the wall locations).
    Current <==> (previous position at time t-1) & (took action to move to x, y)
    """
    "*** YOUR CODE HERE ***"
    #print (walls_grid) a 5*5 grid, 25 boolean grid
    #to get the current state, we need to know the up,down,left and right's state
    prev_possible_state = [] #store a possible state when there is no wall
    current_state = logic.PropSymbolExpr(pacman_str,x,y,t)
    up_coordinate = walls_grid[x][y+1]
    down_coordinate = walls_grid[x][y-1]
    left_coordinate = walls_grid[x-1][y]
    right_coordinate= walls_grid[x+1][y]
    if up_coordinate == False : ##up isn't a wall
        up_position = logic.PropSymbolExpr(pacman_str,x,y+1,t-1)
        up_to_now_action = logic.PropSymbolExpr('South',t-1)
        prev_possible_state.append(logic.conjoin(up_position,up_to_now_action))
    if down_coordinate == False :##down isn't a wall
        down_position = logic.PropSymbolExpr(pacman_str,x,y-1,t-1)
        down_to_now_action = logic.PropSymbolExpr('North',t-1)
        prev_possible_state.append(logic.conjoin(down_position,down_to_now_action))
    if left_coordinate == False : ##left isn't a wall
        left_position = logic.PropSymbolExpr(pacman_str,x-1,y,t-1)
        left_to_now_action = logic.PropSymbolExpr('East',t-1)
        prev_possible_state.append(logic.conjoin(left_position,left_to_now_action))
    if right_coordinate == False :##right isn't a wall
        right_position = logic.PropSymbolExpr(pacman_str,x+1,y,t-1)
        right_to_now_action = logic.PropSymbolExpr('West',t-1)
        prev_possible_state.append(logic.conjoin(right_position,right_to_now_action))
    #return the current <==> (previous position at time t-1) & (took action to move to x, y)
    return current_state % logic.disjoin(prev_possible_state)

def positionLogicPlan(problem):
    """
    Given an instance of a PositionPlanningProblem, return a list of actions that lead to the goal.
    Available actions are game.Directions.{NORTH,SOUTH,EAST,WEST}
    Note that STOP is not an available action.
    """
    walls = problem.walls
    width, height = problem.getWidth(), problem.getHeight()
    "*** YOUR CODE HERE ***"
    #print(problem.getGoalState()) (1,1)
    #print (walls) 4*4grid
    #print (problem.getStartState()) (2,2)
    actions  = ['North','South','West','East'] #all of the actions will in this list
    start_x,start_y = problem.getStartState()
    start_expression = logic.PropSymbolExpr(pacman_str,start_x,start_y,0)
    end_x,end_y = problem.getGoalState()
    Manhattan_distance = abs(end_x - start_x) + abs(end_y - start_y) - 1
    #idea:
    #1.get the knowledge base from conjoin several expressions
    #2.give the kb to the SAT solver to get a model : findmodel()
    #3.use this model to extract action sequences : extractActionsSequences()
    #since the test won't be more than 50 and it must take an action with step increasing
    #so we can add an state into kb each step with step increasing
    if start_x == end_x and start_y == end_y :##if start is the goal
        return None
    #or we need to use kb to get the actions
    t = 0
    kb = start_expression #now we only know start expression in kb, but we can add one expression each step
    while t <= 50 :
        #when time steps >= Manhattan distance between the start and the goal(least path),we try to find model
        take_action_t = []
        for action in actions :
            take_action_t.append(logic.PropSymbolExpr(action,t))
        #we can only take exactly one action at a time
        action_t = exactlyOne(take_action_t)
        kb = logic.conjoin(kb,action_t) #update the kb
        #then we need to get all the not-wall position state
        for x in range(1,width+1) :
            for y in range(1,height+1) :
                if walls[x][y] == False :
                    #print('hjhjhjhj')
                    if t == 0 : #we only get one start position
                        if (x,y) != (start_x,start_y) : #we should mark other position as NOT at t = 0
                            kb = logic.conjoin(kb,~(logic.PropSymbolExpr(pacman_str,x,y,0)))
                    #get all successorStateAxiom
                    if (x,y) != (start_x,start_y) :  #while this (x,y) can be the next position
                        successor_state_axiom = pacmanSuccessorStateAxioms(x,y,t+1,walls) #whether we can get to the position at t+1
                        kb = logic.conjoin(kb,successor_state_axiom)
        #each time we assume next step is our goal
        end_expression = logic.PropSymbolExpr(pacman_str,end_x,end_y,t+1)
        #now we add the end_espression as the last new thing in kb
        start_to_end = logic.conjoin(kb,end_expression)
        #check whether we can try find a model when we get the least step
        if t >= Manhattan_distance :
            model_exists = findModel(start_to_end)
            if model_exists != False : ##we can find a model for current kb
                return extractActionSequence(model_exists,actions)
            else:
                t = t + 1
        else:
            t = t + 1
    util.raiseNotDefined()


def foodLogicPlan(problem):
    """
    Given an instance of a FoodPlanningProblem, return a list of actions that help Pacman
    eat all of the food.
    Available actions are game.Directions.{NORTH,SOUTH,EAST,WEST}
    Note that STOP is not an available action.
    """
    walls = problem.walls
    width, height = problem.getWidth(), problem.getHeight()

    "*** YOUR CODE HERE ***"
    actions = ['North','South','West','East']
    #print (problem.getStartState())#((2,3),food_grid)
    start_x,start_y = problem.getStartState()[0]
    #print (start_x,start_y)
    start_expression = logic.PropSymbolExpr(pacman_str,start_x,start_y,0)
    food = problem.getStartState()[1] #it is a boolean table for whether there is food or not
    #like the idea for Q5, we need to explore the kb until we find a model so that our kb is true
    #we need to explore kb with step increasing
    t = 0 #initial t , which is not more than 50
    find_way_kb = start_expression #at first, our find_way_kb just contains the start state
    #Q6 we should consider the food and the path, when all the food are eaten and path is valid,then return the action list
    while t <= 50 :
        take_action_t = []
        for action in actions :
            take_action_t.append(logic.PropSymbolExpr(action,t))
        #we can only take exactly one action at a time
        action_t = exactlyOne(take_action_t)
        find_way_kb = logic.conjoin(find_way_kb,action_t) #update the kb
        #then we need to get all the not-wall position state
        #what's more, we need to check the food can be eaten or not
        food_kb = [] #initially,each t we should have a new food_kb cause the prev is wrong or the first is unknown
        food_possible_eaten = [] #to store all the possible eaten food state
        for x in range(1,width+1) :
            for y in range(1,height+1) :
                if walls[x][y] == False :
                    if t == 0 : #we only get one start position
                        if (x,y) != (start_x,start_y) : #we should mark other position as NOT at t = 0
                            find_way_kb = logic.conjoin(find_way_kb,~(logic.PropSymbolExpr(pacman_str,x,y,0)))
                    #get all successorStateAxiom
                    if True :  #while this (x,y) can be the next position confused
                        successor_state_axiom = pacmanSuccessorStateAxioms(x,y,t+1,walls) #whether we can get to the position at t+1
                        find_way_kb = logic.conjoin(find_way_kb,successor_state_axiom)
        #next is the difference compared to the Q5: 
        #we need to consider the food grid
                if food[x][y] == True : #this position has food
                    i = 0
                    while i <= t + 1:#for the current state, we can get the good in the time region from 0~t+1
                        possible_food_state = logic.PropSymbolExpr(pacman_str,x,y,i)
                        food_possible_eaten.append(possible_food_state)
                        i = i + 1
                    #may be we can or can't get the food within t
                    #we should use disjoin to explore that right eat_food_state
                    whether_eat_food = logic.disjoin(food_possible_eaten)
                    #we update the food kb
                    food_kb.append(whether_eat_food)
                    food_possible_eaten = [] #reset to empty for the next food position
        #now food_kb have all the food possible state at (0~t+1), conjoin them to check
        food_kb = logic.conjoin(food_kb)
        total_kb = logic.conjoin(food_kb,find_way_kb)
        model_exists = findModel(total_kb)
        if model_exists != False : #exist a model
            return extractActionSequence(model_exists,actions)
        else :
            t = t + 1
    util.raiseNotDefined()


# Abbreviations
plp = positionLogicPlan
flp = foodLogicPlan

# Some for the logic module uses pretty deep recursion on long expressions
sys.setrecursionlimit(100000)
    