# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        
        return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 1)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game

          gameState.isWin():
            Returns whether or not the game state is a winning state

          gameState.isLose():
            Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        infinity = float("inf") #it is an infinity value
        #since muti agents, the pacman index is always = 0, ghosts index always >=1
        number_of_ghosts = gameState.getNumAgents()  #Returns the total number of agents in the game
        def is_terminal_state(gameState,depth): #when we can call evaluation function
          if gameState.isWin() == True or gameState.isLose() == True or depth == self.depth :
            return True
          else :
            return False
        #def value is the main minmax idea, it DFS the minmax tree by recursive
        #it will compute each successors' min and max,from root to the leaf to get the value
        #pacman's depth is 0,to recursive, we need to recurse by depth
        #since we could not know the actual depth is how many, so we increase the depth until self.depth==depth
        def value(gameState,depth,agent_index) :
          if agent_index == number_of_ghosts : # we have traverse to the last agent
            if is_terminal_state(gameState,depth) == True : #if the state is a terminal state,return the state's utility
              return self.evaluationFunction(gameState)
            else :#we are not get the real depth,increase the depth , start again from the tree root
              return value(gameState,depth+1,0)
          else : #consider minmax
            if agent_index == 0: # for pacman, we should get the max value
              return max_value(gameState,depth)
            else : #for ghosts, we should get the min value
              return min_value(gameState,agent_index,depth)
        def max_value(gameState,depth): #the max value shoule get the max value from the ghost 1
          v = -infinity #initial v
          legal_actions = gameState.getLegalActions(0) #get the legal actions of pacman
          if (len(legal_actions) == 0) : #no more actions
            return self.evaluationFunction(gameState)
          else :
            for legal_action in legal_actions : #traverse all successors value so that we can fnd the max
              v = max(v,value(gameState.generateSuccessor(0,legal_action),depth,1))
            return v
        def min_value(gameState,agent_index,depth) : #the min value is to get the min value from the next agent
          v= infinity #initial v
          legal_actions = gameState.getLegalActions(agent_index) #get the legal actions of the ghost
          if (len(legal_actions) == 0) : #no more actions
            return self.evaluationFunction(gameState)
          else :
            for legal_action in legal_actions : #traverse all successors value so that we can fnd the min from the next agent
              v = min(v,value(gameState.generateSuccessor(agent_index,legal_action),depth,agent_index+1))
            return v
        #find which of the pacman action leads to the max value
        pacman_legal_actions = gameState.getLegalActions(0)
        v = -infinity
        for action in pacman_legal_actions : #find the max value
          v = max(v,value(gameState.generateSuccessor(0,action),1,1))
        for action in pacman_legal_actions : #find the max value action
          current_v = value(gameState.generateSuccessor(0,action),1,1)
          if current_v == v : #it is the result action,return it
            return action
          else:
            continue
        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        infinity = float("inf") #it is an infinity value
        #alpha is the best max choice on the path from root to leaf
        #beta is the best min choice on the path from root to leafinfinity = float("inf") #it is an infinity value
        #since muti agents, the pacman index is always = 0, ghosts index always >=1
        number_of_ghosts = gameState.getNumAgents()  #Returns the total number of agents in the game
        def is_terminal_state(gameState,depth): #when we can call evaluation function
          if gameState.isWin() == True or gameState.isLose() == True or depth == self.depth : #here we have reach our own depth
            return True
          else :
            return False
        #in alpha-beta algorithm,we need to pruning, 
        #so we need to keep track of the alpha and beta on the base of the min-max 
        def minmax(gameState,depth,agent_index,alpha,beta):
          if is_terminal_state(gameState,depth) == True: #should call return evaluation
            return (self.evaluationFunction(gameState),None)
          else:
            if agent_index == 0: #the agent is pacman, we should get the max_value
              return max_value(gameState,depth,agent_index,alpha,beta)
            else: #the agent is ghost, we should get the min_value
              return min_value(gameState,depth,agent_index,alpha,beta)
        def max_value(gameState,depth,agent_index,alpha,beta):
          if is_terminal_state(gameState,depth) == True: #should call return evaluation
            return (self.evaluationFunction(gameState),None)
          value = -infinity #initial value
          result_action = None #initial result_action
          v = (value,result_action) #the max_value() find the max value and the the action to get the max value
          legal_actions = gameState.getLegalActions(agent_index) #get all the legal actions
          if len(legal_actions) == 0:
            return (self.evaluationFunction(gameState),None) #no more action can get
          for action in legal_actions:
            if minmax(gameState.generateSuccessor(agent_index,action),depth,agent_index+1,alpha,beta)[0] > v[0]: # we choose the larger one
              v = (minmax(gameState.generateSuccessor(agent_index,action),depth,agent_index+1,alpha,beta)[0],action)#update the new v
            if v[0] > beta: #algorithm from the lectures
              return v
            alpha = max(alpha,minmax(gameState.generateSuccessor(agent_index,action),depth,agent_index+1,alpha,beta)[0])
          return v
        def min_value(gameState,depth,agent_index,alpha,beta):
          if is_terminal_state(gameState,depth) == True: #should call return evaluation
            return (self.evaluationFunction(gameState),None)
          value = infinity #initial value
          result_action = None #initial result_action
          v = (value,result_action)
          legal_actions = gameState.getLegalActions(agent_index) #get all the legal actions
          if len(legal_actions) == 0:
            return (self.evaluationFunction(gameState),None) #no more action can get
          for action in legal_actions:
            ######
            ######
            #from the if else , we can found that the minmax for pacman 's cost is very high, if l do not save the minmax() of pacman , it always time out 
            #it cost me a lot of time to find this problem
            if agent_index != number_of_ghosts - 1 : #we are not got the leaf, we still find min value from the next agent
              if minmax(gameState.generateSuccessor(agent_index,action),depth,agent_index+1,alpha,beta)[0] < v[0] : #we choose the smaller one
                v = (minmax(gameState.generateSuccessor(agent_index,action),depth,agent_index+1,alpha,beta)[0],action) #update the v
              if minmax(gameState.generateSuccessor(agent_index,action),depth,agent_index+1,alpha,beta)[0] < alpha:
                return minmax(gameState.generateSuccessor(agent_index,action),depth,agent_index+1,alpha,beta)
              beta = min(beta,v[0]) #algorithm from the lectures
            else:#the next we will got the leaf, so we have traversed all the depth, we should call the pacman to minmax() to find the max value
              current_minmax = minmax(gameState.generateSuccessor(agent_index,action),depth+1,0,alpha,beta)
              if current_minmax[0] < v[0] : #we choose the smaller one
                v= (current_minmax[0],action) #update the v
              if current_minmax[0] < alpha:
                return current_minmax
              beta = min(beta,v[0]) #algorithm from the lectures
          return v
        # now we can call the main step
        # initially 
        alpha = -infinity
        beta = infinity
        #pacman is 0, tree root is 0
        #return the action is return [1]
        #this is a easier way to get the result_action
        return minmax(gameState,0,0,alpha,beta)[1] #call pacman to start from tree root to the tree leaf
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        infinity = float("inf") #it is an infinity value
        #since muti agents, the pacman index is always = 0, ghosts index always >=1
        number_of_ghosts = gameState.getNumAgents()  #Returns the total number of agents in the game
        def is_terminal_state(gameState,depth): #when we can call evaluation function
          if gameState.isWin() == True or gameState.isLose() == True or depth == self.depth :
            return True
          else :
            return False
        #on the base of the min max, we replace min part with average
        #pacman's depth is 0,to recursive, we need to recurse by depth
        #since we could not know the actual depth is how many, so we increase the depth until self.depth==depth
        def value(gameState,depth,agent_index) :
          if agent_index == number_of_ghosts : # we have traverse to the last agent
            if is_terminal_state(gameState,depth) == True : #if the state is a terminal state,return the state's utility
              return self.evaluationFunction(gameState)
            else :#we are not get the real depth,increase the depth , start again from the tree root
              return value(gameState,depth+1,0)
          else : #consider minmax
            if agent_index == 0: # for pacman, we should get the max value
              return max_value(gameState,depth)
            else : #for ghosts, we should get the min value
              return exp_value(gameState,agent_index,depth)
        def max_value(gameState,depth): #the max value shoule get the max value from the ghost 1
          v = -infinity #initial v
          legal_actions = gameState.getLegalActions(0) #get the legal actions of pacman
          if (len(legal_actions) == 0) : #no more actions
            return self.evaluationFunction(gameState)
          else :
            for legal_action in legal_actions : #traverse all successors value so that we can fnd the max
              v = max(v,value(gameState.generateSuccessor(0,legal_action),depth,1))
            return v
        def exp_value(gameState,agent_index,depth) : #the average value is to get the min value from the averge agent
          v= 0 #initial v
          legal_actions = gameState.getLegalActions(agent_index) #get the legal actions of the ghost
          if (len(legal_actions) == 0) : #no more actions
            return self.evaluationFunction(gameState)
          else :
            for legal_action in legal_actions : #traverse all successors value so that we can fnd the min from the next agent
              v = v + value(gameState.generateSuccessor(agent_index,legal_action),depth,agent_index+1)
            v = v / len(legal_actions) # this means the probability is 1/len(legal_actions)
            return v
        #find which of the pacman action leads to the max value
        pacman_legal_actions = gameState.getLegalActions(0)
        v = -infinity
        for action in pacman_legal_actions : #find the max value
          v = max(v,value(gameState.generateSuccessor(0,action),1,1))#let index = 1 and depth = 1 so that the min max tree can be DFS in my value function
        for action in pacman_legal_actions : #find the max value action
          current_v = value(gameState.generateSuccessor(0,action),1,1)
          if current_v == v : #it is the result action,return it
            return action
          else:
            continue
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 4).

      DESCRIPTION: 
      <write something here so we know what you did>
      In this part, we should add more features which is efficient for a better evaluation.
      Here l use the hint , which is the linear combination of the improtant features,
      and combine those features by multiplying them by different values and adding the results together.
      l think the following features are important :
      1.Manhattan distance to the closet ghost
      2.All the rest food position information
      3.Manhattan distance to the closet food
      4.The ghosts' scared time
    """
    "*** YOUR CODE HERE ***"
    current_position = currentGameState.getPacmanPosition() #get the current pacman position
    #now we need to get the rest food information
    rest_food_matrix = currentGameState.getFood() #getFood() will return a grid of boolean food indicator variables
    #since the getGood() returns an matrix informantion, we need to convert the array into a list
    #print (rest_food_array)
    #there is a asList() function in the Class Grid,which can convert the boolean into the matix position and put that into a list
    rest_food_position_list = rest_food_matrix.asList()
    #print(rest_food_list)
    #now to find the closest food
    closest_food_manhattan_distance = 999999999999999 #initial very large
    for food_position in rest_food_position_list:
      closest_food_manhattan_distance = min(closest_food_manhattan_distance,util.manhattanDistance(current_position, food_position))
    #now closest_food_manhattan_distance is the manhattab=n distance from current position to the nearest food
    #now we need to find the closest distance to the Ghost
    ghosts = currentGameState.getGhostStates() #get all the ghosts so that we can find the closest one
    closest_ghost_manhattan_distance = 9999999999999999 #initial very large
    for ghost in ghosts: #traverse all the ghosts
      closest_ghost_manhattan_distance = min(closest_ghost_manhattan_distance,util.manhattanDistance(current_position, ghost.getPosition()))
    #now closest_ghost_manhattan_distance is found
    #now we need to find the ghosts' scary time
    scared_time_sum = 0 #initial as 0
    for ghost in ghosts: # find each ghost's scary time
      scared_time_sum = scared_time_sum + ghost.scaredTimer #add the current ghost scary time if it has
    #the final score should includ the nearset food,nearset ghost and the ghosts scary time sum
    #the more close the closest food is , the food part get higher scores
    closest_food_part_score = 1.0 / float(closest_food_manhattan_distance)
    #the more close the closest ghost is , the ghost part get lower scores
    closest_ghost_part_score = -closest_ghost_manhattan_distance
    scary_part_score = scared_time_sum
    final_score = currentGameState.getScore() + closest_food_part_score + closest_ghost_part_score + scary_part_score
    return final_score
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction