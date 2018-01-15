# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        iteration_times = self.iterations
        all_mdp_states = self.mdp.getStates()
        #in total we iteration iteration_times
        i = 0
        while i < iteration_times :
          #every iteration we creat a tmp counter to store the values of the current iteration
          tmp_value = util.Counter()
          #traverse each state for each iterations
          for state in all_mdp_states:
            #consider the terminal state
            if self.mdp.isTerminal(state) == True :
              current_value = 0
            else :
              #find the most possible action in the current given state
              most_possible_action = self.computeActionFromValues(state)
              #get the Q_value according to the most possible action
              current_value = self.computeQValueFromValues(state,most_possible_action)
              #update the self.values for the current given state
              tmp_value[state] = current_value
          #self.values should store the values of the last iteration
          self.values = tmp_value
          i = i + 1

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        #initial Q_value
        Q_value = 0
        #get the transition states and probability
        transition_information = self.mdp.getTransitionStatesAndProbs(state,action)
        #traverse all the transition
        for next_state , probability in transition_information:
          next_state_value = self.getValue(next_state)
          next_state_reward = self.mdp.getReward(state,action,next_state)
          Q_value = Q_value + probability * (next_state_reward + self.discount * next_state_value)
        return Q_value

        #util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        all_possible_actions = self.mdp.getPossibleActions(state)
        #the case at the terminal state, return None
        if self.mdp.isTerminal(state) :
          return None
        else :
          #initial the Q_value and policy
          max_Q_value = float('-inf')
          policy = None
          #traverse all the possible action to find the best policy
          for action in all_possible_actions :
            current_Q_value = self.computeQValueFromValues(state,action)
            if current_Q_value > max_Q_value :
              max_Q_value = current_Q_value
              policy = action
            else :
              continue
          return policy

        #util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        #get all the MDP states
        MDP_states = self.mdp.getStates()
        state_number = len(MDP_states)
        iteration_times = self.iterations
        i = 0
        while i < iteration_times :
          #for each iteration, we need to compute the new value of the state
          pick_index = i % state_number
          state = MDP_states[pick_index]
          #then get the most possible action 
          action = self.computeActionFromValues(state)
          #then get the value of the action
          if self.mdp.isTerminal(state) == True : ####in the terminal state
            current_value = 0
          else :
            current_value = self.computeQValueFromValues(state,action)
          self.values[state] = current_value
          i = i + 1


class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        #get all the MDP states
        MDP_states = self.mdp.getStates()
        #initial a dictionary for the predecessor
        predecessors = {}
        #each predecessor of a state should be stored in a list
        for state in MDP_states :
          predecessors[state] = set()
        #Compute predecessors of all states.
        for state in MDP_states :
          all_possible_actions = self.mdp.getPossibleActions(state)
          for possible_action in all_possible_actions :
            #get the transition states and probability
            transition_information = self.mdp.getTransitionStatesAndProbs(state,possible_action)
            #traverse all the transition
            for next_state , probability in transition_information:
              if probability > 0:
                predecessors[next_state].add(state)
        #Initialize an empty priority queue.
        priority_queue = util.PriorityQueue()
        #For each non-terminal state s, do:
        #Find the absolute value of the difference between the current value of s in self.values and the highest Q-value across all possible actions from s (this represents what the value should be); call this number diff. Do NOT update self.values[s] in this step.
        #Push s into the priority queue with priority -diff (note that this is negative). We use a negative because the priority queue is a min heap, but we want to prioritize updating states that have a higher error.
        for state in MDP_states :
          if self.mdp.isTerminal(state) == False :
            #we need to find the highest Q-value for the current state
            max_Q_value = float('-inf')
            for possible_action in self.mdp.getPossibleActions(state) :
              current_Q_value = self.getQValue(state,possible_action)
              if current_Q_value > max_Q_value :
                max_Q_value = current_Q_value
              else :
                continue
            difference = abs(self.values[state] - max_Q_value)
            priority_queue.push(state,-difference)
        #For iteration in 0, 1, 2, ..., self.iterations - 1, do:
        iteration_times = self.iterations
        i = 0
        while i < iteration_times :
          #If the priority queue is empty, then terminate.
          if priority_queue.isEmpty() == True :
            break
          else :
            #Pop a state s off the priority queue.
            #Update s's value (if it is not a terminal state) in self.values.
            state = priority_queue.pop()
            max_s_Q_value = float('-inf')
            for possible_action in self.mdp.getPossibleActions(state) :
              current_s_Q_value = self.getQValue(state,possible_action)
              if current_s_Q_value > max_s_Q_value :
                max_s_Q_value = current_s_Q_value
              else :
                continue
            self.values[state] = max_s_Q_value
            #For each predecessor p of s, do:
            for predecessor in predecessors[state] :
              max_p_Q_value = float('-inf')
              for possible_action in self.mdp.getPossibleActions(predecessor) :
                current_p_Q_value = self.getQValue(predecessor,possible_action)
                if current_p_Q_value > max_p_Q_value :
                  max_p_Q_value = current_p_Q_value
                else :
                  continue
              difference = abs(self.values[predecessor] - max_p_Q_value)
              #If diff > theta, push p into the priority queue with priority -diff
              if difference > self.theta :
                priority_queue.update(predecessor,-difference)
          i = i + 1



        

