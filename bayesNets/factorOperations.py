# factorOperations.py
# -------------------
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


from bayesNet import Factor
import operator as op
import util

def joinFactorsByVariableWithCallTracking(callTrackingList=None):


    def joinFactorsByVariable(factors, joinVariable):
        """
        Input factors is a list of factors.
        Input joinVariable is the variable to join on.

        This function performs a check that the variable that is being joined on 
        appears as an unconditioned variable in only one of the input factors.

        Then, it calls your joinFactors on all of the factors in factors that 
        contain that variable.

        Returns a tuple of 
        (factors not joined, resulting factor from joinFactors)
        """

        if not (callTrackingList is None):
            callTrackingList.append(('join', joinVariable))

        currentFactorsToJoin =    [factor for factor in factors if joinVariable in factor.variablesSet()]
        currentFactorsNotToJoin = [factor for factor in factors if joinVariable not in factor.variablesSet()]

        # typecheck portion
        numVariableOnLeft = len([factor for factor in currentFactorsToJoin if joinVariable in factor.unconditionedVariables()])
        if numVariableOnLeft > 1:
            print "Factor failed joinFactorsByVariable typecheck: ", factor
            raise ValueError, ("The joinBy variable can only appear in one factor as an \nunconditioned variable. \n" +  
                               "joinVariable: " + str(joinVariable) + "\n" +
                               ", ".join(map(str, [factor.unconditionedVariables() for factor in currentFactorsToJoin])))
        
        joinedFactor = joinFactors(currentFactorsToJoin)
        return currentFactorsNotToJoin, joinedFactor

    return joinFactorsByVariable

joinFactorsByVariable = joinFactorsByVariableWithCallTracking()


def joinFactors(factors):
    """
    Question 3: Your join implementation 

    Input factors is a list of factors.  
    
    You should calculate the set of unconditioned variables and conditioned 
    variables for the join of those factors.

    Return a new factor that has those variables and whose probability entries 
    are product of the corresponding rows of the input factors.

    You may assume that the variableDomainsDict for all the input 
    factors are the same, since they come from the same BayesNet.

    joinFactors will only allow unconditionedVariables to appear in 
    one input factor (so their join is well defined).

    Hint: Factor methods that take an assignmentDict as input 
    (such as getProbability and setProbability) can handle 
    assignmentDicts that assign more variables than are in that factor.

    Useful functions:
    Factor.getAllPossibleAssignmentDicts
    Factor.getProbability
    Factor.setProbability
    Factor.unconditionedVariables
    Factor.conditionedVariables
    Factor.variableDomainsDict
    """

    # typecheck portion
    setsOfUnconditioned = [set(factor.unconditionedVariables()) for factor in factors]
    if len(factors) > 1:
        intersect = reduce(lambda x, y: x & y, setsOfUnconditioned)
        if len(intersect) > 0:
            print "Factor failed joinFactors typecheck: ", factor
            raise ValueError, ("unconditionedVariables can only appear in one factor. \n"
                    + "unconditionedVariables: " + str(intersect) + 
                    "\nappear in more than one input factor.\n" + 
                    "Input factors: \n" +
                    "\n".join(map(str, factors)))


    "*** YOUR CODE HERE ***"
    #print factors
    #first we get all the unconditioned variables
    #if a var is conditioned in one of the factors, it is also conditioned unless it was an unconditioned variable in one of the argument factors 
    unconditioned_vars = [] #initial a list to store
    for factor in factors:
        #get current factor's unconditioned vars
        current_factor_unconditioned_vars = factor.unconditionedVariables()
        #traverse all the current_factor_unconditioned_vars,put them into unconditioned_vars without reprtition
        for current_factor_unconditioned_var in current_factor_unconditioned_vars :
            if current_factor_unconditioned_var in unconditioned_vars : #check reprtition
                continue
            else :
                unconditioned_vars.append(current_factor_unconditioned_var)
    #second,we get all the conditioned variables
    conditioned_vars = [] #initial a list to store
    for factor in factors:
        #get current factor's conditioned vars
        current_factor_conditioned_vars = factor.conditionedVariables()
        #traverse all the current_factor_unconditioned_vars,put them into conditioned_vars without reprtition
        for current_factor_conditioned_var in current_factor_conditioned_vars :
            if current_factor_conditioned_var in conditioned_vars : #check reprtition
                continue
            else :
                conditioned_vars.append(current_factor_conditioned_var)
    #remove repetition
    for unconditioned_var in unconditioned_vars:
        if unconditioned_var in conditioned_vars:
            conditioned_vars.remove(unconditioned_var)
    #get the variable domain dictionary
    var_domain_dict = factors[0].variableDomainsDict() #Retuns a copy of the variable domains in the factor
    #get the joined factor
    #def __init__(self, inputUnconditionedVariables, inputConditionedVariables, inputVariableDomainsDict):
    joined_factor = Factor(unconditioned_vars,conditioned_vars,var_domain_dict)
    #get probability for the each assighment
    for assignment in joined_factor.getAllPossibleAssignmentDicts():
        assignment_probability = 1#initial , to cascade multiply
        for factor in factors :
            #get all the multiply
            assignment_probability = assignment_probability * factor.getProbability(assignment)
        #set peobability
        joined_factor.setProbability(assignment,assignment_probability)
    return joined_factor
    #util.raiseNotDefined()


def eliminateWithCallTracking(callTrackingList=None):

    def eliminate(factor, eliminationVariable):
        """
        Question 4: Your eliminate implementation 

        Input factor is a single factor.
        Input eliminationVariable is the variable to eliminate from factor.
        eliminationVariable must be an unconditioned variable in factor.
        
        You should calculate the set of unconditioned variables and conditioned 
        variables for the factor obtained by eliminating the variable
        eliminationVariable.

        Return a new factor where all of the rows mentioning
        eliminationVariable are summed with rows that match
        assignments on the other variables.

        Useful functions:
        Factor.getAllPossibleAssignmentDicts
        Factor.getProbability
        Factor.setProbability
        Factor.unconditionedVariables
        Factor.conditionedVariables
        Factor.variableDomainsDict
        """
        # autograder tracking -- don't remove
        if not (callTrackingList is None):
            callTrackingList.append(('eliminate', eliminationVariable))

        # typecheck portion
        if eliminationVariable not in factor.unconditionedVariables():
            print "Factor failed eliminate typecheck: ", factor
            raise ValueError, ("Elimination variable is not an unconditioned variable " \
                            + "in this factor\n" + 
                            "eliminationVariable: " + str(eliminationVariable) + \
                            "\nunconditionedVariables:" + str(factor.unconditionedVariables()))
        
        if len(factor.unconditionedVariables()) == 1:
            print "Factor failed eliminate typecheck: ", factor
            raise ValueError, ("Factor has only one unconditioned variable, so you " \
                    + "can't eliminate \nthat variable.\n" + \
                    "eliminationVariable:" + str(eliminationVariable) + "\n" +\
                    "unconditionedVariables: " + str(factor.unconditionedVariables()))

        "*** YOUR CODE HERE ***"
        #get the variable domain dictionary
        var_domain_dict = factor.variableDomainsDict()
        #get the conditioned variables of the input factor
        conditioned_vars = [] #initial a list to store
        for conditioned_var in factor.conditionedVariables() :
            if conditioned_var == eliminationVariable :#discard the elimination variable
                continue
            else :
                conditioned_vars.append(conditioned_var)
        #get the unconditiond variables of the input factor
        unconditioned_vars = [] #initial a list to store
        for unconditioned_var in factor.unconditionedVariables() :
            if unconditioned_var == eliminationVariable : #discard the elimination variable
                continue
            else :
                unconditioned_vars.append(unconditioned_var)
        #use the things above, create a new factor
        #def __init__(self, inputUnconditionedVariables, inputConditionedVariables, inputVariableDomainsDict):
        after_elimination_factor = Factor(unconditioned_vars,conditioned_vars,var_domain_dict)
        #sum total probabilities for all domains of the elimination variable
        new_assignment_dict = after_elimination_factor.getAllPossibleAssignmentDicts()
        original_assignment_dict = factor.getAllPossibleAssignmentDicts()
        for new_assignment in new_assignment_dict :
            #initial the new_assignment_probability 
            assignment_probability = 0 #for the later sum
            for original_assignment in original_assignment_dict :
                flag = 0 # turn to 1 if we find a changed
                for key in new_assignment.keys() :
                    if new_assignment[key] != original_assignment[key] :
                        flag = 1 #track a changed
                        break
                    else:
                        continue
                if flag == 0 :
                        assignment_probability = assignment_probability + factor.getProbability(original_assignment)
                else:
                    continue
            after_elimination_factor.setProbability(new_assignment,assignment_probability)
        return after_elimination_factor
        #util.raiseNotDefined()
    return eliminate

eliminate = eliminateWithCallTracking()


def normalize(factor):
    """
    Question 5: Your normalize implementation 

    Input factor is a single factor.

    The set of conditioned variables for the normalized factor consists 
    of the input factor's conditioned variables as well as any of the 
    input factor's unconditioned variables with exactly one entry in their 
    domain.  Since there is only one entry in that variable's domain, we 
    can either assume it was assigned as evidence to have only one variable 
    in its domain, or it only had one entry in its domain to begin with.
    This blurs the distinction between evidence assignments and variables 
    with single value domains, but that is alright since we have to assign 
    variables that only have one value in their domain to that single value.

    Return a new factor where the sum of the all the probabilities in the table is 1.
    This should be a new factor, not a modification of this factor in place.

    If the sum of probabilities in the input factor is 0,
    you should return None.

    This is intended to be used at the end of a probabilistic inference query.
    Because of this, all variables that have more than one element in their 
    domain are assumed to be unconditioned.
    There are more general implementations of normalize, but we will only 
    implement this version.

    Useful functions:
    Factor.getAllPossibleAssignmentDicts
    Factor.getProbability
    Factor.setProbability
    Factor.unconditionedVariables
    Factor.conditionedVariables
    Factor.variableDomainsDict
    """

    # typecheck portion
    variableDomainsDict = factor.variableDomainsDict()
    for conditionedVariable in factor.conditionedVariables():
        if len(variableDomainsDict[conditionedVariable]) > 1:
            print "Factor failed normalize typecheck: ", factor
            raise ValueError, ("The factor to be normalized must have only one " + \
                            "assignment of the \n" + "conditional variables, " + \
                            "so that total probability will sum to 1\n" + 
                            str(factor))

    "*** YOUR CODE HERE ***"
    #get the variable domain dictionary
    var_domain_dict = factor.variableDomainsDict()
    #get all the conditioned variables
    conditioned_vars = [] #initial a list to store
    for conditioned_var in factor.conditionedVariables() :
        conditioned_vars.append(conditioned_var)
    #get all the unconditioned variables
    unconditioned_vars = [] #initial a list to store
    for unconditioned_var in factor.unconditionedVariables() :
        unconditioned_vars.append(unconditioned_var)
    sum_of_probability = 0#for later cascade sum
    assignment_dict = factor.getAllPossibleAssignmentDicts()
    #Get the total sum of probability
    for assignment in assignment_dict :
        sum_of_probability = sum_of_probability + factor.getProbability(assignment)
    if sum_of_probability == 0 : #If the sum of probabilities in the input factor is 0,
        return None                                                       #you should return None.
    elif sum_of_probability == 1 : #just return this perfect result
        return factor
    else : #now we need to do normalization
        for var in var_domain_dict :
            if len(var_domain_dict[var]) == 1 and var in unconditioned_vars:# an unconditioned variable only has 1 domain
                unconditioned_vars.remove(var)
                conditioned_vars.append(var)#it should become a conditioned variable in the new factor
        #now we can create a new factor
        normalized_factor = Factor(unconditioned_vars,conditioned_vars,var_domain_dict)
        #new_assignment_dict = normalized_factor.getAllPossibleAssignmentDicts()
        for assignment in assignment_dict :
            assignment_probability = factor.getProbability(assignment) / sum_of_probability
            normalized_factor.setProbability(assignment,assignment_probability)
        return normalized_factor
    #util.raiseNotDefined()

