# myAgents.py
# ---------------
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

from game import Agent
from searchProblems import PositionSearchProblem

import util
import time
import search

"""
IMPORTANT
`agent` defines which agent you will use. By default, it is set to ClosestDotAgent,
but when you're ready to test your own agent, replace it with MyAgent
"""
def createAgents(num_pacmen, agent='ClosestDotAgent'): #완성하면 바꾸기.
    return [eval(agent)(index=i) for i in range(num_pacmen)]

class MyAgent(Agent):
    """
    Implementation of your agent.
    """

    def getAction(self, state):
        """
        Returns the next action the agent will take
        """

        "*** YOUR CODE HERE ***"
        legalMoves = state.getLegalActions()
        #print(legalMoves)
        # Choose one of the best actions
        scores = [self.evaluationFunction(state, action) for action in legalMoves]
        #print(scores)
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        return legalMoves[chosenIndex]

    def initialize(self):
        """
        Intialize anything you want to here. This function is called
        when the agent is first created. If you don't need to use it, then
        leave it blankpac
        """

        "*** YOUR CODE HERE"
        return 0
        raise NotImplementedError()

"""
Put any other SearchProblems or search methods below. You may also import classes/methods in
search.py and searchProblems.py. (ClosestDotAgent as an example below)
"""

class ClosestDotAgent(Agent):

    def findPathToClosestDot(self, gameState):
        # UCS -> A*로 바꾸자.
        """
        Returns a path (a list of actions) to the closest dot, starting from
        gameState.
        """
        # Here are some useful elements of the startState
        startPosition = gameState.getPacmanPosition(self.index)
        food = gameState.getFood()
        walls = gameState.getWalls()
        problem = AnyFoodSearchProblem(gameState, self.index)


        "*** YOUR CODE HERE ***"

        pacmanCurrent = [problem.getStartState(), [], 0] #state, direction, cost
        #-->fringe.push( (startState, [], 0), 0 ) 
        visitedPosition = set()
        # visitedPosition.add(problem.getStartState())
        fringe = util.PriorityQueue()
        fringe.push(pacmanCurrent, pacmanCurrent[2])
        while not fringe.isEmpty():
            pacmanCurrent = fringe.pop()
            if pacmanCurrent[0] in visitedPosition: #방문한 곳이면
                continue #continue
            else:
                visitedPosition.add(pacmanCurrent[0]) #아니면 추가
            if problem.isGoalState(pacmanCurrent[0]): #골이면 
                return pacmanCurrent[1] #경로 리턴
            else: # 아니면 경로에 추가
                pacmanSuccessors = problem.getSuccessors(pacmanCurrent[0])
                # [ ((34, 15), 'South', 1), ((33, 16), 'West', 1) ]
            Successor = []
            for item in pacmanSuccessors:  # item: [(x,y), 'direction', cost]
                if item[0] not in visitedPosition:
                    pacmanRoute = pacmanCurrent[1].copy()
                    pacmanRoute.append(item[1])
                    sumCost = pacmanCurrent[2]
                    Successor.append([item[0], pacmanRoute, sumCost + item[2]])
            for item in Successor:
                fringe.push(item, item[2])
        return pacmanCurrent[1] #경로 리턴

    def getAction(self, state):
        return self.findPathToClosestDot(state)[0]

class AnyFoodSearchProblem(PositionSearchProblem):
    """
    A search problem for finding a path to any food.

    This search problem is just like the PositionSearchProblem, but has a
    different goal test, which you need to fill in below.  The state space and
    successor function do not need to be changed.

    The class definition above, AnyFoodSearchProblem(PositionSearchProblem),
    inherits the methods of the PositionSearchProblem.

    You can use this search problem to help you fill in the findPathToClosestDot
    method.
    """

    def __init__(self, gameState, agentIndex):
        "Stores information from the gameState.  You don't need to change this."
        # Store the food for later reference
        self.food = gameState.getFood()

        # Store info for the PositionSearchProblem (no need to change this)
        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition(agentIndex)
        self.costFn = lambda x: 1
        self._visited, self._visitedlist, self._expanded = {}, [], 0 # DO NOT CHANGE

    def isGoalState(self, state):
        """
        The state is Pacman's position. Fill this in with a goal test that will
        complete the problem definition.
        """
        x,y = state
        if self.food[x][y] == True:
            return True
        return False

