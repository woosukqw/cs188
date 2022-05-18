# myTeam.py
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


from captureAgents import CaptureAgent
import random, time, util
from game import Directions
import game

from util import nearestPoint
from math import inf

from types import FunctionType

import os
import sys
import pandas as pd

# The simulation code below is commented out because it has to import minicontest2.capture,
# just making sure this to work in case if this file is outside of minicontest2 framework.

# from capture import runGames, readCommand
# import pickle

# def make_tuple(x):
#     return tuple([i for i in x])

# def read_previous_score():
#     df = pd.read_csv('output.csv', sep=',')
#     import pdb; pdb.set_trace()

# filename = 'parameters'

# def default_params():
#     parameters = [
#         # this value is from simulation result.
#         [1,1,4,0.5,1,-6],
#         [1,1,4,1,2,-4],
#         [1.5,1,6,1,1,0.5],
#     ]
#     # 1 or -1
#     weights = [
#         [1,1,1,1,1,1],
#         [1,1,1,1,1,1],
#         [1,1,1,1,1,1],
#     ]
#     indices = [0,0,0]
#     b = 0.5

#     return parameters, weights, indices, b

# def load_params():
#     return pickle.load(open(filename, 'rb'))

# def save_params(*args):
#     return pickle.dump(make_tuple(args), open(filename, 'wb'))

# def simulate(num = 10, enemy_team='your_baseline1.py'):

#     if os.path.isfile('parameters') is False:
#         print('initialize')
#         pickle.dump(default_params(), open(filename, 'wb'))

#     latest_avg_score = 0
#     latest_win_rate = 0

#     sys.argv.append('-q')
#     sys.argv.append('-n')
#     sys.argv.append(str(num))
#     sys.argv.append('-r')
#     # myself
#     sys.argv.append('your_best.py')

#     while True:

#         parameters, weights, indices, b = load_params()
#         # 0, 1, 2
#         x = random.randint(0, 2)

#         # apply
#         parameters[x][indices[x]] += b * weights[x][indices[x]]
#         print(parameters)

#         # do game
#         # createTeam() is in this func
#         options = readCommand(sys.argv[1:], enemy_team)
#         games, Avg_score, redWinRate, redLoseRate = runGames(**options)

#         WinRate = redWinRate if redWinRate>redLoseRate else -redLoseRate

#         print('===\nnextX: {}, WR: {}, Score: {}'.format(x, WinRate, Avg_score))


#         if WinRate > latest_win_rate and Avg_score > latest_avg_score:
#             print('We are on a right track.')
#         elif WinRate == latest_win_rate or Avg_score == latest_avg_score:
#             print('Keep going..')
#         else:
#             print('We are on a wrong track. Rolling back..')
#             parameters[x][indices[x]] -= b * weights[x][indices[x]]


#         indices[x] += 1
#         if indices[x] >= len(parameters[x]):
#             indices[x] = 0

#         # saving indices
#         save_params(parameters, weights, indices, b)


# if __name__ == '__main__':
#     simulate()

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
                first = 'EnhancedMixedReflexAgent', second = 'EnhancedMixedReflexAgent'):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """

    parameters, weights, indices, b = load_params()

    p_pacman_1 = make_tuple(parameters[0])
    p_ghost_1 = make_tuple(parameters[1])
    p_returning_pacman_1 = make_tuple(parameters[2])

    p_pacman_2 = make_tuple(parameters[0])
    p_ghost_2 = make_tuple(parameters[1])
    p_returning_pacman_2 = make_tuple(parameters[2])

    # The following line is an example only; feel free to change it.
    return [eval(first)(firstIndex,
                p=[p_pacman_1, p_ghost_1, p_returning_pacman_1], preferred_directions=['North']),
            eval(second)(secondIndex,
                p=[p_pacman_2, p_ghost_2, p_returning_pacman_2], preferred_directions=['South'])]

##########
# Agents #
##########

class EnhancedMixedReflexAgent(CaptureAgent):
    """
    1. If agent is ghost agent should chase for pacman in our site
    2. If agent is pacman agent should grab some food

    More behavior is discussed in a report.

    * This class is created by Suho Lee (susemeee@korea.ac.kr)
    """

    def __init__(self, *args, preferred_directions=[],
        p=[(1,1,1,1,1,1,), (1,1,1,1,1,1,), (1,1,1,1,1,1,)], **kwargs):

        super().__init__(*args, **kwargs)
        # preferred_directions means the agent's 'preferred' direction,
        # if two or more action yields same score to evaulate.
        self.preferred_directions = preferred_directions
        # p = parameters
        self.p = {
            'pacman': p[0],
            'returningPacman': p[1],
            'ghost': p[2],
        }

    def registerInitialState(self, gameState):
        """Given code"""
        CaptureAgent.registerInitialState(self, gameState)
        """Code from baseline.py"""
        self.start = gameState.getAgentPosition(self.index)

    def chooseAction(self, gameState):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = gameState.getLegalActions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(gameState, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]

        foodLeft = len(self.getFood(gameState).asList())

        if foodLeft <= 2:
            bestDist = 9999
            for action in actions:
                successor = self.getSuccessor(gameState, action)
                pos2 = successor.getAgentPosition(self.index)
                dist = self.getMazeDistance(self.start,pos2)
                if dist < bestDist:
                    bestAction = action
                    bestDist = dist
            return bestAction

        # prefer given action if evaluation score is on tie
        for action in bestActions:
            if action in self.preferred_directions:
                return action

        return random.choice(bestActions)


    def getSuccessor(self, gameState, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

    def evaluate(self, gameState, action):
        """
        1. Get all possible features for game (ps)
        2. Get Weights as a function (f)
        3. return sum([f(p) for p in ps])
        """
        features = self.getFeatures(gameState, action)

        if features['isPacmanToReturnHome'] is True:
            weights = self.getWeightsForReturningPacman(gameState, action)
        elif features['isPacman'] is True:
            weights = self.getWeightsForPacman(gameState, action)
        else:
            weights = self.getWeightsForGhost(gameState, action)

        final_score = 0
        for k, func in weights.items():
            val = features.get(k, 0)

            # val can be boolean; everything should be overriden if so.
            if val is True:
                return inf
            elif val is False:
                return -inf
            elif val is None:
                continue

            if type(func) is FunctionType:
                # applies function (if it is function)
                final_score += func(val)
            elif type(func) is int:
                # applies weight
                final_score += val * func
            else:
                raise ValueError('value of weights is not int or function.')

        return final_score


    def getFeatures(self, gameState, action):
        """
        """
        successor = self.getSuccessor(gameState, action)

        myNextState = successor.getAgentState(self.index)
        myNextPos = myNextState.getPosition()

        rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]

        # For ghost, computes distance to invaders we can see
        currentEnemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        currentPacmans = [a for a in currentEnemies if a.isPacman is True and a.getPosition() != None]

        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        invaderDists = [self.getMazeDistance(myNextPos, a.getPosition())
            for a in enemies if a.isPacman is True and a.getPosition() != None]
        ghostDists = [self.getMazeDistance(myNextPos, a.getPosition())
            for a in enemies if a.isPacman is False and a.getPosition() != None]

        foodList = self.getFood(successor).asList()
        minFoodDistance = min([self.getMazeDistance(myNextPos, food) for food in foodList])

        defendingFoodList = self.getFoodYouAreDefending(successor).asList()
        minDefendingFoodDistance = min([self.getMazeDistance(myNextPos, food) for food in defendingFoodList])

        capsuleList = self.getCapsules(successor)

        teammateDistances = [self.getMazeDistance(myNextPos, teammate.getPosition())
                for teammate in [successor.getAgentState(ti) for ti in self.getTeam(successor)]
                if teammate.isPacman is True]

        currentState = gameState.getAgentState(self.index)

        MIN_NUM_TO_RETURN_HOME = 2

        features = {
            # Computes whether we're pacman (1) or ghost (0)
            # This idea is from baseline.py but renamed for making it easier to understand.
            'isPacman': currentState.isPacman or currentState.scaredTimer > 0,
            'isPacmanToReturnHome': currentState.numCarrying > MIN_NUM_TO_RETURN_HOME,
            # we don't have to stop as much as possible
            'stop': False if action == Directions.STOP else 0,
            # (or reverse)
            'reverse': 1 if action == rev else 0,
            # foods currently carrying for pacman
            'carryingFoods': currentState.numCarrying,
            # successor's score when this agent does a behavior [action]
            'successorScore': self.getScore(successor),
            # Distance between invader(enemy pacman on our side) and myself
            'invaderDistance': min(invaderDists) if len(invaderDists) > 0 else None,
            # Distance between enemy ghost and myself
            'ghostDistance': min(ghostDists) if len(ghostDists) > 0 else None,
            # Distance between 'any' enemies and myself
            'generalOpponentDistance': min(invaderDists + ghostDists),
            # Compute distance to the nearest food (opponent-side)
            'distanceToFood': minFoodDistance,
            # Compute distance to the nearest food (our-side)
            'distanceToDefendingFood': minDefendingFoodDistance,
            # Distance between closest teammate and myself
            'teammateDistance': min(teammateDistances) if len(teammateDistances) > 0 else None,
            # True if invader is going to be killed at successor state, otherwise 0
            'willAnyInvaderBeKilled': True if len(currentPacmans) > len(invaderDists) else 0,
            # True if capsule is 'consumed' by this agent, which yields capsuleList.length == 0, otherwise 0
            'isCapsuleClose': True if len(capsuleList) == 0 else 0,
        }

        return features

    def getWeightsForPacman(self, gameState, action):

        a, b, c, d, e, f = self.p['pacman']

        return {
            'stop': -100,
            'reverse': -50,
            'successorScore': 1,
            'carryingFoods': 1000,
            'distanceToFood': lambda x: (1 / ((a * x + b) or 0.0001)) * c,
            'ghostDistance': lambda x: (1 / ((d * x + e) or 0.0001)) * f,
            'teammateDistance': 0,
            'isCapsuleClose': 1,
        }

    def getWeightsForReturningPacman(self, gameState, action):

        a, b, c, d, e, f = self.p['returningPacman']

        return {
            'stop': -100,
            'reverse': -50,
            'successorScore': 1,
            'distanceToDefendingFood': lambda x: (1 / ((a * x + b) or 0.0001)) * c,
            'ghostDistance': lambda x: (1 / ((d * x + e) or 0.0001)) * f,
            'teammateDistance': 0,
            'isCapsuleClose': 1,
        }

    def getWeightsForGhost(self, gameState, action):

        a, b, c, d, e, f = self.p['ghost']

        return {
            'stop': -100,
            'reverse': -2,
            'invaderDistance': lambda x: (1 / ((a * x + b) or 0.0001)) * c if x is not None else 0,
            'generalOpponentDistance': lambda x: (1 / ((d * x + e) or 0.0001)) * f,
            'willAnyInvaderBeKilled': 1,
        }
