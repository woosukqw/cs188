
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

import time
from math import inf
from types import FunctionType

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
                first = 'AlphaBetaPrunedMinimaxAgent', second = 'AlphaBetaPrunedMinimaxAgent'):
    return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

class AlphaBetaPrunedMinimaxAgent(CaptureAgent):


    def getFeatures(self, gameState):
        """
        Scoring method from Baseline 1 and 2.
        """

        myState = gameState.getAgentState(self.index)
        myPos = myState.getPosition()

        # For ghost, computes distance to invaders we can see
        currentEnemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        currentPacmans = [a for a in currentEnemies if a.isPacman is True and a.getPosition() != None]

        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        invaderDists = [self.getMazeDistance(myPos, a.getPosition())
            for a in enemies if a.isPacman is True and a.getPosition() != None]
        ghostDists = [self.getMazeDistance(myPos, a.getPosition())
            for a in enemies if a.isPacman is False and a.getPosition() != None]

        foodList = self.getFood(gameState).asList()
        minFoodDistances = sorted([self.getMazeDistance(myPos, food) for food in foodList])

        defendingFoodList = self.getFoodYouAreDefending(gameState).asList()
        minDefendingFoodDistances = sorted([self.getMazeDistance(myPos, food) for food in defendingFoodList])

        capsuleList = self.getCapsules(gameState)

        teammateDistances = [self.getMazeDistance(myPos, teammate.getPosition())
                for teammate in [gameState.getAgentState(ti) for ti in self.getTeam(gameState)]
                if teammate.isPacman is True]

        currentState = gameState.getAgentState(self.index)

        MIN_NUM_TO_RETURN_HOME = 0

        features = {
            # Computes whether we're pacman (1) or ghost (0)
            # This idea is from baseline.py but renamed for making it easier to understand.
            'isPacman': currentState.isPacman or currentState.scaredTimer > 0,
            'isPacmanToReturnHome': currentState.numCarrying > MIN_NUM_TO_RETURN_HOME,
            # foods currently carrying for pacman
            'carryingFoods': currentState.numCarrying,
            # Distance between invader(enemy pacman on our side) and myself
            'invaderDistance': min(invaderDists) if len(invaderDists) > 0 else None,
            # Distance between enemy ghost and myself
            'ghostDistance': min(ghostDists) if len(ghostDists) > 0 else None,
            # Distance between 'any' enemies and myself
            'generalOpponentDistance': min(invaderDists + ghostDists),
            # Compute distance to the nearest food (opponent-side)
            'distanceToFood': 0 if minFoodDistances[0] <= 1 else sum(minFoodDistances[:2]),
            # Compute distance to the nearest food (our-side)
            'distanceToDefendingFood': sum(minDefendingFoodDistances[:1]),
            # Distance between closest teammate and myself
            'teammateDistance': min(teammateDistances) if len(teammateDistances) > 0 else None,
            # True if invader is going to be killed at successor state, otherwise 0
            'willAnyInvaderBeKilled': 1 if len(currentPacmans) > len(invaderDists) else 0,
            # True if capsule is 'consumed' by this agent, which yields capsuleList.length == 0, otherwise 0
            'isCapsuleClose': 1 if len(capsuleList) == 0 else 0,
        }

        return features

    def getWeightsForPacman(self, gameState):
        b = 10
        return {
            'carryingFoods': lambda x: (x + b) * 10,
            'distanceToFood': lambda x: (x + b) * -2,
            'ghostDistance': lambda x: (x + b) * 1 if x is not None else 0,
            'teammateDistance': 0,
            'isCapsuleClose': 100,
        }

    def getWeightsForReturningPacman(self, gameState):
        b = 10
        return {
            'distanceToDefendingFood': lambda x: (x + b) * -2,
            'ghostDistance': lambda x: (x + b) * 1 if x is not None else 0,
            'teammateDistance': 0,
            'isCapsuleClose': 100,
        }

    def getWeightsForGhost(self, gameState):
        b = 10
        return {
            'invaderDistance': lambda x: (x + b) * -1 if x is not None else 0,
            'generalOpponentDistance': lambda x: (x + b) * -1,
            'willAnyInvaderBeKilled': 100,
        }


    def evaluateScore(self, gameState):
        features = self.getFeatures(gameState)

        if features['isPacmanToReturnHome'] is True:
            weights = self.getWeightsForReturningPacman(gameState)
        elif features['isPacman'] is True:
            weights = self.getWeightsForPacman(gameState)
        else:
            weights = self.getWeightsForGhost(gameState)

        final_score = 0
        for k, func in weights.items():
            val = features.get(k, 0)

            if val is None:
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


    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)
        self.depth = 2

    def shouldStopAndEvaluate(self, state, depth):
        if state.isOver():
            return True
        elif depth == 0:
            return True
        return False

    def getOppositeAgent(self, state, agent_index):
        if agent_index in state.getRedTeamIndices():
            # if red team
            enemy_agent = state.getBlueTeamIndices()[state.getRedTeamIndices().index(agent_index)]
        else:
            # if blue team
            enemy_agent = state.getRedTeamIndices()[state.getBlueTeamIndices().index(agent_index)]
        return enemy_agent

    def getMoves(self, state, agent):
        # I think stop is useless
        return [action for action in state.getLegalActions(agent) if not (action == Directions.STOP)]

    def minimizer(self, state, agent, depth, alpha, beta):

        if self.shouldStopAndEvaluate(state, depth):
            return self.evaluateScore(state), Directions.STOP

        # min-score = v
        min_score = inf
        actions = []

        for action in self.getMoves(state, agent):
            next_state = state.generateSuccessor(agent, action)

            score, _ = self.maximizer(next_state, agent=self.getOppositeAgent(state, agent), depth=depth - 1, alpha=alpha, beta=beta)

            if score <= min_score:
                actions = []
                actions.append(action)
                min_score = score

            if min_score < alpha:
                return min_score, action
            beta = min(beta, min_score)

        return min_score, random.choice(actions)

    def maximizer(self, state, agent, depth, alpha, beta):

        if self.shouldStopAndEvaluate(state, depth):
            return self.evaluateScore(state), Directions.STOP

        # max-score = v
        max_score = -1 * inf
        actions = []

        for action in self.getMoves(state, agent):
            next_state = state.generateSuccessor(agent, action)
            score, _ = self.minimizer(next_state, agent=self.getOppositeAgent(state, agent), depth=depth, alpha=alpha, beta=beta)

            if score >= max_score:
                actions = []
                actions.append(action)
                max_score = score

            if max_score > beta:
                return max_score, action
            alpha = max(alpha, max_score)

        return max_score, random.choice(actions)

    def chooseAction(self, gameState):
        """
        Returns the minimax action
        """
        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        _, action = self.maximizer(gameState, self.index, depth=self.depth, alpha=-1 * inf, beta=inf)
        # print('eval time for agent %d: %.4f' % (self.index, time.time() - start))
        return action
