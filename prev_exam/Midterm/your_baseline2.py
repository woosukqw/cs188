
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

    def score(self, state):
        return state.getScore()

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

    def minimizer(self, state, agent, depth, alpha, beta):

        if self.shouldStopAndEvaluate(state, depth):
            return self.score(state), Directions.STOP

        # min-score = v
        min_score = inf
        actions = []

        for action in state.getLegalActions(agent):
            next_state = state.generateSuccessor(agent, action)

            score, _ = self.maximizer(next_state, agent=self.getOppositeAgent(state, agent), depth=depth - 1, alpha=alpha, beta=beta)

            if score < min_score:
                actions = []
                actions.append(action)
                min_score = score

            if min_score < alpha:
                return min_score, action
            beta = min(beta, min_score)

        return min_score, actions[0]

    def maximizer(self, state, agent, depth, alpha, beta):

        if self.shouldStopAndEvaluate(state, depth):
            return self.score(state), Directions.STOP

        # max-score = v
        max_score = -1 * inf
        actions = []

        for action in state.getLegalActions(agent):
            next_state = state.generateSuccessor(agent, action)
            score, _ = self.minimizer(next_state, agent=self.getOppositeAgent(state, agent), depth=depth, alpha=alpha, beta=beta)

            if score > max_score:
                actions = []
                actions.append(action)
                max_score = score

            if max_score > beta:
                return max_score, action
            alpha = max(alpha, max_score)

        return max_score, actions[0]

    def chooseAction(self, gameState):
        """
        Returns the minimax action
        """
        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        _, action = self.maximizer(gameState, self.index, self.depth, alpha=-1 * inf, beta=inf)
        # print('eval time for agent %d: %.4f' % (self.index, time.time() - start))
        return action
