# qlearningAgents.py
# ------------------
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


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math
from collections import defaultdict
import pickle

class QLearningAgent(ReinforcementAgent):
    """
        Q-Learning Agent

        Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

        Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

        Functions you should use
        - self.getLegalActions(state)
            which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)
        self.qvs = {}


    def hash_state(self, state, action=''):
        key = '{}:{}'.format(state.__str__(), action)
        return key


    def getQValue(self, state, action):
        """
            Returns Q(state,action)
            Should return 0.0 if we have never seen a state
            or the Q node value otherwise
        """
        return self.qvs.get(self.hash_state(state, action), 0.0)


    def computeValueFromQValues(self, state):
        """
            Returns max_action Q(state,action)
            where the max is over legal actions.  Note that if
            there are no legal actions, which is the case at the
            terminal state, you should return a value of 0.0.
        """
        legalActions = self.getLegalActions(state)

        qvs = [
            (self.getQValue(state, action), action,)
            for action in legalActions
        ]
        if len(qvs) == 0:
            return 0.0
        else:
            return max(qvs, key=lambda x: x[0])[0]


    def computeActionFromQValues(self, state):
        """
            Compute the best action to take in a state.  Note that if there
            are no legal actions, which is the case at the terminal state,
            you should return None.
        """
        legalActions = self.getLegalActions(state)

        qvs = [
            (self.getQValue(state, action), action,)
            for action in legalActions
        ]
        if len(qvs) == 0:
            return None
        else:
            maxQ = max(qvs, key=lambda x: x[0])[0]
            maxActions = [action for q, action in qvs if q == maxQ]
            return random.choice(maxActions) if len(maxActions) > 0 else None


    def getAction(self, state):
        """
            Compute the action to take in the current state.  With
            probability self.epsilon, we should take a random action and
            take the best policy action otherwise.  Note that if there are
            no legal actions, which is the case at the terminal state, you
            should choose None as the action.
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None

        if len(legalActions) == 0:
            action = None
        elif util.flipCoin(self.epsilon) is True:
            action = random.choice(legalActions)
        else:
            action = self.computeActionFromQValues(state)

        return action

    def update(self, state, action, nextState, reward):
        """
            The parent class calls this to observe a
            state = action => nextState and reward transition.
            You should do your Q-Value update here

            NOTE: You should never call this function,
            it will be called on your behalf
        """

        self.qvs[self.hash_state(state, action)] = \
          self.getQValue(state, action) + self.alpha * (
            reward +
            self.discount * self.computeValueFromQValues(nextState) -
            self.getQValue(state, action)
            )


    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
        ApproximateQLearningAgent

        You should only have to overwrite getQValue
        and update.  All other QLearningAgent functions
        should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()
        self.cum_weights = defaultdict(list)

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        q = 0
        for key, feature in self.featExtractor.getFeatures(state, action).items():
            q += self.getWeights()[key] * feature

        return q

    def update(self, state, action, nextState, reward):
        """
            Should update your weights based on transition
        """

        maxQ = self.computeValueFromQValues(nextState)
        features = self.featExtractor.getFeatures(state, action)
        qdiff = reward + (self.discount * maxQ) - self.getQValue(state, action)

        for key, feature in features.items():
            self.weights[key] += feature * self.alpha * qdiff

        self.write()

    def write(self):
        for i in ["bias", "#-of-ghosts-1-step-away", "eats-food", "closest-food", "ghosts-scared-timer"]:
            self.cum_weights[i].append(self.weights[i])

    def save(self):
        with open('./cmu_weights.pkl','wb') as f:
            pickle.dump(self.cum_weights,f)

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            print('training done')
            self.save()