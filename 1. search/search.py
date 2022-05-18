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
'''
!!!!!!!!!Fringe!!!!!!!!!!
If you're performing a tree (or graph) search, 
then the set of all nodes at the end of all visited paths is called the fringe, frontier or border.
'''

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
    '''
    '''
    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    getSuccessors -> successor, action, stepcost를 반환하는 함수임.
    successor는 다음에 이동할 수 있는 state
    action은 현재 state에서 다음 state로 이동하려면 어느 방향으로 가야하는지 
    stepcost는 이동하는데에 드는 비용

    TINY::  x  y
    Start: (5, 5)
    Is the start a goal? False
    Start's successors: [((5, 4), 'South', 1), ((4, 5), 'West', 1)]
    MEDIUM::
    Start: (34, 16)
    Is the start a goal? False
    Start's successors: [((34, 15), 'South', 1), ((33, 16), 'West', 1)]
    """
    "*** YOUR CODE HERE ***"
    # Run: python3 pacman.py -l mediumMaze -p SearchAgent -a fn=depthFirstSearch
    #util.raiseNotDefined()
    #print("Start:", problem.getStartState())
    #print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    #print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    startState = problem.getStartState()
    fringe = util.Stack()
    fringe.push((startState, []))
    visited = []
    
    while not fringe.isEmpty():
        node, direction = fringe.pop()
        visited.append(node)
        if (problem.isGoalState(node)): #
            return direction

        for coordinate, NWSE, stepcost in problem.getSuccessors(node):
            if not coordinate in visited: #방문 안한 노드였다면 tuple쌍으로 (가는 노드 위치정보, 지금까지 이동하는 위치정보 총합)을 넣음.
                fringe.push( (coordinate, direction+[NWSE]) )
        
    return []


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    '''
    1단계. 시작 노드를 방문했던 노드에 삽입한다. 
    2단계. 방문할 노드에 시작노드의 Child Node를 삽입한다. 
    3단계. Child노드를 중심으로 다시 1~2단계를 거쳐 탐색한다. 
    '''
    #util.raiseNotDefined()
    startState = problem.getStartState()
    fringe = util.Queue()
    fringe.push((startState, []))
    visited = []

    while not fringe.isEmpty():
        node, direction = fringe.pop()

        if (problem.isGoalState(node)):
            return direction
        
        for coordinate, NWSE, stepcost in problem.getSuccessors(node):
            if not coordinate in visited:
                fringe.push( (coordinate, direction+[NWSE]) )
                visited.append(coordinate)
        
        visited.append(node)

    return []

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    #util.raiseNotDefined()
    startState = problem.getStartState()
    fringe = util.PriorityQueue()
    #class priorityQueue.push(self, item, priority) 니까 
    #priority에 0을 넣고, item에 (state, direction, cost)튜플을 넣어줌.
    fringe.push( (startState, [], 0), 0 ) # state와 direction과 cost. 
    visited = dict() # 간 위치 (x,y)와 cost를 매핑해야함.
    # 특정 state에 도달하기 까지 방문한 state들과 그 state에 도달하기까지의 cost(최소비용인가?..)를 매핑

    while not fringe.isEmpty():
        node, direction, cost = fringe.pop()
        if (problem.isGoalState(node)):
            return direction

        visited[node] = cost

        for coordinate, NWSE, stepcost in problem.getSuccessors(node):
            # [ ((34, 15), 'South', 1), ((33, 16), 'West', 1) ]
            #다음 State가 이전에 방문한 적이 없거나, 방문했던 곳보다 cost합이 더 적은지 검사.
            if ( not coordinate in visited or (coordinate in visited and visited[coordinate] > cost + stepcost) ):
                visited[coordinate] = cost + stepcost # state-cumulative cost 매핑
                #direction.append(NWSE) append하면 안됨!!!!! direction에 들어갈 정답인지 확정이 아니기 때문.
                # 다음으로 이동할 state와 state로 이동하기 위한 방향들, 그리고 가기 위한 비용을 push함.
                fringe.push( (coordinate, direction+[NWSE], cost+stepcost), cost+stepcost )

    return []
def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 1
    
def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    #util.raiseNotDefined()
    startState = problem.getStartState()
    fringe = util.PriorityQueue()
    #class priorityQueue.push(self, item, priority) 니까 
    #priority에 0을 넣고, item에 (state, direction, cost)튜플을 넣어줌.
    fringe.push( (startState, [], 0), heuristic(startState, problem) ) # state와 direction과 cost=(g+h). 
    visited = dict() # 간 위치 (x,y)와 cost를 매핑해야함.
    # 특정 state에 도달하기 까지 방문한 state들과 그 state에 도달하기까지의 cost(최소비용인가?..)를 매핑

    while not fringe.isEmpty():
        node, direction, cost = fringe.pop()
        if (problem.isGoalState(node)):
            return direction
            
        visited[node] = cost + heuristic(node, problem) # g: cost, h: heu~
        
        '''
        for문의 역할이 방문하지 않은 다음 노드를 가기위한 비용을 계산하면서 fringe에 넣는거잖아?
        '''
        for coordinate, NWSE, stepcost in problem.getSuccessors(node):
            # [ ((34, 15), 'South', 1), ((33, 16), 'West', 1) ]
            #다음 State가 이전에 방문한 적이 없거나, 방문했던 곳보다 cost합이 더 적은지 검사.
            if ( not coordinate in visited or (coordinate in visited and visited[coordinate] > cost + stepcost + heuristic(coordinate, problem)) ):
                visited[coordinate] = cost + stepcost # state-cumulative cost 매핑
                # ((다음으로 이동할 state와 state로 이동하기 위한 방향들, 그리고 가기 위한 비용), (비용))을 push함.
                fringe.push(  ( coordinate, direction+[NWSE], cost+stepcost  ), cost+stepcost+heuristic(coordinate, problem) )


    return []


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
