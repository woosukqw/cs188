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
#2021320090 컴퓨터학과 최우석

from util import manhattanDistance
from game import Directions
import random, util
import sys

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
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()
        #print(legalMoves)
        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        #print(scores)
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        #"Add more of your code here if you want to"
        #s = [float("%.3f" %i) for i in scores ]
        #print("legal:", legalMoves, "scores:",s, "bestIndex: ", bestIndices, file = sys.stdout )
        #if (legalMoves[chosenIndex] == 'Stop'):
        #    print('STOPP')
        #
        #if (legalMoves[chosenIndex] == 'Stop'):
        #    pass
        #    #print(game)
            
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
        """
        Logic is here.
        #1: 다음 행동들(actions)중 한 행동(action)으로 이동을 할 때 어떻게 되는지 판단을 해 점수를 return하는것임.
        #2: 이 action을 하면 가장 가까운 음식에 대해서 더 가까워지는지, 그리고 ghost와 가까워지는지를 판단함.
        #3: Note: As features, try the reciprocal of important values (such as distance to food) 
        #         rather than just the values themselves.
        #4: Note: The evaluation function you're writing is evaluating state-action pairs; 
        #         in later parts of the project, you'll be evaluating states.
        #5: Note: Remember that newFood has the function asList()
        #6: Note: You may find it useful to view the internal contents of various objects for debugging. 
        #           You can do this by printing the objects’ string representations. 
        #           For example, you can print newGhostStates with print(newGhostStates) .

        1.현재 게임 상태에서 팩맨의 위치정보 받아서 저장
        2. 현재 게임 상태에서 음식의 위치 정보 받아서 저장
        3. 현재 게임 상태에서 맵의 높이와 너비를 구하기 위해 벽의 정보를 받아와서 저장.
        4. 게임 상에서 팩맨과 유령 혹은 음식이 가장 멀리 있는 경우는 맵의 양 대각선 끝쪽이기때문에 맵의 높이와 너비를 더한다.
        5. 팩맨이 움직이면 올 수 있는 다음 게임 상태에 대한 정보를 저장
        6. 다음 게임 상태에서 팩맨의 위치정보를 저장
        7. 다음 게임 상태에서 음식의 위치정보를 저장
        --> 기본 정보들 준비.

        8. 상태를 평가하기 위한 점수를 0으로 선언
        9. 다음으로의 상태에서의 팩맨의 위치가 현재 상태에서의 음식의 위치들중 하나와 맞는지 검사
         9-1.음식을 전부먹어야지 게임에서 승리하므로 음식의 위치로 가는것에 점수를 많이 부여(10)
        10. 음식과 팩맨의 최소거리를 구하기 위해 우선적으로 무한값으로 선언
        11. 다음 상태에서 음식마다 반복문을 실행
         11-1. 음식과 팩맨의 위치간의 거리를 맨허튼 거리 함수를 이용해 구함
         11-2. 음식과 팩맨 사이의 위치들 중 최소값을 구하기 위해 min()사용

        12. 유령과 팩맨의 최소거리를 구하기 위해 우선적으로 무한 값으로 선언
        13. 다음으로의 상태에서 유령마다 반복문을 실행
         13-1.유령과 팩맨의 위치간의 거리를 맨허튼 거리 함수를 이용해 구함
         13-2.유령과 팩맨 사이의 위치들 중 최소값을 구하기 위해 min()사용

        14. 팩맨과 유령과의 최소 거리가 2보다 작은지 검사
         14-1. 유령과 부딫힐 경우 게임 패배, 거리가 2보다 작으면 점수를 많이 많이 감소시킨다.

        15.평가하기 위한 점수는 
        -앞에서 구한 음식을 먹었는지 여부에 따른 점수
        -유령과 거리가 2보다 작은지의 여부에 따른 점수
        -추가적으로 음식과의 최소거리의 역수
        -유령과의 최소거리에 거리중 최대로 나올 수 있는 maxlength를 나눈값을 더해준다.
        -유령들중 최소거리/최대거리?
        16. 점수 반환
        newPos: (1, 1)
        newFood:
        FFFFF
        FFTFF
        FTFTF
        FFTFF
        FTFTF
        FFFFF
        FFFTF
        FFFFF
        FFFTF
        FFFFF
        a: Ghost: (x,y)=(2, 7), Stop
        newScared:  [0] 

        """
        """
        s = currentGameState.generatePacmanSuccessor(action)
        
        #pacmanPos = successorGameState.getPacmanPosition()
        #foodPos = successorGameState.getFood()
        print('pacmanPos:', s.getPacmanPosition())
        print('getFood():', s.getFood())
        print('getGhostStates[0]:', s.getGhostStates()[0])
        print('getLegalActions():', s.getLegalActions())
        print('getNumAgents():', s.getNumAgents())
        print('getScore():', s.getScore())
        print('getWalls()):',currentGameState.getWalls())
        print('\n')
        """
        '''
        pacmanPos: (1, 1)
        getFood(): FFFFF
        FFTFF
        FTFTF
        FFTFF
        FTFTF
        FFFFF
        FFFTF
        FFFFF
        FFFTF
        FFFFF
        getGhostStates[0]: Ghost: (x,y)=(2, 7), Stop
        getLegalActions(): ['Stop', 'East', 'North']
        getNumAgents(): 2
        getScore(): -1.0
        getWalls()): TTTTT
        TFFFT
        TFFFT
        TFFFT
        TFFFT
        TFFFT
        TFFFT
        TFFFT
        TFFFT
        TTTTT
        '''
        '''
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action) #pacman.py -> generatePacmanSuccessor -> 
        newPos = successorGameState.getPacmanPosition() # Pacman position after moving
        newFood = successorGameState.getFood() #remaining food
        newGhostStates = successorGameState.getGhostStates() #귀신의 state겠지 뭐,, 2마리일땐 어떻게나옴?
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        print("newPos:", newPos)
        print("newFood:\n", newFood)
        #print("newGhost: ", newGhostStates)
        a = newGhostStates
        print("a:", a[0])

        print("newScared: ", newScaredTimes, "\n\n")
        # 각 귀신이 무서워하는 상태로 남아있는 움직임의 개수
        print(sGS.getFood()):
        FFFFF
        FFTFF
        FTFTF
        FFTFF
        FTFTF
        FFFFF
        FFFTF
        FFFFF
        FFFTF
        FFFFF
        for i in foodPos:
            print(i)
        [False, False, False, False, False, False, False, False, False, False]
        [False, False, False, False, False, True, False, True, False, False]
        [False, False, False, False, False, False, True, False, True, False]
        [False, True, False, True, False, True, False, True, False, False]
        [False, False, False, False, False, False, False, False, False, False]
        "*** YOUR CODE HERE ***"
        '''
        #1 현재의 팩맨 위치
        pacmanPos = currentGameState.getPacmanPosition()
        #2 현재 음식들의 위치 - grid class로 표현됨
        foodPos = currentGameState.getFood()
        #3 맵 정보 -> #4에서 가로 세로 길이 이용
        mapLayout = currentGameState.getWalls()
        #4 맨허튼 거리로 최대한 긴 거리일때 얼마인지 측정 (가로, 세로에 벽도 포함되서 4를 뺌)
        max_length = mapLayout.width + mapLayout.height - 4 # 맨허튼 거리기준 가장 긴 거리
        #5 action(특정 방향)으로 이동한 state의 정보를 받아옴
        sGS= currentGameState.generatePacmanSuccessor(action) # successorGameState 
        #print('cur:',currentGameState.getPacmanPosition(), 'next:', sGS.getPacmanPosition())
        #6 이동했다 가정하고, 거기서 팩맨의 위치
        sGS_successor_position = sGS.getPacmanPosition()
        #7 이동했다 가정하고, 거기서 음식들의 위치
        sGS_food_Position = sGS.getFood()
        #8 이동했다 가정하고, possible next state, 만약 score가 같다면 이거 개수로 판별(가능성이 많은거니)
        possible_next_states = sGS.getLegalActions()
        #9 캡슐(아마 고스트 먹을 수 있게 하는 아이템)들의 위치
        cur_capsules_position = currentGameState.getCapsules()
        #print(currentGameState.getCapsules())
        #print(sGS_successor_position[0])
        #print(sGS.getFood())
        #10 위 정보들을 토대로 점수를 매길거임. 그거 init
        total_score = 0
        #11 현재 action으로 가면 food를 먹는지 확인 -> 먹는다면 score++
        if (foodPos[sGS_successor_position[0]][sGS_successor_position[1]]):
            #print("NEXT ACTION WILL GET FOOD!")
            total_score += 10
        #12 이동후의 팩맨 기준, 음식들 중 가장 최소거리 구하기 시작
        min_distance = 10000000000
        #13 LOOP START - 음식과의 최소거리의 역수 반환할거임
        #print('::\n')
        pacman_x = sGS_successor_position[0]
        pacman_y = sGS_successor_position[1]
        _x, _y = -1, -1
        for x in range(1, mapLayout.width-1): #0, -1은 벽이니까 제외.  range(1,4)
            for y in range(1, mapLayout.height-1): #0, -1은 벽이니까 제외. range(1,9)
                #print(foodPos[x][y], end='\t')
                if (foodPos[x][y] == True):
                    cur_mahattan = abs(pacman_x-x) + abs(pacman_y-y)
                    if (min_distance > cur_mahattan): #가장 작은 거리 업데이트
                        min_distance = cur_mahattan
                        _x, _y = x, y #only for debug. del it when it fin.
            #print('')
        # score에 음식과의 최소거리의 역수 추가
        if (min_distance == 0): 
            pass #위에서 10점이나 줬자나
        else: 
            total_score += (1/min_distance)
            #total_score += min_distance/max_length
        #print("mind:",min_distance)
        
        #14 이동후의 팩맨 기준, 유령과 팩맨의 최소거리 구하기 시작
        min_distance2 = 10000000000
        #15 LOOP START 
        #sGS.getGhostPositions() != currentGameState.getGhostPositions() 결과 같음.
        for x,y in currentGameState.getGhostPositions():
            cur_mahattan2 = abs(pacman_x-x) + abs(pacman_y-y)
            if (min_distance2 > cur_mahattan2):
                min_distance2 = cur_mahattan2
        if (min_distance2 < 2):
            total_score -= 100
        else:
            total_score -= 1/min_distance2
            #total_score += min_distance2/(max_length) #멀면 멀수록 좋음 

        #16 score 계산
        #total_score = total_score + (1/min_distance) + (min_distance2/max_length)
        #return successorGameState.getScore()
        return total_score



def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    
    '''
    max_val = float("-inf")
    for action in currentGameState.getLegalActions():
        #1 현재의 팩맨 위치
        pacmanPos = currentGameState.getPacmanPosition()
        #2 현재 음식들의 위치 - grid class로 표현됨
        foodPos = currentGameState.getFood()
        #3 맵 정보 -> #4에서 가로 세로 길이 이용
        mapLayout = currentGameState.getWalls()
        #4 맨허튼 거리로 최대한 긴 거리일때 얼마인지 측정 (가로, 세로에 벽도 포함되서 4를 뺌)
        max_length = mapLayout.width + mapLayout.height - 4 # 맨허튼 거리기준 가장 긴 거리
        #5 action(특정 방향)으로 이동한 state의 정보를 받아옴
        sGS= currentGameState.generatePacmanSuccessor(action) # successorGameState 
        #6 이동했다 가정하고, 거기서 팩맨의 위치
        sGS_successor_position = sGS.getPacmanPosition()
        #7 이동했다 가정하고, 거기서 음식들의 위치
        sGS_food_Position = sGS.getFood()
        #8 이동했다 가정하고, possible next state, 만약 score가 같다면 이거 개수로 판별(가능성이 많은거니)
        possible_next_states = sGS.getLegalActions()
        #9 캡슐(아마 고스트 먹을 수 있게 하는 아이템)들의 위치
        cur_capsules_position = currentGameState.getCapsules()
        #10 위 정보들을 토대로 점수를 매길거임. 그거 init
        total_score = 0
        #11 현재 action으로 가면 food를 먹는지 확인 -> 먹는다면 score++
        if (foodPos[sGS_successor_position[0]][sGS_successor_position[1]]):
            #print("NEXT ACTION WILL GET FOOD!")
            total_score += 10
        #12 이동후의 팩맨 기준, 음식들 중 가장 최소거리 구하기 시작
        min_distance = 10000000000

        pacman_x = sGS_successor_position[0]
        pacman_y = sGS_successor_position[1]
        _x, _y = -1, -1
        for x in range(1, mapLayout.width-1): #0, -1은 벽이니까 제외.  range(1,4)
            for y in range(1, mapLayout.height-1): #0, -1은 벽이니까 제외. range(1,9)
                #print(foodPos[x][y], end='\t')
                if (foodPos[x][y] == True):
                    cur_mahattan = abs(pacman_x-x) + abs(pacman_y-y)
                    if (min_distance > cur_mahattan): #가장 작은 거리 업데이트
                        min_distance = cur_mahattan
                        _x, _y = x, y #only for debug. del it when it fin
        # score에 음식과의 최소거리의 역수 추가
        if (min_distance == 0): 
            pass #위에서 10점이나 줬자나
        else: 
            total_score += (1/min_distance)

        #14 이동후의 팩맨 기준, 유령과 팩맨의 최소거리 구하기 시작
        min_distance2 = 10000000000
        #15 LOOP START 
        for x,y in currentGameState.getGhostPositions():
            cur_mahattan2 = abs(pacman_x-x) + abs(pacman_y-y)
            if (min_distance2 > cur_mahattan2):
                min_distance2 = cur_mahattan2
        if (min_distance2 < 3):
            total_score -= 500
        else:
            total_score -= 1/min_distance2
            
        max_val = max(max_val, total_score)
    return max_val + currentGameState.getScore()
    '''
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
    Your minimax agent (question 2)
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
        #util.raiseNotDefined()
        """
        max -> min -> man -> min 순서로 recursively하게 호출해서 연산할거임.
                                    max    #depth=4의 트리. 종료 조건: depth=4 or isWin/isLose==True 
                            min             min
                        max     max      max     max
                    min   min min min  min min min min #노드가 2개가 아닐수도?
        내가 이해한 minNode에서 ghost기준 최선 선택인 이유.
        -팩맨은 유령의 최선의 선택들 중, 그러니까 팩맨기준 최악의 선택들 중에서 최선의 선택을 해야 함,
          why? 모든 경우에서의 최선을 택하면 똑똑한 유령에게 항상 죽을테니까.
        그래서 차악을 선택하는 방식임. 

        각 노드에서 갈 수 있는 방향들에 대한 점수 evaluation하고 max면 max(score1, score2,...) min이면 min(~) 

        - 유령이 여러개인 경우를 고려해야함. agentIndex=1,2,~

        """
        def maxNode(state, depth): #return [max_val, direction]
            if (depth==self.depth or state.isWin()==True or state.isLose()==True):
                return [self.evaluationFunction(state), 0]

            max_val = float("-inf") #max를 위해 -inf
            legal_actions = state.getLegalActions(0) #key=0이 default긴 함.
            '''
            for agentIndex in range(1, state.getNumAgents()-1): # 유령이 2개 이상인 경우 각각에 대해 돌림.
                for action in legal_actions: # 팩맨의 각 행동(이동방향)에 따라 loop.
                    successor = state.generateSuccessor(0, action) 
                    max_val = min(max_val, maxNode(successor, depth))
            for action in legal_actions: # 팩맨의 각 행동(이동방향)에 따라 loop.
                successor = state.generateSuccessor(0, action) 
                max_val = min(max_val, minNode(successor, depth+1, state.getNumAgents()-1))
            '''
            for action in legal_actions:
                successor = state.generateSuccessor(0, action)
                if (max_val < minNode(successor, depth, 1)[0]): 
                    # < 일때는 가만히 있음. <= 일때는 왔다갔다함.
                    max_val = minNode(successor, depth, 1)[0]
                    direction = action #max값을 위한 방향

            return [max_val, direction]

        def minNode(state, depth, agentIndex):
            if (depth==self.depth or state.isWin()==True or state.isLose()==True):
                # isWin등을 함수 재귀 호출 전에 비교해서 알아보고 그 안에서 처리한 후, 이때 그냥 값 반환하는건 어떰?
                #여기가 발동된다는건, max_val = max(max_val, minNode(successor, depth+1, agentIndex))에서
                #값이 전달되어야 한다는 의미. -> agentIndex에 따른 휴리스틱 값을 리턴?
                return [self.evaluationFunction(state), 0]
            min_val = float("inf")
            legal_actions = state.getLegalActions(agentIndex) #agentIndex. 즉 각 유령의 인덱스에 대한 LegalActions()

            if (agentIndex < state.getNumAgents()-1): #마지막 유령이 아니라면 
                for action in legal_actions:
                    successor = state.generateSuccessor(agentIndex, action)
                    min_val = min(min_val, minNode(successor, depth, agentIndex+1)[0])
            else: #end point
                for action in legal_actions: #그동안 쌓인 min_val들과 그 아래서 쌓인 maxNode값 간의 min_val
                    successor = state.generateSuccessor(agentIndex, action)
                    min_val = min(min_val, maxNode(successor, depth+1)[0])

            return [min_val, action] #단순히 maxNode의 리턴값과 형식을 같게하기 위해 한것. 실제로 action은 쓰지않음.

        #init process
        '''
        legal_actions를 for문으로 돌리면서 각 action들을 max인지 min인지에 넣어서 점수 비교 최고인곳 return
        '''
        legal_actions = gameState.getLegalActions()
        direction = Directions.STOP
        max_val = float("-inf")
        #print(maxNode(gameState, 0)[1], file=sys.stderr)
        #if (maxNode(gameState, 0)[1] == "Stop"):
        #    legal_actions.remove("Stop")
        #    return random.choice(legal_actions)
        return maxNode(gameState, 0)[1]




class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """
    '''
    maxNode에서는 alpha값을 update하며, beta값과 max_val를 비교한다. 
    minNode에서는 beta값을 update하며, alpha값과 min_val를 비교한다.
    
    maxNode에서는 if (max_val > beta):로 beta보다 큰 값이 생기면 그걸 바로 return하는.
    minNode에서는 if (min_val < alpha):로 alpha보다 작은값이 생기면 그걸 바로 return하는.
    if조건이 맞지 않으면, alpha와 beta를 update함.

    그 이외에는 minimax와 동일 
    '''

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()
        def maxNode(state, depth, alpha, beta): #return [max_val, direction]
            if (depth==self.depth or state.isWin()==True or state.isLose()==True):
                return [self.evaluationFunction(state), 0]

            max_val = float("-inf") #max를 위해 -inf
            legal_actions = state.getLegalActions(0) #key=0이 default긴 함.

            for action in legal_actions:
                successor = state.generateSuccessor(0, action)
                if (max_val < minNode(successor, depth, alpha, beta, 1)[0]): #max 역할
                    max_val = minNode(successor, depth, alpha, beta, 1)[0]
                    direction = action #max값을 위한 방향
                if (max_val > beta):
                    return [max_val, direction]
                alpha = max(alpha, max_val)

            return [max_val, direction]

        def minNode(state, depth, alpha, beta, agentIndex):
            if (depth==self.depth or state.isWin()==True or state.isLose()==True):
                return [self.evaluationFunction(state), 0]
            min_val = float("inf")
            legal_actions = state.getLegalActions(agentIndex) #agentIndex. 즉 각 유령의 인덱스에 대한 LegalActions()

            if (agentIndex < state.getNumAgents()-1): #마지막 유령이 아니라면 
                for action in legal_actions:
                    successor = state.generateSuccessor(agentIndex, action)
                    min_val = min(min_val, minNode(successor, depth, alpha, beta, agentIndex+1)[0])
                    if (min_val < alpha): 
                        return [min_val, 0]
                    beta = min(beta, min_val)
            else: #end point
                for action in legal_actions: #그동안 쌓인 min_val들과 그 아래서 쌓인 maxNode값 간의 min_val
                    successor = state.generateSuccessor(agentIndex, action)
                    min_val = min(min_val, maxNode(successor, depth+1, alpha, beta)[0])
                    if (min_val < alpha): 
                        return [min_val, 0]
                    beta = min(beta, min_val)

            return [min_val, action] #단순히 maxNode의 리턴값과 형식을 같게하기 위해 한것. 실제로 action은 쓰지않음.

        #init process
        '''
        legal_actions를 for문으로 돌리면서 각 action들을 max인지 min인지에 넣어서 점수 비교 최고인곳 return
        '''
        legal_actions = gameState.getLegalActions()
        direction = Directions.STOP
        max_val = float("-inf")
        #alpha, beta값은 한 트리 서칭내에서 global하게 쓰여야 하니 여기서 선언.
        alpha = float("-inf") 
        beta = float("inf")

        return maxNode(gameState, 0, alpha, beta)[1]

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
