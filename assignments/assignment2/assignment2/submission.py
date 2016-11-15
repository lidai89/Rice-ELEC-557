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
  def __init__(self):
    self.lastPositions = []
    self.dc = None


  def getAction(self, gameState):
    """
    getAction chooses among the best options according to the evaluation function.

    getAction takes a GameState and returns some Directions.X for some X in the set {North, South, West, East, Stop}
    ------------------------------------------------------------------------------
    Description of GameState and helper functions:

    A GameState specifies the full game state, including the food, capsules,
    agent configurations and score changes. In this function, the |gameState| argument 
    is an object of GameState class. Following are a few of the helper methods that you 
    can use to query a GameState object to gather information about the present state 
    of Pac-Man, the ghosts and the maze.
    
    gameState.getLegalActions(): 
        Returns the legal actions for the agent specified. Returns Pac-Man's legal moves by default.

    gameState.generateSuccessor(agentIndex, action): 
        Returns the successor state after the specified agent takes the action. 
        Pac-Man is always agent 0.

    gameState.getPacmanState():
        Returns an AgentState object for pacman (in game.py)
        state.pos gives the current position
        state.direction gives the travel vector

    gameState.getGhostStates():
        Returns list of AgentState objects for the ghosts

    gameState.getNumAgents():
        Returns the total number of agents in the game

    
    The GameState class is defined in pacman.py and you might want to look into that for 
    other helper methods, though you don't need to.
    
    """
    # Collect legal moves and successor states
    legalMoves = gameState.getLegalActions()

    # Choose one of the best actions
    scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best

    return legalMoves[chosenIndex]

  def evaluationFunction(self, currentGameState, action):
    """
    The evaluation function takes in the current and proposed successor
    GameStates (pacman.py) and returns a number, where higher numbers are better.

    The code below extracts some useful information from the state, like the
    remaining food (oldFood) and Pacman position after moving (newPos).
    newScaredTimes holds the number of moves that each ghost will remain
    scared because of Pacman having eaten a power pellet.
    """
    # Useful information you can extract from a GameState (pacman.py)
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = successorGameState.getPacmanPosition()
    oldFood = currentGameState.getFood()
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
    Your minimax agent (problem 1)
    
    The auto grader will check the running time of your algorithm. Friendly reminder: passing the auto grader
    does not necessarily mean that your algorithm is correct.
  """

  
#  def getAction(self, gameState):
#    legalMoves = gameState.getLegalActions()
#
#    # Choose one of the best actions
#    scores = [self.evaluationFunction(gameState) for action in legalMoves]
#    bestScore = max(scores)
#    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
#    chosenIndex = random.choice(bestIndices) # Pick randomly among the best
#
#    return legalMoves[chosenIndex]
  def getAction(self, gameState):
       
        res = self.value(gameState, 0)
        return res[0]

  def value(self, gameState, depth):
        if depth == self.depth * gameState.getNumAgents() or gameState.isWin() or gameState.isLose():
            return (None, self.evaluationFunction(gameState))
        if depth % gameState.getNumAgents() == 0:
            # pacman
            return self.maxFunc(gameState, depth)
        else:
            # ghosts
            return self.minFunc(gameState, depth)

  def minFunc(self, gameState, depth):
        actions = gameState.getLegalActions(depth % gameState.getNumAgents())
        if len(actions) == 0:
            return (None, self.evaluationFunction(gameState))

        min_val = (None, float("inf"))
        for action in actions:
            succ = gameState.generateSuccessor(depth % gameState.getNumAgents(), action)
            res = self.value(succ, depth+1)
            if res[1] < min_val[1]:
                min_val = (action, res[1])
        return min_val

  def maxFunc(self, gameState, depth):
        actions = gameState.getLegalActions(0)
        if len(actions) == 0:
            return (None, self.evaluationFunction(gameState))

        max_val = (None, -float("inf"))
        for action in actions:
            succ = gameState.generateSuccessor(0, action)
            res = self.value(succ, depth+1)
            if res[1] > max_val[1]:
                max_val = (action, res[1])
        return max_val   
      





class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning (problem 2)
    
    The auto grader will check the running time of your algorithm. Friendly reminder: passing the auto grader
    does not necessarily mean your algorithm is correct.
  """

  def getAction(self, gameState):
    """
      Returns the minimax action using self.depth and self.evaluationFunction
      
      The same methods used in MinimaxAgent should also be useful here   
      
      It is recommended you have separate functions: value(), max_value(), and min_value() as in the slides
      and call these functions here to make the code clear   
    """

    # BEGIN_YOUR_CODE (around 45 lines of code expected)
    raise Exception("Not implemented yet")
    # END_YOUR_CODE

class ExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent (problem 3)
    
    The auto grader will check the running time of your algorithm. Friendly reminder: passing the auto grader
    does not necessarily mean your algorithm is correct.
  """

  def getAction(self, gameState):
    """
      Returns the expectimax action using self.depth and self.evaluationFunction

      All ghosts should be modeled as choosing uniformly at random from their
      legal moves.
      
      The same methods used in MinimaxAgent should also be useful here   
      
      It is recommended you have separate functions: value(), max_value(), and expect_value() as in the slides
      and call these functions here to make the code clear
    """

    # BEGIN_YOUR_CODE (around 35 lines of code expected)
    raise Exception("Not implemented yet")
    # END_YOUR_CODE

def betterEvaluationFunction(currentGameState):
  """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (problem 4).

    DESCRIPTION: <write something here so we know what you did>
  """

  # BEGIN_YOUR_CODE (around 50 lines of code expected)
  raise Exception("Not implemented yet")
  # END_YOUR_CODE

# Abbreviation
better = betterEvaluationFunction


