import numpy as np


def checkBound(state, gridSize):
	xPos, yPos = state
	if xPos >= gridSize or xPos < 0:
		return False
	elif yPos >= gridSize or yPos < 0:
		return False
	return True


class TransitionFunction():
	def __init__(self, gridSize):
		self.gridSize = gridSize

	def __call__(self, state, action):
		newState = np.array(state) + np.array(action)
		newState = tuple(newState)
		if checkBound(newState, self.gridSize):
			return newState
		return state


class IsTerminal():
	def __init__(self):
		return

	def __call__(self, state, targetPosition):
		if (np.array(state) == targetPosition).all():
			return True
		return False


class Reset():
	def __init__(self, actionSpace, agentStateSpace, targetStateSpace):
		self.actionSpace = actionSpace
		self.agentStateSpace = agentStateSpace
		self.targetStateSpace = targetStateSpace

	def __call__(self):
		initialAgentState = self.agentStateSpace[np.random.randint(0, len(self.agentStateSpace))]
		targetPosition = self.targetStateSpace[np.random.randint(0, len(self.targetStateSpace))]
		return initialAgentState, targetPosition