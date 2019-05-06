import numpy as np
from AnalyticGeometryFunctions import computeVectorNorm


def checkBound(state, xBoundary, yBoundary):
	xMin, xMax = xBoundary
	yMin, yMax = yBoundary
	xPos, yPos = state
	if xPos >= xMax or xPos <= xMin:
		return False
	elif yPos >= yMax or yPos <= yMin:
		return False
	return True


def getEachState(state):
	return state[:2], state[2:]


class TransitionFunction():
	def __init__(self, xBoundary, yBoundary, velocity):
		self.xBoundary = xBoundary
		self.yBoundary = yBoundary
		self.velocity = velocity

	def __call__(self, state, action):
		agentState, targetPosition = getEachState(state)
		actionMagnitude = computeVectorNorm(np.array(action))
		modifiedAction = np.array(action) * self.velocity / actionMagnitude
		newAgentState = np.array(agentState) + modifiedAction
		if checkBound(newAgentState, self.xBoundary, self.yBoundary):
			return np.concatenate([newAgentState, targetPosition])
		return np.concatenate([agentState, targetPosition])


class IsTerminal():
	def __init__(self, minDistance):
		self.minDistance = minDistance
		return

	def __call__(self, state):
		agentState, targetPosition = getEachState(state)
		relativeVector = np.array(agentState) - np.array(targetPosition)
		relativeDistance = computeVectorNorm(relativeVector)
		if relativeDistance <= self.minDistance:
			return True
		return False


class Reset():
	def __init__(self, xBoundary, yBoundary):
		self.xBoundary = xBoundary
		self.yBoundary = yBoundary

	def __call__(self):
		xMin, xMax = self.xBoundary
		yMin, yMax = self.yBoundary
		initialAgentState = np.array([np.random.uniform(xMin, xMax), np.random.uniform(yMin, yMax)])
		targetPosition = np.array([np.random.uniform(xMin, xMax), np.random.uniform(yMin, yMax)])
		while not (checkBound(initialAgentState, self.xBoundary, self.yBoundary) and checkBound(targetPosition, self.xBoundary, self.yBoundary)):
			initialAgentState = np.array([np.random.uniform(xMin, xMax), np.random.uniform(yMin, yMax)])
			targetPosition = np.array([np.random.uniform(xMin, xMax), np.random.uniform(yMin, yMax)])
		return np.concatenate([initialAgentState, targetPosition])