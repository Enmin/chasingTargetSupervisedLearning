import numpy as np
from AnalyticGeometryFunctions import computeVectorNorm, computeAngleBetweenVectors
import pygame as pg
import os

numStateSpace = 4
actionSpace = [[0, 1], [1, 0], [-1, 0], [0, -1], [1, 1], [-1, -1], [1, -1], [-1, 1]]
numActionSpace = len(actionSpace)
xBoundary = [0, 180]
yBoundary = [0, 180]
vel = 1


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
	def __init__(self, xBoundary, yBoundary, velocity, wolfPolicy):
		self.xBoundary = xBoundary
		self.yBoundary = yBoundary
		self.velocity = velocity
		self.wolfPolicy = wolfPolicy

	def __call__(self, state, action):
		oldSheepPos, oldWolfPos = getEachState(state)
		# wolf
		wolfAction = self.wolfPolicy(state)
		wolfActionMagnitude = computeVectorNorm(np.array(wolfAction))
		modifiedWolfAction = np.array(wolfAction) * self.velocity / wolfActionMagnitude
		newWolfPos = np.array(oldWolfPos) + modifiedWolfAction
		# sheep
		sheepActionMagnitude = computeVectorNorm(np.array(action))
		modifiedSheepAction = np.array(action) * self.velocity / sheepActionMagnitude
		newSheepPos = np.array(oldSheepPos) + modifiedSheepAction
		if checkBound(newSheepPos, self.xBoundary, self.yBoundary):
			sheepPos = newSheepPos
		else:
			sheepPos = oldSheepPos
		if checkBound(newWolfPos, self.xBoundary, self.yBoundary):
			wolfPos = newWolfPos
		else:
			wolfPos = oldWolfPos
		return np.concatenate([sheepPos, wolfPos])


class IsTerminal():
	def __init__(self, minDistance):
		self.minDistance = minDistance

	def __call__(self, state):
		sheepState, wolfState = getEachState(state)
		relativeVector = np.array(sheepState) - np.array(wolfState)
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


class FixedReset():
	def __init__(self, xBoundary, yBoundary):
		self.xBoundary = xBoundary
		self.yBoundary = yBoundary

	def __call__(self):
		xMin, xMax = self.xBoundary
		yMin, yMax = self.yBoundary
		initialAgentState = np.array([np.random.uniform(xMin, xMax), np.random.uniform(yMin, yMax)])
		targetPosition = np.array([np.random.uniform(xMin, xMax), np.random.uniform(yMin, yMax)])
		initialDistance = computeVectorNorm(targetPosition - initialAgentState)
		while not (checkBound(initialAgentState, self.xBoundary, self.yBoundary) and checkBound(targetPosition, self.xBoundary, self.yBoundary) and initialDistance >= 20):
			initialAgentState = np.array([np.random.uniform(xMin, xMax), np.random.uniform(yMin, yMax)])
			targetPosition = np.array([np.random.uniform(xMin, xMax), np.random.uniform(yMin, yMax)])
			initialDistance = computeVectorNorm(targetPosition - initialAgentState)
		return np.concatenate([initialAgentState, targetPosition])


class Render():
	def __init__(self, numAgent, numOneAgentState, positionIndex, screen, screenColor, circleColorList, circleSize, saveImage, saveImagePath):
		self.numAgent = numAgent
		self.numOneAgentState = numOneAgentState
		self.positionIndex = positionIndex
		self.screen = screen
		self.screenColor = screenColor
		self.circleColorList = circleColorList
		self.circleSize = circleSize
		self.saveImage = saveImage
		self.saveImagePath = saveImagePath
	def __call__(self, state):
		for j in range(1):
			for event in pg.event.get():
				if event.type == pg.QUIT:
					pg.quit()
			self.screen.fill(self.screenColor)
			for i in range(self.numAgent):
				oneAgentState = state[self.numOneAgentState * i: self.numOneAgentState * (i + 1)]
				oneAgentPosition = oneAgentState[min(self.positionIndex): max(self.positionIndex) + 1]
				pg.draw.circle(self.screen, self.circleColorList[i], [np.int(oneAgentPosition[0]),np.int(oneAgentPosition[1])], self.circleSize)
			pg.display.flip()
			if self.saveImage==True:
				filenameList = os.listdir(self.saveImagePath)
				pg.image.save(self.screen, self.saveImagePath+'/'+str(len(filenameList))+'.png')
			pg.time.wait(1)


class WolfHeatSeekingPolicy:
	def __init__(self, actionSpace):
		self.actionSpace = actionSpace

	def __call__(self, state):
		sheepState, wolfState = getEachState(state)
		relativeVector = np.array(sheepState) - np.array(wolfState)
		angleBetweenVectors = {computeAngleBetweenVectors(relativeVector, action): action for action in
							   np.array(self.actionSpace)}
		action = angleBetweenVectors[min(angleBetweenVectors.keys())]
		return action


class SheepNaiveEscapingPolicy:
	def __init__(self, actionSpace):
		self.actionSpace = actionSpace

	def __call__(self, state):
		sheepState, wolfState = getEachState(state)
		relativeVector = np.array(wolfState) - np.array(sheepState)
		angleBetweenVectors = {computeAngleBetweenVectors(relativeVector, action): action for action in
							   np.array(self.actionSpace)}
		action = angleBetweenVectors[max(angleBetweenVectors.keys())]
		return action


class SheepRandomPolicy:
	def __init__(self, actionSpace):
		self.actionSpace = actionSpace

	def __call__(self, state):
		actionIndex = np.random.randint(len(actionSpace))
		action = actionSpace[actionIndex]
		return action