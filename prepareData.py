import numpy as np
import pickle
import random
from AnalyticGeometryFunctions import computeAngleBetweenVectors


def getOptimalAction(agentState, targetState, actionSpace):
	relativeVector = np.array(targetState) - np.array(agentState)
	angleBetweenVectors = {computeAngleBetweenVectors(relativeVector, action): action for action in np.array(actionSpace)}
	optimalAction = angleBetweenVectors[min(angleBetweenVectors.keys())]
	return optimalAction


def generateOptimalPolicy(stateSpace, actionSpace):
	optimalPolicy = {(agentState + targetState): getOptimalAction(agentState, targetState, actionSpace) for agentState, targetState in stateSpace}
	return optimalPolicy


class SampleTrajectory():
	def __init__(self, maxTimeStep, transitionFunction, isTerminal, reset):
		self.maxTimeStep = maxTimeStep
		self.transitionFunction = transitionFunction
		self.isTerminal = isTerminal
		self.reset = reset

	def __call__(self, policy):
		initialAgentState, targetPosition = self.reset()
		# print('AgentState: {}'.format(initialAgentState))
		# print('TargetState: {}'.format(targetPosition))
		isTerminal = self.isTerminal
		while isTerminal(initialAgentState, targetPosition):
			initialAgentState, targetPosition = self.reset()
		oldState, action = initialAgentState, [0, 0]
		trajectory = []

		for time in range(self.maxTimeStep):
			action = policy[oldState + targetPosition]
			newState = self.transitionFunction(oldState, action)
			terminal = self.isTerminal(oldState, targetPosition)
			if terminal:
				break
			trajectory.append((list(oldState + targetPosition), action))
			oldState = newState
		return zip(*trajectory)


def generateData(sampleTrajectory, policy, trajNumber, path, actionSpace):
	totalStateBatch = []
	totalActionBatch = []
	for index in range(trajNumber):
		states, actions = sampleTrajectory(policy)
		totalStateBatch = totalStateBatch + list(states)
		oneHotAction = [[1 if (np.array(action) == np.array(actionSpace[index])).all() else 0 for index in range(len(actionSpace))] for action in actions]
		totalActionBatch = totalActionBatch + oneHotAction
	dataSet = list(zip(totalStateBatch, totalActionBatch))
	saveFile = open(path, "wb")
	pickle.dump(dataSet, saveFile)


def loadData(path):
	pklFile = open(path, "rb")
	dataSet = pickle.load(pklFile)
	pklFile.close()
	return dataSet


def sampleData(data, batchSize):
	batch = random.sample(data, batchSize)
	batchInput = [x for x, _ in batch]
	batchOutput = [y for _, y in batch]
	return batchInput, batchOutput