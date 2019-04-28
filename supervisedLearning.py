import itertools as it
import numpy as np
from AnalyticGeometryFunctions import computeAngleBetweenVectors
import gridEnv
import neuralNetwork


def getOptimalAction(agentState, targetState, actionSpace):
	relativeVector = np.array(targetState) - np.array(agentState)
	angleBetweenVectors = {computeAngleBetweenVectors(relativeVector, action): action for action in np.array(actionSpace)}
	optimalAction = angleBetweenVectors[min(angleBetweenVectors.keys())]
	return optimalAction


def generateOptimalPolicy(stateSpace, actionSpace):
	optimalPolicy = {(agentState, targetState): getOptimalAction(agentState, targetState, actionSpace) for agentState, targetState in stateSpace}
	return optimalPolicy


class ApproximatePolicy():
	def __init__(self, actionSpace):
		self.actionSpace = actionSpace
		self.numActionSpace = len(self.actionSpace)
	def __call__(self, stateBatch, model):
		graph = model.graph
		state_ = graph.get_tensor_by_name('inputs/state_:0')
		actionDistribution_ = graph.get_tensor_by_name('outputs/actionDistribution_:0')
		actionDistributionBatch = model.run(actionDistribution_, feed_dict = {state_ : stateBatch})
		# print(stateBatch)
		actionIndexBatch = [np.random.choice(range(self.numActionSpace), p = actionDistribution) for actionDistribution in actionDistributionBatch]
		actionBatch = np.array([self.actionSpace[actionIndex] for actionIndex in actionIndexBatch])
		# print(actionBatch)
		return actionBatch


class SampleTrajectory():
	def __init__(self, maxTimeStep, transitionFunction, isTerminal, actionSpace, optimalPolicy):
		self.maxTimeStep = maxTimeStep
		self.transitionFunction = transitionFunction
		self.isTerminal = isTerminal
		self.actionSpace = actionSpace
		self.optimalPolicy = optimalPolicy

	def __call__(self, actor, initialAgentState, targetPosition):
		oldState, action = initialAgentState, [0, 0]
		trajectory = []

		for time in range(self.maxTimeStep):
			if actor is None:
				action = list(self.optimalPolicy[(oldState, targetPosition)])
			else:
				action = actor([list(oldState + targetPosition)])[0]
			newState = self.transitionFunction(oldState, action)
			terminal = self.isTerminal(oldState, targetPosition)
			if terminal:
				break
			trajectory.append((list(oldState + targetPosition), action))
			oldState = newState
		return zip(*trajectory)


def sampleTotalTrajectory(sampleTrajectory, agentStatesSpace, targetStatesSpace):
	initialAgentState = agentStatesSpace[np.random.randint(0, len(agentStatesSpace))]
	targetPosition = targetStatesSpace[np.random.randint(0, len(targetStatesSpace))]
	# print('AgentState: {}'.format(initialAgentState))
	# print('TargetState: {}'.format(targetPosition))
	isTerminal = gridEnv.IsTerminal()
	while isTerminal(initialAgentState, targetPosition):
		initialAgentState = agentStatesSpace[np.random.randint(0, len(agentStatesSpace))]
		targetPosition = targetStatesSpace[np.random.randint(0, len(targetStatesSpace))]
	stateBatch, actionBatch = sampleTrajectory(None, initialAgentState, targetPosition)
	return stateBatch, actionBatch
