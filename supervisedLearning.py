import itertools as it
import numpy as np
from chasingTargetTrial.AnalyticGeometryFunctions import computeAngleBetweenVectors
import chasingTargetTrial.gridEnv as gridEnv
import chasingTargetTrial.neuralNetwork as nn


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


def sampleTotalTrajectory(sampleTrajectory):
	initialAgentState = agentStates[np.random.randint(0, len(agentStates))]
	targetPosition = targetStates[np.random.randint(0, len(targetStates))]
	# print('AgentState: {}'.format(initialAgentState))
	# print('TargetState: {}'.format(targetPosition))
	isTerminal = gridEnv.IsTerminal()
	while isTerminal(initialAgentState, targetPosition):
		initialAgentState = agentStates[np.random.randint(0, len(agentStates))]
		targetPosition = targetStates[np.random.randint(0, len(targetStates))]
	stateBatch, actionBatch = sampleTrajectory(None, initialAgentState, targetPosition)
	return stateBatch, actionBatch


if __name__ == '__main__':
	gridSize = 10
	actionSpace = [[0,1], [1,0], [-1,0], [0,-1], [1,1], [-1,-1], [1,-1], [-1,1]]
	agentStates = [state for state in it.product([x for x in range(10)], [y for y in range(10)])]
	targetStates = [state for state in it.product([x for x in range(10)], [y for y in range(10)])]
	stateSpace = [state for state in it.product(agentStates, targetStates)]
	numOfaction = len(actionSpace)
	numOfState = len(stateSpace)
	maxTimeStep = int(gridSize * gridSize / 2)
	learningRate = 0.01
	dataNum = 2000
	maxEpisode = 300
	totalStateBatch = []
	totalActionBatch = []
	print('Generating Optimal Policy...')
	optimalPolicy = generateOptimalPolicy(stateSpace, actionSpace)
	print('Optimal Policy Generated.')


	transitionFunction = gridEnv.TransitionFunction(gridSize)
	generatePolicyNet = nn.GeneratePolicyNet(numOfState, numOfaction, learningRate, optimalPolicy, actionSpace)
	model = generatePolicyNet(2, 128)
	graph = model.graph
	isTerminal = gridEnv.IsTerminal()
	sampleTrajectory = SampleTrajectory(maxTimeStep, transitionFunction, isTerminal, actionSpace, optimalPolicy)
	batch = [sampleTotalTrajectory(sampleTrajectory) for data in range(dataNum)]
	for stateBatch, actionBatch in batch:
		[totalStateBatch.append(np.array(state)) for state in stateBatch]
		[totalActionBatch.append(np.array(action)) for action in actionBatch]
	print(len(totalStateBatch))
	print(len(totalActionBatch))
	totalActionBatchLabel = [[1 if (action == actionSpace[index]).all() else 0 for index in range(len(actionSpace))] for action in totalActionBatch]

	for episode in range(maxEpisode):
		state_ = graph.get_tensor_by_name('inputs/state_:0')
		actionLabel_ = graph.get_tensor_by_name('inputs/actionLabel_:0')
		loss_ = graph.get_tensor_by_name('outputs/loss_:0')
		trainOpt_ = graph.get_operation_by_name('train/adamOpt_')
		loss, trainOpt = model.run([loss_, trainOpt_], feed_dict={state_ : totalStateBatch, actionLabel_ : totalActionBatchLabel})
		print(loss)

	print("Testing Model...")
	totalPlayStates = 0
	totalOptimalStates = 0
	testTimes = 100
	optimalStates = None
	playStates = None
	optimal = None
	playEpisode = None
	for index in range(0, testTimes):
		initialAgentState = agentStates[np.random.randint(0, len(agentStates))]
		targetPosition = targetStates[np.random.randint(0, len(targetStates))]
		# initialAgentState = (6,4)
		# targetPosition = (1,2)
		while isTerminal(initialAgentState, targetPosition):
			initialAgentState = agentStates[np.random.randint(0, len(agentStates))]
			targetPosition = targetStates[np.random.randint(0, len(targetStates))]
		approximatePolicy = ApproximatePolicy(actionSpace)
		samplePlay = SampleTrajectory(maxTimeStep, transitionFunction, isTerminal, actionSpace, optimalPolicy)
		policy = lambda state: approximatePolicy(state, model)
		playEpisode = samplePlay(policy, initialAgentState, targetPosition)
		optimal = samplePlay(None, initialAgentState, targetPosition)
		optimalStates, optimalActions = optimal
		totalOptimalStates += len(optimalStates)
		playStates, playActions = playEpisode
		totalPlayStates += len(playStates)

	print("Play: {}, Optimal:{}".format(totalPlayStates/testTimes, totalOptimalStates/testTimes))
