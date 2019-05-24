import numpy as np
import pickle
import random
import functools as ft
from anytree import AnyNode as Node
from mcts import MCTS, CalculateScore, GetActionPrior, selectNextRoot, SelectChild, Expand, RollOut, backup, InitializeChildren


class SampleTrajectory:
	def __init__(self, maxTimeStep, transitionFunction, isTerminal, reset, render=None):
		self.maxTimeStep = maxTimeStep
		self.transitionFunction = transitionFunction
		self.isTerminal = isTerminal
		self.reset = reset
		self.render = render

	def __call__(self, policy):
		state = self.reset()
		while self.isTerminal(state):
			state = self.reset()
		trajectory = []
		for _ in range(self.maxTimeStep):
			if self.render is not None:
				self.render(state)
			action = policy(state)
			trajectory.append((state, action))
			newState = self.transitionFunction(state, action)
			if self.isTerminal(newState):
				break
			state = newState
		return trajectory


class SampleTrajectoryWithMCTS:
	def __init__(self, maxTimeStep, isTerminal, reset, render=None):
		self.maxTimeStep = maxTimeStep
		self.isTerminal = isTerminal
		self.reset = reset
		self.render = render

	def __call__(self, mcts):
		currState = self.reset()
		while self.isTerminal(currState):
			currState = self.reset()
		rootNode = Node(id={None: currState}, num_visited=0, sum_value=0, is_expanded=True)

		trajectory = []
		for _ in range(self.maxTimeStep):
			state = list(rootNode.id.values())[0]
			if self.render is not None:
				self.render(state)
			if self.isTerminal(state):
				break
			actionDict = mcts(rootNode)
			actionDistribution = np.array(list(actionDict.values()))
			maxIndex = np.argmax(actionDistribution).flatten()
			selected_child_index = np.random.choice(maxIndex)
			selected_child = rootNode.children[selected_child_index]
			next_root_id = selected_child.id
			nextNode = Node(id=next_root_id, num_visited=0, sum_value=0, is_expanded=False)
			trajectory.append((state, actionDistribution))
			rootNode = nextNode
		return trajectory


class AccumulateRewards():
	def __init__(self, decay, rewardFunction):
		self.decay = decay
		self.rewardFunction = rewardFunction
	def __call__(self, trajectory):
		rewards = [self.rewardFunction(state, action) for state, action in trajectory]
		accumulateReward = lambda accumulatedReward, reward: self.decay * accumulatedReward + reward
		accumulatedRewards = np.array([ft.reduce(accumulateReward, reversed(rewards[TimeT: ])) for TimeT in range(len(rewards))])
		return accumulatedRewards


def generateData(sampleTrajectory, accumulateRewards, policy, actionSpace, trajNumber, path, withReward=True,
				 partialTrajSize=None, reportInterval=100):
	totalStateBatch = []
	totalActionBatch = []
	totalRewardBatch = []
	for index in range(trajNumber):
		if index % reportInterval == 0: print("{} trajectories generated".format(index))

		trajectory = sampleTrajectory(policy)
		length = len(trajectory)

		if (partialTrajSize is None) or (partialTrajSize >= length):
			selectedTimeSteps = list(range(length))
		else:
			# selectedTimeSteps = random.sample(list(range(length)), partialTrajSize)
			selectedTimeSteps = list(range(partialTrajSize))

		if withReward:
			accumulatedRewards = accumulateRewards(trajectory)
			partialAccumulatedRewards = np.array([accumulatedRewards[t] for t in selectedTimeSteps])
			totalRewardBatch.append(partialAccumulatedRewards)

		partialTrajectory = [trajectory[t] for t in selectedTimeSteps]
		states, actions = zip(*partialTrajectory)
		# oneHotActions = [[1 if (np.array(action) == np.array(actionSpace[index])).all() else 0 for index in range(len(actionSpace))] for action in actions]
		totalStateBatch += states
		totalActionBatch += actions

	totalStateBatch = np.array(totalStateBatch)
	totalActionBatch = np.array(totalActionBatch)

	if withReward:
		totalRewardBatch = np.concatenate(totalRewardBatch).reshape(-1, 1)
		dataSet = list(zip(totalStateBatch, totalActionBatch, totalRewardBatch))
	else:
		dataSet = list(zip(totalStateBatch, totalActionBatch))

	saveFile = open(path, "wb")
	pickle.dump(dataSet, saveFile)
	return dataSet


def loadData(path):
	pklFile = open(path, "rb")
	dataSet = pickle.load(pklFile)
	pklFile.close()
	return dataSet


def sampleData(data, batchSize):
	batch = [list(varBatch) for varBatch in zip(*random.sample(data, batchSize))]
	return batch


def renderDataSet(path, render):
	dataset = loadData(path)
	for i in range(len(dataset)):
		state, action, reward = dataset[i]
		render(state)


def prepareDataContinuousEnv():
	import continuousEnv as env
	xbound = [0, 180]
	ybound = [0, 180]
	vel = 1
	transitionFunction = env.TransitionFunction(xbound, ybound, vel)
	isTerminal = env.IsTerminal(vel+.5)
	reset = env.Reset(xbound, ybound)

	maxTimeStep = 10000
	sampleTraj = SampleTrajectory(maxTimeStep, transitionFunction, isTerminal, reset)

	decay = 0.99
	rewardFunction = lambda state, action: -1
	accumulateRewards = AccumulateRewards(decay, rewardFunction)

	policy = env.OptimalPolicy(env.actionSpace)
	trajNum = 2000
	partialTrajSize = 5
	path = "./continuous_data_with_reward.pkl"
	data = generateData(sampleTraj, accumulateRewards, policy, env.actionSpace, trajNum, path, withReward=True,
						partialTrajSize=partialTrajSize)

	print("{} data points in {}".format(len(data), path))

	# data = loadData(path)
	# for d in data: print(d)

	# batch = sampleData(data, 5)
	# for b in batch: print(b)


def prepareSheepEscapingEnvData():
	import sheepEscapingEnv as env
	actionSpace = env.actionSpace
	numActionSpace = env.numActionSpace
	xBoundary = env.xBoundary
	yBoundary = env.yBoundary
	vel = env.vel
	wolfHeatSeekingPolicy = env.WolfHeatSeekingPolicy(actionSpace)
	transition = env.TransitionFunction(xBoundary, yBoundary, vel, wolfHeatSeekingPolicy)
	isTerminal = env.IsTerminal(minDistance=vel + 5)
	reset = env.Reset(xBoundary, yBoundary)

	rewardFunction = lambda state, action: 1

	cInit = 1
	cBase = 1
	calculateScore = CalculateScore(cInit, cBase)
	selectChild = SelectChild(calculateScore)

	getActionPrior = GetActionPrior(actionSpace)
	initializeChildren = InitializeChildren(actionSpace, transition, getActionPrior)
	expand = Expand(transition, isTerminal, initializeChildren)

	maxRollOutSteps = 10
	numSimulations = 600
	maxTrajLen = 100
	rolloutPolicy = lambda state: actionSpace[np.random.choice(range(numActionSpace))]
	rollout = RollOut(rolloutPolicy, maxRollOutSteps, transition, rewardFunction, isTerminal)
	mcts = MCTS(numSimulations, selectChild, expand, rollout, backup, selectNextRoot)
	sampleTraj = SampleTrajectoryWithMCTS(maxTrajLen, isTerminal, reset)

	# policy = env.SheepNaiveEscapingPolicy(actionSpace)
	# sampleTraj = SampleTrajectory(maxRunningSteps, transition, isTerminal, reset, render=None)

	rewardDecay = 0.99
	accumulateRewards = AccumulateRewards(rewardDecay, rewardFunction)

	trajNum = 500
	partialTrajSize = None
	path = "./500trajs_sheepEscapingEnv_data_actionDist.pkl"
	reportInterval = 10
	data = generateData(sampleTraj, accumulateRewards, mcts, actionSpace, trajNum, path, withReward=True, partialTrajSize=partialTrajSize, reportInterval=reportInterval)

	print("{} data points in {}".format(len(data), path))

	# data = loadData(path)
	# for d in data: print(d)

	# batch = sampleData(data, 5)
	# for b in batch: print(b)


if __name__ == "__main__":
	# prepareDataContinuousEnv()
	np.random.seed(129)
	prepareSheepEscapingEnvData()
