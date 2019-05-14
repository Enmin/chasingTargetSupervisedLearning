import numpy as np
import pickle
import random
import functools as ft
from anytree import AnyNode as Node
from mcts import MCTS, CalculateScore, GetActionPrior, selectNextRoot, SelectChild, Expand, RollOut, backup, InitializeChildren
import pygame as pg


class SampleTrajectory:
	def __init__(self, maxTimeStep, transitionFunction, isTerminal, reset):
		self.maxTimeStep = maxTimeStep
		self.transitionFunction = transitionFunction
		self.isTerminal = isTerminal
		self.reset = reset

	def __call__(self, policy):
		state = self.reset()
		while self.isTerminal(state):
			state = self.reset()
		trajectory = []
		for _ in range(self.maxTimeStep):
			action = policy(state)
			trajectory.append((state, action))
			newState = self.transitionFunction(state, action)
			if self.isTerminal(newState):
				break
			state = newState
		return trajectory


class SampleTrajectoryWithMCTS:
	def __init__(self, maxTimeStep, isTerminal, reset, render):
		self.maxTimeStep = maxTimeStep
		self.isTerminal = isTerminal
		self.reset = reset
		self.render = render

	def __call__(self, mcts):
		rootNode = self.reset()
		currState = list(rootNode.id.values())[0]
		while self.isTerminal(currState):
			rootNode = self.reset()
			currState = list(rootNode.id.values())[0]

		trajectory = []
		for _ in range(self.maxTimeStep):
			state = list(rootNode.id.values())[0]
			self.render(state)
			if self.isTerminal(state):
				break
			nextNode = mcts(rootNode)
			action = list(nextNode.id.keys())[0]
			trajectory.append((state, action))
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
				 partialTrajSize=None):
	totalStateBatch = []
	totalActionBatch = []
	totalRewardBatch = []
	for index in range(trajNumber):
		if index % 100 == 0: print("{} trajectories generated".format(index))

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
		oneHotActions = [[1 if (np.array(action) == np.array(actionSpace[index])).all() else 0 for index in range(len(actionSpace))] for action in actions]
		totalStateBatch += states
		totalActionBatch += oneHotActions

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


def prepareMCTSData():
	import sheepEscapingEnv as env
	actionSpace = [(0, 1), (1, 0), (-1, 0), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)]
	numActionSpace = 8
	xBoundary = [0, 180]
	yBoundary = [0, 180]
	extendedBound = 30
	vel = 20
	maxTraj = 10
	wolfHeatSeekingPolicy = env.WolfHeatSeekingPolicy(actionSpace)
	# Hyper-parameters
	numSimulations = 600
	maxRunningSteps = 70
	rewardFunction = lambda state, action: 1
	# MCTS algorithm
	# Select child
	cInit = 1
	cBase = 1
	calculateScore = CalculateScore(cInit, cBase)
	selectChild = SelectChild(calculateScore)

	# render
	screen = pg.display.set_mode([xBoundary[1] + extendedBound, yBoundary[1] + extendedBound])
	screenColor = [255, 255, 255]
	circleColorList = [[50, 255, 50], [50, 50, 50], [50, 50, 50], [50, 50, 50], [50, 50, 50], [50, 50, 50],
	                   [50, 50, 50], [50, 50, 50], [50, 50, 50]]
	circleSize = 8
	saveImage = True
	numOneAgentState = 2
	positionIndex = [0, 1]
	numAgent = 2
	savePath = './sheepDemo'
	render = env.Render(numAgent, numOneAgentState, positionIndex, screen, screenColor, circleColorList, circleSize,
	                    saveImage, savePath)
	# expand
	transition = env.TransitionFunction(xBoundary, yBoundary, vel, wolfHeatSeekingPolicy)
	getActionPrior = GetActionPrior(actionSpace)
	isTerminal = env.IsTerminal(minDistance=vel+0.5)
	reset = env.ResetForMCTS(xBoundary, yBoundary, actionSpace, numActionSpace)
	initializeChildren = InitializeChildren(actionSpace, transition, getActionPrior)
	expand = Expand(transition, isTerminal, initializeChildren)
	# selectNextRoot = selectNextRoot

	# Rollout
	rolloutPolicy = lambda state: actionSpace[np.random.choice(range(numActionSpace))]
	maxRollOutSteps = 10
	rollout = RollOut(rolloutPolicy, maxRollOutSteps, transition, rewardFunction, isTerminal)

	mcts = MCTS(numSimulations, selectChild, expand, rollout, backup, selectNextRoot)
	sampleTraj = SampleTrajectoryWithMCTS(maxRunningSteps, isTerminal, reset, render)
	sampleTraj(mcts)
if __name__ == "__main__":
	# prepareDataContinuousEnv()
	prepareMCTSData()