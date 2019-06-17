import numpy as np
import pickle
import random
import functools as ft


class SampleTrajectory:
	def __init__(self, maxTimeStep, transition, isTerminal, reset, includeTerminalState=False):
		self.maxTimeStep = maxTimeStep
		self.transition = transition
		self.isTerminal = isTerminal
		self.reset = reset
		self.includeTerminalState = includeTerminalState

	def __call__(self, policy):
		state = self.reset()
		while self.isTerminal(state):
			state = self.reset()
		trajectory = []
		for _ in range(self.maxTimeStep):
			action = policy(state)
			trajectory.append((state, action))
			newState = self.transition(state, action)
			if self.isTerminal(newState):
				if self.includeTerminalState:
					trajectory.append((newState, None))
				break
			state = newState
		return trajectory


def greedyActionFromDist(actionDist):
	actions = list(actionDist.keys())
	probs = list(actionDist.values())
	maxIndices = np.argwhere(probs == np.max(probs)).flatten()
	selectedIndex = np.random.choice(maxIndices)
	selectedAction = actions[selectedIndex]
	return selectedAction


class SampleTrajectoryWithActionDistribution:
	def __init__(self, maxTimeStep, transition, isTerminal, reset, distToAction, includeTerminalState=False):
		self.maxTimeStep = maxTimeStep
		self.transition = transition
		self.isTerminal = isTerminal
		self.reset = reset
		self.distToAction = distToAction
		self.includeTerminalState = includeTerminalState

	def __call__(self, policy):
		state = self.reset()
		while self.isTerminal(state):
			state = self.reset()
		trajectory = []
		for _ in range(self.maxTimeStep):
			actionDist = policy(state)
			action = self.distToAction(actionDist)
			trajectory.append((state, actionDist))
			newState = self.transition(state, action)
			if self.isTerminal(newState):
				if self.includeTerminalState:
					trajectory.append((newState, None))
				break
			state = newState
		return trajectory


def trajDistToLabel(traj):
	trajWithLabel = [(state, np.array(list(actionDist.values()))) for state, actionDist in traj]
	return trajWithLabel


def trajActionToLabel(actionSpace, traj):
	actionToOneHot = lambda action: np.array([1 if (np.array(action) == np.array(a)).all() else 0 for a in actionSpace])
	trajWithLabel = [(state, actionToOneHot(action)) for state, action in traj]
	return trajWithLabel


class AccumulateRewards:
	def __init__(self, decay, rewardFunction):
		self.decay = decay
		self.rewardFunction = rewardFunction

	def __call__(self, trajectory):
		rewards = [self.rewardFunction(state, action) for state, action in trajectory]
		accumulateReward = lambda accumulatedReward, reward: self.decay * accumulatedReward + reward
		accumulatedRewards = np.array([ft.reduce(accumulateReward, reversed(rewards[TimeT: ])) for TimeT in range(len(rewards))])
		return accumulatedRewards


def addValuesToTraj(traj, trajValueFunc):
	values = trajValueFunc(traj)
	trajWithValues = [(s, a, v) for (s, a), v in zip(traj, values)]
	return trajWithValues


def sampleData(data, batchSize):
	batch = [list(varBatch) for varBatch in zip(*random.sample(data, batchSize))]
	return batch


def main():
	import sheepEscapingEnv as env
	actionSpace = env.actionSpace
	xBoundary = env.xBoundary
	yBoundary = env.yBoundary
	vel = env.vel
	wolfHeatSeekingPolicy = env.WolfHeatSeekingPolicy(actionSpace)
	transition = env.TransitionFunction(xBoundary, yBoundary, vel, wolfHeatSeekingPolicy)
	isTerminal = env.IsTerminal(minDistance=vel + 5)
	reset = env.Reset(xBoundary, yBoundary)

	maxTimeStep = 5
	sampleTraj = SampleTrajectory(maxTimeStep, transition, isTerminal, reset)

	numTrajs = 1
	policy = env.SheepNaiveEscapingPolicy(env.actionSpace)
	trajs = [sampleTraj(policy) for _ in range(numTrajs)]

	trajsPath = "testDataTools/trajs.pkl"
	with open(trajsPath, "wb") as f:
		pickle.dump(trajsPath, f)

	labeledTrajs = [trajActionToLabel(env.actionSpace, traj) for traj in trajs]

	decay = 1
	rewardFunction = lambda s, a: 1
	trajValueFunc = AccumulateRewards(decay, rewardFunction)
	trajsWithValues = [addValuesToTraj(traj, trajValueFunc) for traj in labeledTrajs]

	dataSet = zip(*sum(trajsWithValues, []))
	dataSetPath = "testDataTools/data.pkl"
	with open(dataSetPath, "wb") as f:
		pickle.dump(dataSet, f)


if __name__ == "__main__":
	main()
