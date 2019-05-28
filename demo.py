import numpy as np
import pygame as pg
# import stochasticPolicyValueNet as net
import policyValueNet as net
import data
import sheepEscapingEnv as env
import sheepEscapingEnvRender as envRender
import evaluateSheepEscapingPolicy as eval


def nnDemo(modelPath, trajNum, renderOn, savePath=None, seed=128):
	np.random.seed(seed)

	xBoundary = env.xBoundary
	yBoundary = env.yBoundary
	actionSpace = env.actionSpace

	wolfHeatSeekingPolicy = env.WolfHeatSeekingPolicy(actionSpace)
	transition = env.TransitionFunction(xBoundary, yBoundary, env.vel, wolfHeatSeekingPolicy)
	isTerminal = env.IsTerminal(minDistance=env.vel+5)
	reset = env.Reset(xBoundary, yBoundary)

	generateModel = net.GenerateModelSeparateLastLayer(env.numStateSpace, env.numActionSpace, learningRate=0, regularizationFactor=0, valueRelativeErrBound=0.0)
	model = generateModel([64, 64, 64, 64])
	trainedModel = net.restoreVariables(model, modelPath)
	policy = lambda state: net.approximatePolicy(state, trainedModel, actionSpace)

	maxTrajLen = 100
	sampleTraj = data.SampleTrajectory(maxTrajLen, transition, isTerminal, reset)
	evaluate = eval.Evaluate(sampleTraj, trajNum)
	evalResults, demoStates = evaluate(policy)

	# valueEpisode = [np.array(net.approximateValueFunction(states, trainedModel)) for states in demoStates]

	if renderOn:
		extendedBound = 0
		screen = pg.display.set_mode([xBoundary[1] + extendedBound, yBoundary[1] + extendedBound])
		render = envRender.Render(screen, savePath)
		for sublist in range(len(demoStates)):
			for index in range(len(demoStates[sublist])):
				render(demoStates[sublist][index])

	print(evalResults)
	return evalResults


def mctsDemo(trajNum, renderOn, savePath=None, seed=128):
	np.random.seed(seed)

	xBoundary = env.xBoundary
	yBoundary = env.yBoundary
	actionSpace = env.actionSpace

	wolfHeatSeekingPolicy = env.WolfHeatSeekingPolicy(actionSpace)
	transition = env.TransitionFunction(xBoundary, yBoundary, env.vel, wolfHeatSeekingPolicy)
	isTerminal = env.IsTerminal(minDistance=env.vel+5)
	reset = env.Reset(xBoundary, yBoundary)
	# reset = lambda: [0, 0, 40, 100]

	rewardFunction = lambda state, action: 1

	from mcts import MCTS, CalculateScore, GetActionPrior, selectNextRoot, SelectChild, Expand, RollOut, backup, InitializeChildren
	cInit = 1
	cBase = 1
	calculateScore = CalculateScore(cInit, cBase)
	selectChild = SelectChild(calculateScore)

	getActionPrior = GetActionPrior(actionSpace)
	initializeChildren = InitializeChildren(actionSpace, transition, getActionPrior)
	expand = Expand(transition, isTerminal, initializeChildren)

	maxRollOutSteps = 10
	numSimulations = 600
	rolloutPolicy = lambda state: actionSpace[np.random.choice(range(env.numActionSpace))]
	rollout = RollOut(rolloutPolicy, maxRollOutSteps, transition, rewardFunction, isTerminal)
	mcts = MCTS(numSimulations, selectChild, expand, rollout, backup, selectNextRoot)

	maxTrajLen = 100
	sampleTraj = data.SampleTrajectoryWithMCTS(maxTrajLen, isTerminal, reset, render=None)
	evaluate = eval.Evaluate(sampleTraj, trajNum)

	evalResults, demoStates = evaluate(mcts)

	if renderOn:
		extendedBound = 0
		screen = pg.display.set_mode([xBoundary[1] + extendedBound, yBoundary[1] + extendedBound])
		render = envRender.Render(screen, savePath)
		for sublist in range(len(demoStates)):
			for index in range(len(demoStates[sublist])):
				render(demoStates[sublist][index])

	print(evalResults)
	return evalResults


if __name__ == "__main__":
	import sys
	if len(sys.argv) != 2:
		print("Usage: python3 demo.py modelPath|mcts")
		exit()
	trajNum = 2
	renderOn = True
	if sys.argv[1] == "mcts":
		mctsDemo(trajNum, renderOn)
	else:
		nnDemo(sys.argv[1], trajNum, renderOn)

