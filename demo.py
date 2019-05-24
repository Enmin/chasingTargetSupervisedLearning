import numpy as np
import pygame as pg
import policyValueNet as net
import data
import sheepEscapingEnv as env
import sheepEscapingEnvRender as envRender
import evaluateSheepEscapingPolicy as eval


def nnDemo(seed=128):
	np.random.seed(seed)

	xBoundary = env.xBoundary
	yBoundary = env.yBoundary
	actionSpace = env.actionSpace

	wolfHeatSeekingPolicy = env.WolfHeatSeekingPolicy(actionSpace)
	transition = env.TransitionFunction(xBoundary, yBoundary, env.vel, wolfHeatSeekingPolicy)
	isTerminal = env.IsTerminal(minDistance=env.vel+5)
	reset = env.Reset(xBoundary, yBoundary)
	# reset = env.ResetWithinDataSet(xBoundary, yBoundary, dataSet)

	generateModel = net.GenerateModelSeparateLastLayer(env.numStateSpace, env.numActionSpace, learningRate=0, regularizationFactor=0, valueRelativeErrBound=0.0)
	model = generateModel([64, 64, 64, 128])
	modelPath = "./savedModels/model.ckpt"
	trainedModel = net.restoreVariables(model, modelPath)
	policy = lambda state: net.approximatePolicy(state, trainedModel, actionSpace)

	maxTrajLen = 100
	trajNum = 5
	sampleTraj = data.SampleTrajectory(maxTrajLen, transition, isTerminal, reset)
	evaluate = eval.Evaluate(sampleTraj, trajNum)
	evalResults, demoStates = evaluate(policy)

	# valueEpisode = [np.array(net.approximateValueFunction(states, trainedModel)) for states in demoStates]

	extendedBound = 0
	screen = pg.display.set_mode([xBoundary[1] + extendedBound, yBoundary[1] + extendedBound])
	savePath = None
	render = envRender.Render(screen, savePath)

	renderOn = True
	if renderOn:
		for sublist in range(1, len(demoStates)):
			for index in range(len(demoStates[sublist])):
				render(demoStates[sublist][index])

	print(evalResults)


def mctsDemo(seed=128):
	np.random.seed(seed)

	xBoundary = env.xBoundary
	yBoundary = env.yBoundary
	actionSpace = env.actionSpace

	wolfHeatSeekingPolicy = env.WolfHeatSeekingPolicy(actionSpace)
	transition = env.TransitionFunction(xBoundary, yBoundary, env.vel, wolfHeatSeekingPolicy)
	isTerminal = env.IsTerminal(minDistance=env.vel+5)
	reset = env.Reset(xBoundary, yBoundary)
	# reset = env.ResetWithinDataSet(xBoundary, yBoundary, dataSet)

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
	trajNum = 1
	sampleTraj = data.SampleTrajectoryWithMCTS(maxTrajLen, isTerminal, reset, render=None)
	evaluate = eval.Evaluate(sampleTraj, trajNum)

	evalResults, demoStates = evaluate(mcts)

	extendedBound = 0
	screen = pg.display.set_mode([xBoundary[1] + extendedBound, yBoundary[1] + extendedBound])
	savePath = None
	render = envRender.Render(screen, savePath)

	renderOn = True
	if renderOn:
		for sublist in range(len(demoStates)):
			for index in range(len(demoStates[sublist])):
				render(demoStates[sublist][index])

	print(evalResults)


if __name__ == "__main__":
	# nnDemo()
	mctsDemo()
