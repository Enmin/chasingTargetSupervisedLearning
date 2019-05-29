import numpy as np
import pygame as pg
# import stochasticPolicyValueNet as net
import policyValueNet as net
import dataTools
import sheepEscapingEnv as env
import sheepEscapingEnvRender as envRender
import evaluateSheepEscapingPolicy as eval
import os
import subprocess


def makeVideo(videoName, path):
	absolutePath = os.path.join(os.getcwd(), path)
	os.chdir(absolutePath)
	fps = 5
	crf = 25
	resolution = '1920x1080'
	cmd = 'ffmpeg -r {} -s {} -i %d.png -vcodec libx264 -crf {} -pix_fmt yuv420p {}'.format(fps, resolution, crf, videoName).split(" ")
	subprocess.call(cmd)
	if os.path.exists(videoName):
		[os.remove(file) if file.endswith(".png") else 0 for file in os.listdir(os.getcwd())]
	else:
		print("Demo generate Failed, needs to be done manually")


def nnDemo(modelPath, trajNum, renderOn, savePath=None, seed=128):
	xBoundary = env.xBoundary
	yBoundary = env.yBoundary
	actionSpace = env.actionSpace

	wolfHeatSeekingPolicy = env.WolfHeatSeekingPolicy(actionSpace)
	transition = env.TransitionFunction(xBoundary, yBoundary, env.vel, wolfHeatSeekingPolicy)
	isTerminal = env.IsTerminal(minDistance=env.vel+5)
	reset = env.Reset(xBoundary, yBoundary, initialSeed=seed)

	generateModel = net.GenerateModelSeparateLastLayer(env.numStateSpace, env.numActionSpace, learningRate=0, regularizationFactor=0, valueRelativeErrBound=0.0)
	model = generateModel([64, 64, 64, 64])
	trainedModel = net.restoreVariables(model, modelPath)
	policy = lambda state: net.approximatePolicy(state, trainedModel, actionSpace)

	maxTrajLen = 100
	sampleTraj = dataTools.SampleTrajectory(maxTrajLen, transition, isTerminal, reset)
	evaluate = eval.Evaluate(sampleTraj, trajNum)
	evalResults, demoStates = evaluate(policy)

	valueEpisode = [np.array(net.approximateValueFunction(states, trainedModel)) for states in demoStates]

	if renderOn:
		extendedBound = 10
		screen = pg.display.set_mode([xBoundary[1], yBoundary[1] + extendedBound])
		render = envRender.Render(screen, savePath, 1)
		for sublist in range(len(demoStates)):
			for index in range(len(demoStates[sublist])):
				render(demoStates[sublist][index], int(round(valueEpisode[sublist][index][0])), sublist)

	print(evalResults)
	videoName = "mean_{}_{}_trajectories_nn_demo.mp4".format(evalResults['mean'], trajNum)
	makeVideo(videoName, savePath)
	return evalResults


def mctsDemo(trajNum, renderOn, savePath=None, seed=128):
	xBoundary = env.xBoundary
	yBoundary = env.yBoundary
	actionSpace = env.actionSpace

	wolfHeatSeekingPolicy = env.WolfHeatSeekingPolicy(actionSpace)
	transition = env.TransitionFunction(xBoundary, yBoundary, env.vel, wolfHeatSeekingPolicy)
	isTerminal = env.IsTerminal(minDistance=env.vel+5)
	reset = env.Reset(xBoundary, yBoundary, initialSeed=seed)
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
	sampleTraj = dataTools.SampleTrajectoryWithMCTS(maxTrajLen, isTerminal, reset, render=None)
	evaluate = eval.Evaluate(sampleTraj, trajNum)
	evalResults, demoStates = evaluate(mcts)

	if renderOn:
		extendedBound = 0
		screen = pg.display.set_mode([xBoundary[1] + extendedBound, yBoundary[1] + extendedBound])
		render = envRender.Render(screen, savePath, 1)
		for sublist in range(len(demoStates)):
			for index in range(len(demoStates[sublist])):
				render(demoStates[sublist][index], )

	print(evalResults)
	videoName = "mean_{}_{}_trajectories_mcts_demo.mp4".format(evalResults['mean'], trajNum)
	makeVideo(videoName, savePath)
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
	# mctsDemo(20, renderOn=False, savePath='./sheepDemo')
	# nnDemo(modelPath="savedModels/100k_iter_60000data_64x4_minibatch_150kIter_contState_actionDist", trajNum=20, renderOn=False, savePath='./sheepDemo')

