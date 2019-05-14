import numpy as np
import random
import pickle
import policyValueNet as net
import data as dt
import continuousEnv as env
import pygame as pg


def main(seed=128):
	random.seed(seed)
	np.random.seed(seed)

	dataSetPath = "19976Steps_4000PartialTrajsHead_continuousEnv_reward.pkl"
	dataSet = dt.loadData(dataSetPath)
	random.shuffle(dataSet)

	trainingDataSizes = [1000]  # list(range(3000, 9001, 1000))
	trainingDataList = [[list(varData) for varData in zip(*dataSet[:size])] for size in trainingDataSizes]

	testDataSize = 7000
	testData = [list(varData) for varData in zip(*dataSet[-testDataSize:])]

	numStateSpace = env.numStateSpace
	numActionSpace = env.numActionSpace
	learningRate = 1e-4
	regularizationFactor = 0  # 1e-4
	valueRelativeErrBound = 0.01
	generateModel = net.GenerateModelSeparateLastLayer(numStateSpace, numActionSpace, learningRate, regularizationFactor, valueRelativeErrBound=valueRelativeErrBound)
	models = [generateModel([32]*3) for _ in range(len(trainingDataSizes))]

	maxStepNum = 3000
	reportInterval = 500
	lossChangeThreshold = 1e-6
	lossHistorySize = 10
	train = net.Train(maxStepNum, learningRate, lossChangeThreshold, lossHistorySize, reportInterval,
	                  summaryOn=False, testData=testData)

	# trainedModels = [train(model, data) for model, data in zip(models, trainingDataList)]
	# net.saveVariables(trainedModels[0], "./savedModels/model")

	trainedModel = net.restoreVariables(models[0], "./savedModels/model.ckpt")
	evalTrain = {("Train", size): list(net.evaluate(model, trainingData).values()) for size, trainingData, model in
	             zip(trainingDataSizes, trainingDataList, [trainedModel])}

	print(evalTrain)

	# demo
	actionSpace = [[0, 1], [1, 0], [-1, 0], [0, -1], [1, 1], [-1, -1], [1, -1], [-1, 1]]
	xBoundary = [0, 180]
	yBoundary = [0, 180]
	extendedBound = 30
	vel = 1
	maxTraj = 10
	transition = env.TransitionFunction(xBoundary, yBoundary, vel)
	isTerminal = env.IsTerminal(minDistance=vel+0.5)
	reset = env.Reset(xBoundary, yBoundary)
	sampleTraj = dt.SampleTrajectory(1000, transition, isTerminal, reset)
	policy = lambda state: net.approximatePolicy(state, trainedModel, actionSpace)
	demoEpisode = [zip(*sampleTraj(policy)) for index in range(maxTraj)]
	demoStates = [states for states, actions in demoEpisode]
	valueEpisode = [net.approximateValueFunction(states, trainedModel) for states in demoStates]
	screen = pg.display.set_mode([xBoundary[1]+extendedBound, yBoundary[1] + extendedBound])
	screenColor = [255, 255, 255]
	circleColorList = [[50, 255, 50], [50, 50, 50], [50, 50, 50], [50, 50, 50], [50, 50, 50], [50, 50, 50],
	                   [50, 50, 50], [50, 50, 50], [50, 50, 50]]
	circleSize = 8
	saveImage = True
	numOneAgentState = 2
	positionIndex = [0, 1]
	numAgent = 2
	savePath = './ValueDemo'
	render = env.Render(numAgent, numOneAgentState, positionIndex, screen, screenColor, circleColorList, circleSize,
	                    saveImage, savePath)
	for sublist in range(len(demoStates)):
		for index in range(len(demoStates[sublist])):
			render(demoStates[sublist][index], valueEpisode[sublist][index])


if __name__ == "__main__":
	main(129)
