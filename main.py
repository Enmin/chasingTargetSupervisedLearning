import numpy as np
import itertools as it
import tensorflow as tf
import supervisedLearning as SL
import neuralNetwork as NN
import gridEnv as GE
import prepareData as PD

if __name__ == '__main__':
	np.random.seed(128)
	tf.set_random_seed(128)

	gridSize = 10
	transitionFunction = GE.TransitionFunction(gridSize)
	isTerminal = GE.IsTerminal()

	actionSpace = [[0,1], [1,0], [-1,0], [0,-1], [1,1], [-1,-1], [1,-1], [-1,1]]
	agentStates = [state for state in it.product([x for x in range(10)], [y for y in range(10)])]
	targetStates = [state for state in it.product([x for x in range(10)], [y for y in range(10)])]
	stateSpace = [state for state in it.product(agentStates, targetStates)]
	print('Generating Optimal Policy...')
	optimalPolicy = PD.generateOptimalPolicy(stateSpace, actionSpace)
	print('Optimal Policy Generated.')

	maxTimeStep = int(gridSize * gridSize / 2)
	sampleTrajectory = PD.SampleTrajectory(maxTimeStep, transitionFunction, isTerminal, actionSpace, agentStates, targetStates)

	trajNum = 5000
	dataSetPath = "data.pkl"
	#PD.generateData(sampleTrajectory, optimalPolicy, trajNum, dataSetPath, actionSpace)
	dataSet = PD.loadData(dataSetPath)

	learningRate = 0.001
	generatePolicyNet = NN.GeneratePolicyNet(4, 8, learningRate)
	model = generatePolicyNet(3, 32)

	trainDataSizes = [1000, 2000, 5000, 10000, 15000]
	trainingDataList = [PD.sampleData(dataSet, size) for size in trainDataSizes]
	testDataSize = 5000
	testData = PD.sampleData(dataSet, testDataSize)

	maxEpisode = 5000
	summaryPeriod = 500
	lossChangeThreshold = 1e-6
	for trainingData in trainingDataList:
		print("----training data size = {}----".format(len(trainingData[0])))
		learn = SL.Learn(maxEpisode, learningRate, lossChangeThreshold, trainingData, testData, False, summaryPeriod)
		newModel, trainLoss, trainAccuracy, testLoss, testAccuracy = learn(model)
		print("trainLoss: {} trainAccuracy: {}\n testLoss: {} testAccuracy: {}".format(trainLoss, trainAccuracy, testLoss, testAccuracy))