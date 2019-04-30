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

	trajNum = 20000
	dataSetPath = "data.pkl"
	# PD.generateData(sampleTrajectory, optimalPolicy, trajNum, dataSetPath, actionSpace)
	dataSet = PD.loadData(dataSetPath)

	dataNum = 94214

	learningRate = 0.01
	generatePolicyNet = NN.GeneratePolicyNet(4, 8, learningRate)
	model = generatePolicyNet(3, 32)

	maxTrainSize = int(dataNum / 2)
	maxTestSize = int(dataNum / 2)
	maxEpisode = 10000
	summaryPeriod = 500
	lossChangeThresHold = 1e-6
	trainStateBatch, trainActionBatch = PD.sampleData(dataSet, maxTrainSize)
	testStateBatch, testActionBatch = PD.sampleData(dataSet, maxTestSize)
	learn = SL.Learn(maxEpisode, learningRate, lossChangeThresHold, (trainStateBatch, trainActionBatch), (testStateBatch, testActionBatch), False, summaryPeriod)
	newModel, trainLoss, trainAccuracy, testLoss, testAccuracy = learn(model)
	print("trainLoss: {} trainAccuracy: {}\n testLoss: {} testAccuracy: {}".format(trainLoss, trainAccuracy, testLoss, testAccuracy))