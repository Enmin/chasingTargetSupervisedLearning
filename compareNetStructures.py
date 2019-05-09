import numpy as np
import itertools as it
import random
import pickle
import continuousEnv as env
import policyValueNet as net
import data
import visualize as VI


if __name__ == "__main__":
	random.seed(128)
	np.random.seed(128)

	numStateSpace = env.numStateSpace
	numActionSpace = env.numActionSpace
	dataSetPath = "19920Steps_2000PartialTrajs_continuousEnv_reward.pkl"
	dataSet = data.loadData(dataSetPath)
	random.shuffle(dataSet)

	trainingDataSize = 1000
	trainingData = [list(varData) for varData in zip(*dataSet[:trainingDataSize])]
	learningRate = 0.0001
	regularizationFactor = 0
	generatePolicyNet = net.GenerateModel(numStateSpace, numActionSpace, learningRate, regularizationFactor)
	neuronNums = [96, 192, 384]
	policyNetDepth = [3, 4, 5]
	# nnList = [(round(n/d), d) for n, d in it.product(neuronNums, policyNetDepth)]
	# print(nnList); exit()
	models = {(n, d): generatePolicyNet([round(n/d)]*d) for n, d in it.product(neuronNums, policyNetDepth)}

	maxStepNum = 100000
	reportInterval = 500
	lossChangeThreshold = 1e-6
	lossHistorySize = 10
	train = net.Train(maxStepNum, learningRate, lossChangeThreshold, lossHistorySize, reportInterval,
	                  summaryOn=False, testData=None)

	trainedModels = {key: train(model, trainingData) for key, model in models.items()}

	evalTrain = {key: net.evaluate(model, trainingData) for key, model in trainedModels.items()}

	evaluateDataPath = 'dataForNetStructureComparison.pkl'
	file = open(evaluateDataPath, "wb")
	pickle.dump(dataSet, file)
	file.close()
	VI.draw(evalTrain, ["neurons number", "depth"], ["Loss", "Accuracy", "valueLoss"])