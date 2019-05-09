import numpy as np
import random
import policyValueNet as net
import data
import visualize as VI
import continuousEnv as env


if __name__ == "__main__":
	random.seed(128)
	np.random.seed(128)

	numStateSpace = env.numStateSpace
	numActionSpace = env.numActionSpace
	# dataSetPath = "199731Steps_2000Trajs_continuousEnv_reward.pkl"
	dataSetPath = "19920Steps_2000PartialTrajs_continuousEnv_reward.pkl"
	dataSet = data.loadData(dataSetPath)
	random.shuffle(dataSet)

	# trainingDataSizes = list(range(5000, 6001, 1000))
	trainingDataSizes = [1000]
	trainingDataList = [[list(varData) for varData in zip(*dataSet[:size])] for size in trainingDataSizes]
	testDataSize = 10000
	testData = data.sampleData(dataSet, testDataSize)

	learningRate = 0.0001
	regularizationFactor = 0  # 1e-4
	generateModel = net.GenerateModel(numStateSpace, numActionSpace, learningRate, regularizationFactor)
	models = [generateModel([32]*3) for i in range(len(trainingDataSizes))]

	# for model in models: print(net.evaluate(model, testData))
	# exit()

	maxStepNum = 1000000
	reportInterval = 500
	lossChangeThreshold = 1e-6
	lossHistorySize = 10
	train = net.Train(maxStepNum, learningRate, lossChangeThreshold, lossHistorySize, reportInterval,
	                  summaryOn=True, testData=testData)

	trainedModels = [train(model, data) for model, data in zip(models, trainingDataList)]

	evalTrain = {("Train", size): net.evaluate(model, trainingData) for size, trainingData, model in
	             zip(trainingDataSizes, trainingDataList, trainedModels)}
	evalTest = {("Test", size): net.evaluate(model, testData) for size, trainingData, model in
	            zip(trainingDataSizes, trainingDataList, trainedModels)}
	evalTrain.update(evalTest)

	print(evalTrain)
	VI.draw(evalTrain, ["mode", "training_set_size"], ["Loss", "Accuracy", "valueLoss"])
