import numpy as np
import random
import pickle
import policyValueNet as net
import data
import visualize as VI
import continuousEnv as env


def main(seed=128):
	random.seed(seed)
	np.random.seed(seed)

	dataSetPath = "19976Steps_4000PartialTrajsHead_continuousEnv_reward.pkl"
	dataSet = data.loadData(dataSetPath)
	random.shuffle(dataSet)

	trainingDataSizes = [2000]  # list(range(3000, 9001, 1000))
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

	# for model in models: print(net.evaluate(model, testData))
	# exit()

	maxStepNum = 100000
	reportInterval = 500
	lossChangeThreshold = 1e-6
	lossHistorySize = 10
	train = net.Train(maxStepNum, learningRate, lossChangeThreshold, lossHistorySize, reportInterval,
	                  summaryOn=True, testData=testData)

	trainedModels = [train(model, data) for model, data in zip(models, trainingDataList)]

	evalTrain = {("Train", size): list(net.evaluate(model, trainingData).values()) for size, trainingData, model in
	             zip(trainingDataSizes, trainingDataList, trainedModels)}
	evalTest = {("Test", size): list(net.evaluate(model, testData).values()) for size, trainingData, model in
	            zip(trainingDataSizes, trainingDataList, trainedModels)}
	evalTrain.update(evalTest)

	print(evalTrain)
	# VI.draw(evalTrain, ["mode", "training_set_size"], ["actionLoss", "actionAcc", "valueLoss", "valueAcc"])


if __name__ == "__main__":
	main(129)
