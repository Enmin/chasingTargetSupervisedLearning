import numpy as np
import random
import pickle
import policyValueNet as net
import data
import visualize as VI
import sheepEscapingEnv as env


def main(seed=128, tfseed=128):
	random.seed(seed)
	np.random.seed(seed)

	dataSetPath = "34793steps_500trajs_sheepEscapingEnv_data.pkl"
	# dataSetPath = "35087steps_500trajs_sheepEscapingEnv_data_actionDist.pkl"
	dataSet = data.loadData(dataSetPath)
	random.shuffle(dataSet)

	trainingDataSizes = [5000]
	trainingDataList = [[list(varData) for varData in zip(*dataSet[:size])] for size in trainingDataSizes]

	testDataSize = 5000
	testData = [list(varData) for varData in zip(*dataSet[-testDataSize:])]

	numStateSpace = env.numStateSpace
	numActionSpace = env.numActionSpace
	learningRate = 1e-4
	regularizationFactor = 0  # 1e-4
	valueRelativeErrBound = 0.05
	generateModel = net.GenerateModelSeparateLastLayer(numStateSpace, numActionSpace, learningRate, regularizationFactor, valueRelativeErrBound=valueRelativeErrBound, seed=tfseed)
	models = [generateModel([64, 64, 64, 64]) for _ in range(len(trainingDataSizes))]

	maxStepNum = 50000
	reportInterval = 500
	lossChangeThreshold = 1e-8
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
	# saveFile = open("./11000-12000evalResults.pkl", "wb")
	# pickle.dump(evalTrain, saveFile)

	# VI.draw(evalTrain, ["mode", "training_set_size"], ["actionLoss", "actionAcc", "valueLoss", "valueAcc"])

	net.saveVariables(trainedModels[0], "./savedModels/model.ckpt")


if __name__ == "__main__":
	main()
