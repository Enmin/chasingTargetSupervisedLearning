import numpy as np
import random
import pickle
import policyValueNet as net
import data
import sheepEscapingEnv as env
import visualize as VI


def main(seed=128, tfseed=128):
	random.seed(seed)
	np.random.seed(seed)

	dataSetPath = "72640steps_1000trajs_sheepEscapingEnv_data_actionDist.pkl"
	dataSet = data.loadData(dataSetPath)
	random.shuffle(dataSet)

	trainingDataSizes = [5000, 15000]  # [5000, 15000, 30000, 45000, 60000]
	trainingDataList = [[list(varData) for varData in zip(*dataSet[:size])] for size in trainingDataSizes]

	testDataSize = 12640
	testData = [list(varData) for varData in zip(*dataSet[-testDataSize:])]

	numStateSpace = env.numStateSpace
	numActionSpace = env.numActionSpace
	learningRate = 1e-4
	regularizationFactor = 0
	valueRelativeErrBound = 0.1
	generateModel = net.GenerateModelSeparateLastLayer(numStateSpace, numActionSpace, learningRate, regularizationFactor, valueRelativeErrBound=valueRelativeErrBound, seed=tfseed)
	models = [generateModel([64, 64, 64, 64]) for _ in range(len(trainingDataSizes))]

	# net.restoreVariables(models[0], "./savedModels/64*4_70000steps_minibatch_contState_actionDist")

	maxStepNum = 200000
	batchSize = 4096
	reportInterval = 1000
	lossChangeThreshold = 1e-8
	lossHistorySize = 10
	train = net.Train(maxStepNum, batchSize, lossChangeThreshold, lossHistorySize, reportInterval,
	                  summaryOn=True, testData=testData)

	trainedModels = [train(model, data) for model, data in zip(models, trainingDataList)]

	evalTrain = {("Train", size): list(net.evaluate(model, trainingData).values()) for size, trainingData, model in
	             zip(trainingDataSizes, trainingDataList, trainedModels)}
	evalTest = {("Test", size): list(net.evaluate(model, testData).values()) for size, trainingData, model in
	            zip(trainingDataSizes, trainingDataList, trainedModels)}
	evalTrain.update(evalTest)

	print(evalTrain)
	saveFile = open("diffDataSizesModels/{}evalResults.pkl".format(trainingDataSizes), "wb")
	pickle.dump(evalTrain, saveFile)

	# VI.draw(evalTrain, ["mode", "training_set_size"], ["actionLoss", "actionAcc", "valueLoss", "valueAcc"])

	for size, model in zip(trainingDataSizes, trainedModels):
		net.saveVariables(model, "diffDataSizesModels/{}data_64x4_minibatch_{}kIter_contState_actionDist".format(size, int(maxStepNum/1000)))


if __name__ == "__main__":
	main()
