import numpy as np
import pickle
import random


def generateData(sampleTrajectory, policy, trajNumber, path, actionSpace):
	totalStateBatch = []
	totalActionBatch = []
	for index in range(trajNumber):
		states, actions = sampleTrajectory(policy)
		totalStateBatch.append(np.array(states))
		oneHotAction = [[1 if (np.array(action) == np.array(actionSpace[index])).all() else 0 for index in range(len(actionSpace))] for action in actions]
		totalActionBatch.append(oneHotAction)
	dataSet = list(zip(totalStateBatch, totalActionBatch))
	saveFile = open(path, "wb")
	pickle.dump(dataSet, saveFile)
	return


def loadData(path):
	pklFile = open(path, "rb")
	dataSet = pickle.load(pklFile)
	pklFile.close()
	return dataSet


def sampleData(data, batchSize):
	batch = random.sample(data, batchSize)
	batchInput = [x for x, _ in batch]
	batchOutput = [y for _, y in batch]
	print(batchInput)
	print(batchOutput)
	return batchInput, batchOutput
