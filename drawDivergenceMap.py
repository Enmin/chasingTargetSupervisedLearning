import policyValueNet as net
import sheepEscapingEnv as env
import numpy as np
import pandas as pd
from anytree import AnyNode as Node
from AnalyticGeometryFunctions import calculateCrossEntropy
from pylab import plt


def getNNModel(modelPath):
	generateModel = net.GenerateModelSeparateLastLayer(env.numStateSpace, env.numActionSpace, learningRate=0,
	                                                   regularizationFactor=0, valueRelativeErrBound=0.0)
	model = generateModel([64, 64, 64, 64])
	trainedModel = net.restoreVariables(model, modelPath)
	return trainedModel


def getMCTSModel():
	xBoundary = env.xBoundary
	yBoundary = env.yBoundary
	actionSpace = env.actionSpace

	wolfHeatSeekingPolicy = env.WolfHeatSeekingPolicy(actionSpace)
	transition = env.TransitionFunction(xBoundary, yBoundary, env.vel, wolfHeatSeekingPolicy)
	isTerminal = env.IsTerminal(minDistance=env.vel + 5)
	# reset = env.Reset(xBoundary, yBoundary, initialSeed=seed)

	rewardFunction = lambda state, action: 1

	from mcts import MCTS, CalculateScore, GetActionPrior, selectNextRoot, SelectChild, Expand, RollOut, backup, \
		InitializeChildren
	cInit = 1
	cBase = 1
	calculateScore = CalculateScore(cInit, cBase)
	selectChild = SelectChild(calculateScore)

	getActionPrior = GetActionPrior(actionSpace)
	initializeChildren = InitializeChildren(actionSpace, transition, getActionPrior)
	expand = Expand(transition, isTerminal, initializeChildren)

	maxRollOutSteps = 10
	numSimulations = 100
	rolloutPolicy = lambda state: actionSpace[np.random.choice(range(env.numActionSpace))]
	rollout = RollOut(rolloutPolicy, maxRollOutSteps, transition, rewardFunction, isTerminal)
	mcts = MCTS(numSimulations, selectChild, expand, rollout, backup, selectNextRoot)
	return mcts


def getNodeFromState(state):
	rootNode = Node(id={None: state}, num_visited=0, sum_value=0, is_expanded=False)
	return rootNode


def whetherStateInDfIndex(dfIndex, state, discreteRange):
	for index in range(len(state)):
		if state[index] < (dfIndex[index] - discreteRange) or state[index] >= dfIndex[index]:
			return False
	return True


def evaluateModel(df, mctsModel, nnModel, sampleStates, discreteRange):
	validStates = [state for state in sampleStates if whetherStateInDfIndex(df.index[0], state, discreteRange)]
	if len(validStates) == 0:
		return pd.Series({"cross_entropy": None})
	validNodes = [getNodeFromState(state) for state in validStates]
	actionDistribution = [mctsModel(node) for node in validNodes]
	nnActionDistribution = [net.approximateActionPrior(state, nnModel, env.actionSpace) for state in validStates]
	crossEntropyList = [calculateCrossEntropy(np.array(list(prediction.values())), np.array(list(target.values()))) for prediction, target in zip(nnActionDistribution, actionDistribution)]
	meanCrossEntropy = np.mean(crossEntropyList)
	# print(actionDistribution)
	# print(nnActionDistribution)
	return pd.Series({"cross_entropy": meanCrossEntropy})


def drawHeatMap(dataDF, groupByVariableNames, subplotIndex, subplotIndexName, valueIndexName):
	figure = plt.figure(figsize=(12, 10))
	numOfplot = 0
	for key, subDF in dataDF.groupby(groupByVariableNames):
		numOfplot += 1
		subplot = figure.add_subplot(subplotIndex[0], subplotIndex[1], numOfplot)
		plotDF = subDF.reset_index()
		plotDF.plot.scatter(x=subplotIndexName[0], y=subplotIndexName[1], c=valueIndexName, colormap="viridis", ax=subplot)
	plt.subplots_adjust(wspace=0.7, hspace=0.4)
	plt.savefig("./Graphs/actionDistribution_divergence_HeatMap.png")


def sampleState(reset, terminal, num):
	sampleStates = []
	for index in range(num):
		state = reset()
		while terminal(state):
			state = reset()
		sampleStates.append(state)
	return sampleStates

def main(seed=None):
	discreteFactor = 3
	discreteRange = env.xBoundary[1] / 3
	numOfPoints = 50
	wolfXPosition = [env.xBoundary[1]/discreteFactor * (i+1) for i in range(discreteFactor)]
	wolfYPosition = [env.yBoundary[1]/discreteFactor * (i+1) for i in range(discreteFactor)]
	sheepXPosition = wolfXPosition
	sheepYPosition = wolfYPosition
	levelValues = [wolfXPosition, wolfYPosition, sheepXPosition, sheepYPosition]
	levelNames = ["wolfXPosition", "wolfYPosition", "sheepXPosition", "sheepYPosition"]
	diffIndex = pd.MultiIndex.from_product(levelValues, names=levelNames)
	toSplitFrame = pd.DataFrame(index=diffIndex)

	modelPath = "savedModels/60000data_64x4_minibatch_100kIter_contState_actionDist"
	nnModel = getNNModel(modelPath)
	mctsModel = getMCTSModel()
	reset = env.Reset(env.xBoundary, env.yBoundary, initialSeed=seed)
	isTerminal = env.IsTerminal(minDistance=25)
	sampleStates = sampleState(reset, isTerminal, numOfPoints)
	resultDF = toSplitFrame.groupby(levelNames).apply(evaluateModel, mctsModel, nnModel, sampleStates, discreteRange)
	drawHeatMap(resultDF, ['wolfXPosition', 'wolfYPosition'], [len(wolfXPosition), len(wolfYPosition)], ['sheepXPosition', 'sheepYPosition'], 'cross_entropy')


if __name__ == "__main__":
	np.random.seed(5)
	main()