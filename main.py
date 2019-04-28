import numpy as np
import itertools as it
import supervisedLearning as SL
import neuralNetwork as NN
import gridEnv as GE

if __name__ == '__main__':
	gridSize = 10
	actionSpace = [[0,1], [1,0], [-1,0], [0,-1], [1,1], [-1,-1], [1,-1], [-1,1]]
	agentStates = [state for state in it.product([x for x in range(10)], [y for y in range(10)])]
	targetStates = [state for state in it.product([x for x in range(10)], [y for y in range(10)])]
	stateSpace = [state for state in it.product(agentStates, targetStates)]
	numOfaction = len(actionSpace)
	numOfState = len(stateSpace)
	maxTimeStep = int(gridSize * gridSize / 2)
	learningRate = 0.01
	dataNum = 2000
	maxEpisode = 300
	totalStateBatch = []
	totalActionBatch = []
	print('Generating Optimal Policy...')
	optimalPolicy = SL.generateOptimalPolicy(stateSpace, actionSpace)
	print('Optimal Policy Generated.')

	transitionFunction = GE.TransitionFunction(gridSize)
	generatePolicyNet = NN.GeneratePolicyNet(numOfState, numOfaction, learningRate, optimalPolicy, actionSpace)
	model = generatePolicyNet(2, 128)
	graph = model.graph
	isTerminal = GE.IsTerminal()
	sampleTrajectory = SL.SampleTrajectory(maxTimeStep, transitionFunction, isTerminal, actionSpace, optimalPolicy)
	batch = [SL.sampleTotalTrajectory(sampleTrajectory, agentStates, targetStates) for data in range(dataNum)]
	for stateBatch, actionBatch in batch:
		[totalStateBatch.append(np.array(state)) for state in stateBatch]
		[totalActionBatch.append(np.array(action)) for action in actionBatch]
	print(len(totalStateBatch))
	print(len(totalActionBatch))
	totalActionBatchLabel = [[1 if (action == actionSpace[index]).all() else 0 for index in range(len(actionSpace))] for action in totalActionBatch]

	for episode in range(maxEpisode):
		state_ = graph.get_tensor_by_name('inputs/state_:0')
		actionLabel_ = graph.get_tensor_by_name('inputs/actionLabel_:0')
		loss_ = graph.get_tensor_by_name('outputs/loss_:0')
		trainOpt_ = graph.get_operation_by_name('train/adamOpt_')
		loss, trainOpt = model.run([loss_, trainOpt_], feed_dict={state_: totalStateBatch, actionLabel_: totalActionBatchLabel})
		print(loss)

	print("Testing Model...")
	totalPlayStates = 0
	totalOptimalStates = 0
	testTimes = 100
	optimalStates = None
	playStates = None
	optimal = None
	playEpisode = None
	for index in range(0, testTimes):
		initialAgentState = agentStates[np.random.randint(0, len(agentStates))]
		targetPosition = targetStates[np.random.randint(0, len(targetStates))]
		# initialAgentState = (6,4)
		# targetPosition = (1,2)
		while isTerminal(initialAgentState, targetPosition):
			initialAgentState = agentStates[np.random.randint(0, len(agentStates))]
			targetPosition = targetStates[np.random.randint(0, len(targetStates))]
		approximatePolicy = SL.ApproximatePolicy(actionSpace)
		samplePlay = SL.SampleTrajectory(maxTimeStep, transitionFunction, isTerminal, actionSpace, optimalPolicy)
		policy = lambda state: approximatePolicy(state, model)
		playEpisode = samplePlay(policy, initialAgentState, targetPosition)
		optimal = samplePlay(None, initialAgentState, targetPosition)
		optimalStates, optimalActions = optimal
		totalOptimalStates += len(optimalStates)
		playStates, playActions = playEpisode
		totalPlayStates += len(playStates)

	print("Play: {}, Optimal:{}".format(totalPlayStates / testTimes, totalOptimalStates / testTimes))
