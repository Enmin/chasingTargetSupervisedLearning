import tensorflow as tf
import numpy as np
import data


class GenerateModel:
	def __init__(self, numStateSpace, numActionSpace, learningRate, regularizationFactor, deterministicInit=True):
		self.numStateSpace = numStateSpace
		self.numActionSpace = numActionSpace
		self.learningRate = learningRate
		self.regularizationFactor = regularizationFactor
		self.deterministicInit = deterministicInit

	def __call__(self, hiddenWidths, summaryPath="./tbdata"):
		print("Generating Policy Net with hidden layers: {}".format(hiddenWidths))
		graph = tf.Graph()
		with graph.as_default():
			if self.deterministicInit:
				tf.set_random_seed(128)

			with tf.name_scope("inputs"):
				state_ = tf.placeholder(tf.float32, [None, self.numStateSpace], name="state_")
				actionLabel_ = tf.placeholder(tf.int32, [None, self.numActionSpace], name="actionLabel_")
				valueLabel_ = tf.placeholder(tf.float32, [None, 1], name="valueLabel_")
				tf.add_to_collection("inputs", state_)
				tf.add_to_collection("inputs", actionLabel_)
				tf.add_to_collection("inputs", valueLabel_)

			with tf.name_scope("hidden"):
				initWeight = tf.random_uniform_initializer(-0.03, 0.03)
				initBias = tf.constant_initializer(0.001)

				fc1 = tf.layers.Dense(units=hiddenWidths[0], activation=tf.nn.relu, kernel_initializer=initWeight, bias_initializer=initBias)
				a1_ = fc1(state_)
				w1_, b1_ = fc1.weights
				tf.summary.histogram("w1", w1_)
				tf.summary.histogram("b1", b1_)
				tf.summary.histogram("a1", a1_)

				a_ = a1_
				for i in range(1, len(hiddenWidths)):
					fc = tf.layers.Dense(units=hiddenWidths[i], activation=tf.nn.relu, kernel_initializer=initWeight, bias_initializer=initBias)
					aNext_ = fc(a_)
					a_ = aNext_
					w_, b_ = fc.weights
					tf.summary.histogram("w{}".format(i+1), w_)
					tf.summary.histogram("b{}".format(i+1), b_)
					tf.summary.histogram("a{}".format(i+1), a_)

				fcLastAction = tf.layers.Dense(units=self.numActionSpace, activation=None, kernel_initializer=initWeight, bias_initializer=initBias)
				allActionActivation_ = fcLastAction(a_)
				wLastAction_, bLastAction_ = fcLastAction.weights
				tf.summary.histogram("wLastAction", wLastAction_)
				tf.summary.histogram("bLastAction", bLastAction_)
				tf.summary.histogram("allActionActivation", allActionActivation_)

				fcLastValue = tf.layers.Dense(units=1, activation=None, kernel_initializer=initWeight, bias_initializer=initBias)
				valueActivation_ = fcLastValue(a_)
				wLastValue_, bLastValue_ = fcLastValue.weights
				tf.summary.histogram("wLastValue", wLastValue_)
				tf.summary.histogram("bLastValue", bLastValue_)
				tf.summary.histogram("value", valueActivation_)

			with tf.name_scope("outputs"):
				actionDistribution_ = tf.nn.softmax(allActionActivation_, name='actionDistribution_')
				cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=actionDistribution_, labels=actionLabel_, name='cross_entropy')
				actionLoss_ = tf.reduce_mean(cross_entropy, name='actionLoss_')
				tf.add_to_collection("actionLoss", actionLoss_)
				actionLossSummary = tf.summary.scalar("actionLoss", actionLoss_)

				actionIndices_ = tf.argmax(actionDistribution_, axis=1)
				actionLabelIndices_ = tf.argmax(actionLabel_, axis=1)
				actionAccuracy_ = tf.reduce_mean(tf.cast(tf.equal(actionIndices_, actionLabelIndices_), tf.float32))
				tf.add_to_collection("actionIndices", actionIndices_)
				tf.add_to_collection("actionAccuracy", actionAccuracy_)
				actionAccuracySummary = tf.summary.scalar("actionAccuracy", actionAccuracy_)

				valuePrediction_ = valueActivation_
				valueLoss_ = tf.sqrt(tf.losses.mean_squared_error(valueLabel_, valuePrediction_))
				tf.add_to_collection("valuePrediction", valuePrediction_)
				tf.add_to_collection("valueLoss", valueLoss_)
				valueLossSummary = tf.summary.scalar("valueLoss", valueLoss_)

			with tf.name_scope("train"):
				l2RegularizationLoss_ = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * self.regularizationFactor
				loss_ = 50*actionLoss_ + valueLoss_ + l2RegularizationLoss_
				tf.add_to_collection("loss", loss_)
				regLossSummary = tf.summary.scalar("l2RegLoss", l2RegularizationLoss_)
				lossSummary = tf.summary.scalar("loss", loss_)

				optimizer = tf.train.AdamOptimizer(self.learningRate, name='adamOpt_')
				gradVarPairs_ = optimizer.compute_gradients(loss_)
				trainOp = optimizer.apply_gradients(gradVarPairs_)
				tf.add_to_collection(tf.GraphKeys.TRAIN_OP, trainOp)

				gradients_ = [tf.reshape(grad, [1, -1]) for (grad, _) in gradVarPairs_]
				gradTensor_ = tf.concat(gradients_, 1)
				gradNorm_ = tf.norm(gradTensor_)
				tf.add_to_collection("gradNorm", gradNorm_)
				tf.summary.histogram("gradients", gradTensor_)
				tf.summary.scalar("gradNorm", gradNorm_)

			fullSummary = tf.summary.merge_all()
			evalSummary = tf.summary.merge([lossSummary, actionLossSummary, valueLossSummary, actionAccuracySummary])
			tf.add_to_collection("summaryOps", fullSummary)
			tf.add_to_collection("summaryOps", evalSummary)

			trainWriter = tf.summary.FileWriter(summaryPath + "/train", graph=tf.get_default_graph())
			testWriter = tf.summary.FileWriter(summaryPath + "/test", graph=tf.get_default_graph())
			tf.add_to_collection("writers", trainWriter)
			tf.add_to_collection("writers", testWriter)

			model = tf.Session(graph=graph)
			model.run(tf.global_variables_initializer())

		return model


class GenerateModelSeparateLastLayer:
	def __init__(self, numStateSpace, numActionSpace, learningRate, regularizationFactor, deterministicInit=True):
		self.numStateSpace = numStateSpace
		self.numActionSpace = numActionSpace
		self.learningRate = learningRate
		self.regularizationFactor = regularizationFactor
		self.deterministicInit = deterministicInit

	def __call__(self, hiddenWidths, summaryPath="./tbdata"):
		print("Generating Policy Net with hidden layers: {}".format(hiddenWidths))
		graph = tf.Graph()
		with graph.as_default():
			if self.deterministicInit:
				tf.set_random_seed(128)

			with tf.name_scope("inputs"):
				state_ = tf.placeholder(tf.float32, [None, self.numStateSpace], name="state_")
				actionLabel_ = tf.placeholder(tf.int32, [None, self.numActionSpace], name="actionLabel_")
				valueLabel_ = tf.placeholder(tf.float32, [None, 1], name="valueLabel_")
				tf.add_to_collection("inputs", state_)
				tf.add_to_collection("inputs", actionLabel_)
				tf.add_to_collection("inputs", valueLabel_)

			with tf.name_scope("hidden"):
				initWeight = tf.random_uniform_initializer(-0.03, 0.03)
				initBias = tf.constant_initializer(0.001)

				fc1 = tf.layers.Dense(units=hiddenWidths[0], activation=tf.nn.relu, kernel_initializer=initWeight, bias_initializer=initBias)
				a1_ = fc1(state_)
				w1_, b1_ = fc1.weights
				tf.summary.histogram("w1", w1_)
				tf.summary.histogram("b1", b1_)
				tf.summary.histogram("a1", a1_)

				a_ = a1_
				for i in range(1, len(hiddenWidths)-1):
					fc = tf.layers.Dense(units=hiddenWidths[i], activation=tf.nn.relu, kernel_initializer=initWeight, bias_initializer=initBias)
					aNext_ = fc(a_)
					a_ = aNext_
					w_, b_ = fc.weights
					tf.summary.histogram("w{}".format(i+1), w_)
					tf.summary.histogram("b{}".format(i+1), b_)
					tf.summary.histogram("a{}".format(i+1), a_)

				fcAction = tf.layers.Dense(units=hiddenWidths[-1], activation=tf.nn.relu, kernel_initializer=initWeight, bias_initializer=initBias)
				aAction_ = fcAction(a_)
				fcLastAction = tf.layers.Dense(units=self.numActionSpace, activation=None, kernel_initializer=initWeight, bias_initializer=initBias)
				allActionActivation_ = fcLastAction(aAction_)
				wLastAction_, bLastAction_ = fcLastAction.weights
				tf.summary.histogram("wLastAction", wLastAction_)
				tf.summary.histogram("bLastAction", bLastAction_)
				tf.summary.histogram("allActionActivation", allActionActivation_)

				fcValue = tf.layers.Dense(units=hiddenWidths[-1], activation=tf.nn.relu, kernel_initializer=initWeight, bias_initializer=initBias)
				aValue_ = fcValue(a_)
				fcLastValue = tf.layers.Dense(units=1, activation=None, kernel_initializer=initWeight, bias_initializer=initBias)
				valueActivation_ = fcLastValue(aValue_)
				wLastValue_, bLastValue_ = fcLastValue.weights
				tf.summary.histogram("wLastValue", wLastValue_)
				tf.summary.histogram("bLastValue", bLastValue_)
				tf.summary.histogram("value", valueActivation_)

			with tf.name_scope("outputs"):
				actionDistribution_ = tf.nn.softmax(allActionActivation_, name='actionDistribution_')
				cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=actionDistribution_, labels=actionLabel_, name='cross_entropy')
				actionLoss_ = tf.reduce_mean(cross_entropy, name='actionLoss_')
				tf.add_to_collection("actionLoss", actionLoss_)
				actionLossSummary = tf.summary.scalar("actionLoss", actionLoss_)

				actionIndices_ = tf.argmax(actionDistribution_, axis=1)
				actionLabelIndices_ = tf.argmax(actionLabel_, axis=1)
				actionAccuracy_ = tf.reduce_mean(tf.cast(tf.equal(actionIndices_, actionLabelIndices_), tf.float32))
				tf.add_to_collection("actionIndices", actionIndices_)
				tf.add_to_collection("actionAccuracy", actionAccuracy_)
				actionAccuracySummary = tf.summary.scalar("actionAccuracy", actionAccuracy_)

				valuePrediction_ = valueActivation_
				valueLoss_ = tf.sqrt(tf.losses.mean_squared_error(valueLabel_, valuePrediction_))
				tf.add_to_collection("valuePrediction", valuePrediction_)
				tf.add_to_collection("valueLoss", valueLoss_)
				valueLossSummary = tf.summary.scalar("valueLoss", valueLoss_)

			with tf.name_scope("train"):
				l2RegularizationLoss_ = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * self.regularizationFactor
				loss_ = 50*actionLoss_ + valueLoss_ + l2RegularizationLoss_
				tf.add_to_collection("loss", loss_)
				regLossSummary = tf.summary.scalar("l2RegLoss", l2RegularizationLoss_)
				lossSummary = tf.summary.scalar("loss", loss_)

				optimizer = tf.train.AdamOptimizer(self.learningRate, name='adamOpt_')
				gradVarPairs_ = optimizer.compute_gradients(loss_)
				trainOp = optimizer.apply_gradients(gradVarPairs_)
				tf.add_to_collection(tf.GraphKeys.TRAIN_OP, trainOp)

				gradients_ = [tf.reshape(grad, [1, -1]) for (grad, _) in gradVarPairs_]
				gradTensor_ = tf.concat(gradients_, 1)
				gradNorm_ = tf.norm(gradTensor_)
				tf.add_to_collection("gradNorm", gradNorm_)
				tf.summary.histogram("gradients", gradTensor_)
				tf.summary.scalar("gradNorm", gradNorm_)

			fullSummary = tf.summary.merge_all()
			evalSummary = tf.summary.merge([lossSummary, actionLossSummary, valueLossSummary, actionAccuracySummary])
			tf.add_to_collection("summaryOps", fullSummary)
			tf.add_to_collection("summaryOps", evalSummary)

			trainWriter = tf.summary.FileWriter(summaryPath + "/train", graph=tf.get_default_graph())
			testWriter = tf.summary.FileWriter(summaryPath + "/test", graph=tf.get_default_graph())
			tf.add_to_collection("writers", trainWriter)
			tf.add_to_collection("writers", testWriter)

			model = tf.Session(graph=graph)
			model.run(tf.global_variables_initializer())

		return model


class Train:
	def __init__(self,
                 maxStepNum, learningRate, lossChangeThreshold, lossHistorySize,
                 reportInterval,
                 summaryOn=False, testData=None):
		self.maxStepNum = maxStepNum
		self.learningRate = learningRate
		self.lossChangeThreshold = lossChangeThreshold
		self.lossHistorySize = lossHistorySize

		self.reportInterval = reportInterval

		self.summaryOn = summaryOn
		self.testData = testData

	def __call__(self, model, trainingData):
		graph = model.graph
		state_, actionLabel_, valueLabel_ = graph.get_collection_ref("inputs")
		loss_ = graph.get_collection_ref("loss")[0]
		accuracy_ = graph.get_collection_ref("actionAccuracy")[0]
		trainOp = graph.get_collection_ref(tf.GraphKeys.TRAIN_OP)[0]
		fullSummaryOp = graph.get_collection_ref('summaryOps')[0]
		trainWriter = graph.get_collection_ref('writers')[0]

		stateBatch, actionLabelBatch, valueLabelBatch = trainingData

		lossHistory = np.ones(self.lossHistorySize)
		terminalCond = False

		for stepNum in range(self.maxStepNum):
			if self.summaryOn and (stepNum % self.reportInterval == 0 or stepNum == self.maxStepNum-1 or terminalCond):
				loss, accuracy, _, fullSummary = model.run([loss_, accuracy_, trainOp, fullSummaryOp],
                                                        	feed_dict={state_: stateBatch,
                                                                      actionLabel_: actionLabelBatch,
																	  valueLabel_: valueLabelBatch})
				trainWriter.add_summary(fullSummary, stepNum)
				evaluate(model, self.testData, summaryOn=True, stepNum=stepNum)
			else:
				loss, accuracy, _ = model.run([loss_, accuracy_, trainOp],
                                              feed_dict={state_: stateBatch,
														 actionLabel_: actionLabelBatch,
														 valueLabel_: valueLabelBatch})

			if stepNum % self.reportInterval == 0:
				print("#{} loss: {}, acc: {}".format(stepNum, loss, accuracy))

			if terminalCond:
				break

			lossHistory[stepNum % self.lossHistorySize] = loss
			terminalCond = bool(np.std(lossHistory) < self.lossChangeThreshold)

		return model


class TrainWithMiniBatch:
	def __init__(self,
                 maxStepNum, learningRate, batchSize, lossChangeThreshold, lossHistorySize,
                 reportInterval,
                 summaryOn=False, testData=None):
		self.maxStepNum = maxStepNum
		self.learningRate = learningRate
		self.batchSize = batchSize
		self.lossChangeThreshold = lossChangeThreshold
		self.lossHistorySize = lossHistorySize

		self.reportInterval = reportInterval

		self.summaryOn = summaryOn
		self.testData = testData

	def __call__(self, model, trainingData):
		graph = model.graph
		state_, actionLabel_, valueLabel_ = graph.get_collection_ref("inputs")
		loss_ = graph.get_collection_ref("loss")[0]
		accuracy_ = graph.get_collection_ref("actionAccuracy")[0]
		trainOp = graph.get_collection_ref(tf.GraphKeys.TRAIN_OP)[0]
		fullSummaryOp = graph.get_collection_ref('summaryOps')[0]
		trainWriter = graph.get_collection_ref('writers')[0]

		lossHistory = np.ones(self.lossHistorySize)
		terminalCond = False

		for stepNum in range(self.maxStepNum):
			stateBatch, actionLabelBatch, valueLabelBatch = data.sampleData(list(zip(*trainingData)), self.batchSize)

			if self.summaryOn and (stepNum % self.reportInterval == 0 or stepNum == self.maxStepNum-1 or terminalCond):
				loss, accuracy, _, fullSummary = model.run([loss_, accuracy_, trainOp, fullSummaryOp],
                                                        	feed_dict={state_: stateBatch,
                                                                      actionLabel_: actionLabelBatch,
																	  valueLabel_: valueLabelBatch})
				trainWriter.add_summary(fullSummary, stepNum)
				evaluate(model, self.testData, summaryOn=True, stepNum=stepNum)
			else:
				loss, accuracy, _ = model.run([loss_, accuracy_, trainOp],
                                              feed_dict={state_: stateBatch,
														 actionLabel_: actionLabelBatch,
														 valueLabel_: valueLabelBatch})

			if stepNum % self.reportInterval == 0:
				print("#{} loss: {}, acc: {}".format(stepNum, loss, accuracy))

			if terminalCond:
				break

			lossHistory[stepNum % self.lossHistorySize] = loss
			terminalCond = bool(np.std(lossHistory) < self.lossChangeThreshold)

		return model


def evaluate(model, testData, summaryOn=False, stepNum=None):
	graph = model.graph
	state_, actionLabel_, valueLabel_ = graph.get_collection_ref("inputs")
	loss_ = graph.get_collection_ref("loss")[0]
	valueLoss_ = graph.get_collection_ref("valueLoss")[0]
	accuracy_ = graph.get_collection_ref("actionAccuracy")[0]
	evalSummaryOp = graph.get_collection_ref('summaryOps')[1]
	testWriter = graph.get_collection_ref('writers')[1]

	stateBatch, actionLabelBatch, valueLabelBatch = testData

	if summaryOn:
		loss, accuracy, valueLoss, evalSummary = model.run([loss_, accuracy_, valueLoss_, evalSummaryOp],
                                                		   feed_dict={state_: stateBatch,
														   			  actionLabel_: actionLabelBatch,
														   			  valueLabel_: valueLabelBatch})
		testWriter.add_summary(evalSummary, stepNum)
	else:
		loss, accuracy, valueLoss = model.run([loss_, accuracy_, valueLoss_],
                                   			  feed_dict={state_: stateBatch,
											  			 actionLabel_: actionLabelBatch,
											  			 valueLabel_: valueLabelBatch})
	return loss, accuracy, valueLoss


def approximatePolicy(stateBatch, policyValueNet, actionSpace):
	if np.array(stateBatch).ndim == 1:
		stateBatch = np.array([stateBatch])
	graph = policyValueNet.graph
	state_ = graph.get_collection_ref("inputs")[0]
	actionIndices_ = graph.get_collection_ref("actionIndices")[0]
	actionIndices = policyValueNet.run(actionIndices_, feed_dict={state_: stateBatch})
	actionBatch = [actionSpace[i] for i in actionIndices]
	if len(actionBatch) == 1:
		actionBatch = actionBatch[0]
	return actionBatch


def approximateValueFunction(stateBatch, policyValueNet):
	if np.array(stateBatch).ndim == 1:
		stateBatch = np.array([stateBatch])
	graph = policyValueNet.graph
	state_ = graph.get_collection_ref("inputs")[0]
	valuePrediction_ = graph.get_collection_ref("valuePrediction")[0]
	valuePrediction = policyValueNet.run(valuePrediction_, feed_dict={state_: stateBatch})
	if len(valuePrediction) == 1:
		valuePrediction = valuePrediction[0][0]
	return valuePrediction
