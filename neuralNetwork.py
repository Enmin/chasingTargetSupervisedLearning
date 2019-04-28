import tensorflow as tf


class GeneratePolicyNet:
	def __init__(self, numStateSpace, numActionSpace, learningRate, optimalPolicy, actionSpace):
		self.numStateSpace = numStateSpace
		self.numActionSpace = numActionSpace
		self.learningRate = learningRate
		self.optimalPolicy = optimalPolicy
		self.actionSpace = actionSpace

	def __call__(self, hiddenDepth, hiddenWidth):
		print("Generating Policy Net with hidden layers: {} x {} = {}".format(hiddenDepth, hiddenWidth, hiddenDepth * hiddenWidth))
		with tf.name_scope("inputs"):
			state_ = tf.placeholder(tf.float32, [None, 4], name="state_")
			actionLabel_ = tf.placeholder(tf.int32, [None, 8], name="actionLabel_")
			# accumulatedRewards_ = tf.placeholder(tf.float32, [None, ], name="accumulatedRewards_")

		with tf.name_scope("hidden"):
			# initWeight = tf.random_uniform_initializer(-0.03, 0.03)
			# initBias = tf.constant_initializer(0.001)
			fc1 = tf.layers.Dense(units=hiddenWidth, activation=tf.nn.relu)
			a1_ = fc1(state_)

			w1_, b1_ = fc1.weights
			tf.summary.histogram("w1", w1_)
			tf.summary.histogram("b1", b1_)
			tf.summary.histogram("a1", a1_)

			a_ = a1_
			for i in range(2, hiddenDepth+1):
				fc = tf.layers.Dense(units=hiddenWidth, activation=tf.nn.relu)
				aNext_ = fc(a_)
				a_ = aNext_

				w_, b_ = fc.weights
				tf.summary.histogram("w{}".format(i), w_)
				tf.summary.histogram("b{}".format(i), b_)
				tf.summary.histogram("a{}".format(i), a_)

			fcLast = tf.layers.Dense(units=self.numActionSpace, activation=None)
			allActionActivation_ = fcLast(a_)

			wLast_, bLast_ = fcLast.weights
			tf.summary.histogram("wLast", wLast_)
			tf.summary.histogram("bLast", bLast_)
			tf.summary.histogram("allActionActivation", allActionActivation_)

		with tf.name_scope("outputs"):
			actionDistribution_ = tf.nn.softmax(allActionActivation_, name='actionDistribution_')
			cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=actionDistribution_, labels=actionLabel_, name='cross_entropy')
			loss_ = tf.reduce_mean(cross_entropy, name='loss_')
			tf.summary.scalar("loss", loss_)

		with tf.name_scope("train"):
			optimizer = tf.train.AdamOptimizer(self.learningRate, name='adamOpt_')
			gradVarPairs_ = optimizer.compute_gradients(loss_)
			trainOp = optimizer.apply_gradients(gradVarPairs_)

			tf.add_to_collection(tf.GraphKeys.TRAIN_OP, trainOp)

			gradients_ = [tf.reshape(grad, [1, -1]) for (grad, _) in gradVarPairs_]
			gradTensor_ = tf.concat(gradients_, 1)
			gradMax_ = tf.reduce_max(gradTensor_)
			gradMin_ = tf.reduce_min(gradTensor_)

			tf.add_to_collection("gradRange", gradMin_)
			tf.add_to_collection("gradRange", gradMax_)

			tf.summary.histogram("gradients", gradTensor_)
			tf.summary.scalar('gradMax', gradMax_)
			tf.summary.scalar('gradMin', gradMin_)


		mergedSummary = tf.summary.merge_all()
		tf.add_to_collection("mergedSummary", mergedSummary)
		writer = tf.summary.FileWriter("./tbdata", graph=tf.get_default_graph())
		tf.add_to_collection("writer", writer)

		model = tf.Session()
		model.run(tf.global_variables_initializer())

		return model
