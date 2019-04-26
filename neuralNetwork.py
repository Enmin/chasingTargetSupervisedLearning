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
			fullyConnected_ = tf.layers.dense(inputs=state_, units=hiddenWidth, activation=tf.nn.relu)
			for _ in range(hiddenDepth-1):
				fullyConnected_ = tf.layers.dense(inputs=fullyConnected_, units=self.numActionSpace, activation=tf.nn.relu)
			allActionActivation_ = tf.layers.dense(inputs=fullyConnected_, units=self.numActionSpace, activation=None)

		with tf.name_scope("outputs"):
			actionDistribution_ = tf.nn.softmax(allActionActivation_, name='actionDistribution_')
			cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=actionDistribution_, labels=actionLabel_, name='cross_entropy')
			loss_ = tf.reduce_mean(cross_entropy, name='loss_')
			tf.summary.scalar("Loss", loss_)

		with tf.name_scope("train"):
			trainOpt_ = tf.train.AdamOptimizer(self.learningRate, name='adamOpt_').minimize(loss_)

		mergedSummary = tf.summary.merge_all()

		model = tf.Session()
		model.run(tf.global_variables_initializer())

		return model