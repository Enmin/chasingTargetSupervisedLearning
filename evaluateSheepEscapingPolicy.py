import policyValueNet as net
import numpy as np
import sheepEscapingEnv as env

class Evaluate:
	def __init__(self, actionSpace, sampleTraj, numOfEpisodes):
		self.actionSpace = actionSpace
		self.sampleTraj = sampleTraj
		self.numOfEpisodes = numOfEpisodes

	def __call__(self, model):
		policy = lambda state: net.approximatePolicy(state, model, self.actionSpace)
		# sheepNaivePolicy = env.SheepNaiveEscapingPolicy(self.actionSpace)
		# policy = lambda state: sheepNaivePolicy(state)
		demoEpisode = [zip(*self.sampleTraj(policy)) for index in range(self.numOfEpisodes)]
		demoStates = [states for states, actions in demoEpisode]
		demoStateLength = [len(traj) for traj in demoStates]
		averageLength = np.mean(demoStateLength)
		variance = np.var(demoStateLength)
		return averageLength, variance