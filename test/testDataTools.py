import unittest
from ddt import ddt, data, unpack
import numpy as np
from dataTools import generateSymmetricData
import sheepEscapingEnv as env
xbias = env.xBoundary[1]
ybias = env.yBoundary[1]

@ddt
class TestGenerateData(unittest.TestCase):
	# (0, 1), (1, 0), (-1, 0), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)
	@data(([np.array([[10, 5, 20, 5], [0.25, 0.3, 0, 0, 0.45, 0, 0, 0], [20]])],
		   [np.array([[10, 5, 20, 5], [0.25, 0.3, 0, 0, 0.45, 0, 0, 0], [20]]),
		   np.array([[5, 10, 5, 20], [0.3, 0.25, 0, 0, 0.45, 0, 0, 0], [20]]), # symmetry: [1,1]
           np.array([[-10+xbias, 5, -20+xbias, 5], [0.25, 0, 0.3, 0, 0, 0, 0, 0.45], [20]]),  # symmetry: [0,1]
           np.array([[10, -5+ybias, 20, -5+ybias], [0, 0.3, 0, 0.25, 0, 0, 0.45, 0], [20]]),  # symmetry: [1,0]
           np.array([[-5+xbias, -10+ybias, -5+xbias, -20+ybias], [0, 0, 0.25, 0.3, 0, 0.45, 0, 0], [20]]),  # symmetry: [-1,1]
           np.array([[-5+xbias, 10, -5+xbias, 20], [0.3, 0, 0.25, 0, 0, 0, 0, 0.45], [20]]),  # symmetry: [0,1]
           np.array([[5, -10+ybias, 5, -20+ybias], [0, 0.25, 0, 0.3, 0, 0, 0.45, 0], [20]]),  # symmetry: [1,0]
           np.array([[-10+xbias, -5+ybias, -20+xbias, -5+ybias], [0, 0, 0.3, 0.25, 0, 0.45, 0, 0], [20]])]  # symmetry: [-1,1]
	       ))
	@unpack
	def testGenerateSymmetricData(self, originalDataSet, groundTruth):
		sysmetricDataSet = generateSymmetricData(originalDataSet, None)
		for data in sysmetricDataSet:
			for truthData in groundTruth:
				if np.allclose(data[0], np.array(truthData[0])):
					self.assertSequenceEqual(list(data[1]), list(truthData[1]))



if __name__ == "__main__":
	unittest.main()


