import tensorflow as tf
import numpy as np


def train(stepNum, trainingData, model, summaryOn=True):
    stateBatch, actionLabelBatch = trainingData
    graph = model.graph
    state_, actionLabel_ = graph.get_collection_ref("inputs")
    loss_ = graph.get_collection_ref("loss")[0]
    accuracy_ = graph.get_collection_ref("accuracy")[0]
    trainOp = graph.get_collection_ref(tf.GraphKeys.TRAIN_OP)[0]
    fullSummaryOp = graph.get_collection_ref('summaryOps')[0]
    trainWriter = graph.get_collection_ref('writers')[0]

    if summaryOn:
        loss, accuracy, _, fullSummary = model.run([loss_, accuracy_, trainOp, fullSummaryOp],
                                                   feed_dict={state_: stateBatch, actionLabel_: actionLabelBatch})
        trainWriter.add_summary(fullSummary, stepNum)
    else:
        loss, accuracy, _ = model.run([loss_, accuracy_, trainOp],
                                      feed_dict={state_: stateBatch, actionLabel_: actionLabelBatch})
    return model, loss, accuracy


class Evaluate:
    def __init__(self, testData):
        self.testData = testData

    def __call__(self, stepNum, model, summaryOn=True):
        stateBatch, actionLabelBatch = self.testData

        graph = model.graph
        state_, actionLabel_ = graph.get_collection_ref("inputs")
        loss_ = graph.get_collection_ref("loss")[0]
        accuracy_ = graph.get_collection_ref("accuracy")[0]
        evalSummaryOp = graph.get_collection_ref('summaryOps')[1]
        testWriter = graph.get_collection_ref('writers')[1]

        if summaryOn:
            loss, accuracy, evalSummary = model.run([loss_, accuracy_, evalSummaryOp],
                                                    feed_dict={state_: stateBatch, actionLabel_: actionLabelBatch})
            testWriter.add_summary(evalSummary, stepNum)
        else:
            loss, accuracy = model.run([loss_, accuracy_],
                                       feed_dict={state_: stateBatch, actionLabel_: actionLabelBatch})
        return loss, accuracy


class Learn:
    def __init__(self,
                 maxStepNum, learningRate, lossChangeThreshold,
                 trainingData, testData,
                 summaryOn, reportInterval):
        self.maxStepNum = maxStepNum
        self.learningRate = learningRate
        self.lossChangeThreshold = lossChangeThreshold

        self.trainingData = trainingData
        self.evaluateOnTrainingData = Evaluate(trainingData)
        self.evaluate = Evaluate(testData)

        self.summaryOn = summaryOn
        self.reportInterval = reportInterval

    def __call__(self, model):
        lossHistorySize = 5
        lossHistory = np.ones(lossHistorySize)
        terminalCond = False

        for stepNum in range(self.maxStepNum):
            if self.summaryOn and (stepNum % self.reportInterval == 0 or stepNum == self.maxStepNum-1 or terminalCond):
                newModel, trainLoss, _ = train(stepNum, self.trainingData, model, summaryOn=True)
                _, _ = self.evaluate(stepNum, model, summaryOn=True)
            else:
                newModel, trainLoss, _ = train(stepNum, self.trainingData, model, summaryOn=False)
            model = newModel

            if stepNum % self.reportInterval == 0:
                print("#{} loss: {}".format(stepNum, trainLoss))

            if terminalCond:
                break

            lossHistory[stepNum % lossHistorySize] = trainLoss
            terminalCond = bool(np.std(lossHistory) < 1e-8)

        trainLoss, trainAccuracy = self.evaluateOnTrainingData(stepNum, newModel, summaryOn=False)
        testLoss, testAccuracy = self.evaluate(stepNum, newModel, summaryOn=False)

        return newModel, trainLoss, trainAccuracy, testLoss, testAccuracy
