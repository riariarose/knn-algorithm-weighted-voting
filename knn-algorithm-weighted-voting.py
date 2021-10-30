'''---------------------------------------------
CSC535-Data Mining
HW3: KNN implementation with weighted voting
------------------------------------------------'''
import pandas as pd
import operator
import math

train = pd.read_csv("MNIST_train.csv")
test = pd.read_csv('MNIST_test.csv')

trainingSet = train.values.tolist()
testingSet = test.values.tolist()

def calcEuclideanDistance(a, b, length):
    dist = 0.0
    for i in range(length):
        dist += pow(float(a[i]) - float(b[i]), 2)
    distance = math.sqrt(dist)
    return distance

#function to find k neighbors with the closest euclid. dist. to test set
def getNeighbors(trainingSet, testingSet, k):
    length = len(testingSet)-1
    distances = []
    neighbors = []
    for i in range(len(testingSet)):
        dist = calcEuclideanDistance(testingSet, trainingSet[i], length)
        distances.append((trainingSet[i], dist))
    distances.sort(key=operator.itemgetter(1))

    for i in range(k):
        neighbors.append(distances[i][0])
    return neighbors

#function to calculate the vote of a neighbor using weighted voting
def calcNeighborVote(neighbors):
    Votes = {}
    for i in range(len(neighbors)):
        response = neighbors[i][0]
        if response in Votes:
            Votes[response] += 1
        else:
            Votes[response] = 1
    votesSorted = sorted(Votes.items(), key=operator.itemgetter(1), reverse=True) 
    voteReturned = votesSorted[0][0]
    return voteReturned

def calcAccuracyRate(testCount, testingSet, predictions):
    correct = 0
    incorrect = 0
    for i in range(testCount):
        if testingSet[i][0] == predictions[i]:
            correct += 1
        else:
            incorrect += 1
    accuracy = correct/(len(testingSet)) * 100.0
    return accuracy, incorrect

def KNN(k, testCount):
    predictions = []
    accuracy = []
    for i in range(testCount):
        neighbors = getNeighbors(trainingSet, testingSet[i], k)
        result = calcNeighborVote(neighbors)
        predictions.append(result)
        print('> Desired class:' + repr(result) +', computed class:' + repr(testingSet[i][0]))
    accuracy = calcAccuracyRate(testCount, testingSet, predictions)
    print("K = " + repr(k))
    print('Accuracy: ' + repr(accuracy[0]) + '%')
    print('Number of misclassified test samples: ' + repr(accuracy[1]))
    print('Total number of test samples: ' + repr(testCount))

def main():
    k = 3
    testCount = len(testingSet)
    print("K = " + repr(k))
    KNN(k, testCount)
main()
