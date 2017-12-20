import numpy as np
import random

from helper_functions_from_Stanford.softmax import softmax_sol as softmax
from helper_functions_from_Stanford.sigmoid import sigmoid_sol as sigmoid
from helper_functions_from_Stanford.sgd import *

def softmaxCostAndGradient(predicted, target, outputVectors, dataset):
    """ Softmax cost function for word2vec models """
    probabilities = softmax(predicted.dot(outputVectors.T))
    cost = -np.log(probabilities[target])
    delta = probabilities
    delta[target] -= 1
    N = delta.shape[0]
    D = predicted.shape[0]
    grad = delta.reshape((N,1)) * predicted.reshape((1,D))
    gradPred = (delta.reshape((1,N)).dot(outputVectors)).flatten()

    return cost, gradPred, grad

 
def skipgram(currentWord, C, contextWords, tokens, inputVectors, outputVectors, 
    dataset, word2vecCostAndGradient = softmaxCostAndGradient):
    """ Skip-gram model in word2vec """                                 
    currentI = tokens[currentWord]
    predicted = inputVectors[currentI, :]

    cost = 0.0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)
    for cwd in contextWords:
        idx = tokens[cwd]
        cc, gp, gg = word2vecCostAndGradient(predicted, idx, outputVectors, dataset)
        cost += cc
        gradOut += gg
        gradIn[currentI, :] += gp
    
    return cost, gradIn, gradOut

def negSamplingCostAndGradient(predicted, target, outputVectors, dataset, 
    K=10):
    """ Negative sampling cost function for word2vec models """
    grad = np.zeros(outputVectors.shape)
    gradPred = np.zeros(predicted.shape)

    indices = [target]
    for k in range(K):
        newidx = dataset.sampleTokenIdx()
        while newidx == target:
            newidx = dataset.sampleTokenIdx()
        indices += [newidx]

    labels = np.array([1] + [-1 for k in range(K)])
    vecs = outputVectors[indices,:]

    t = sigmoid(vecs.dot(predicted) * labels)
    cost = -np.sum(np.log(t))

    delta = labels * (t - 1)
    gradPred = delta.reshape((1,K+1)).dot(vecs).flatten()
    gradtemp = delta.reshape((K+1,1)).dot(predicted.reshape(
        (1,predicted.shape[0])))
    for k in range(K+1):
        grad[indices[k]] += gradtemp[k,:]
    
    return cost, gradPred, grad

def normalizeRows_sol(x):
    """ Row normalization function """
    # Implement a function that normalizes each row of a matrix to have unit length
    N = x.shape[0]
    x /= np.sqrt(np.sum(x**2, axis=1)).reshape((N,1)) + 1e-30
    return x


