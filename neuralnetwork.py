#!/usr/bin/env python
# encoding: utf-8
#
# Neural Network example in Python
# by Tim Trueman provided under:
# 
# The MIT License
# 
# Copyright (c) 2009 Tim Trueman
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# http://www.opensource.org/licenses/mit-license.php

from math import *
from random import *
import pickle

class Neuron:
  def __init__(self, learningRate, numInputs):
    self.learningRate = learningRate
    self.weights = [random() for r in range(numInputs)]
    self.lastActivation = 0

  def forwardPropagation(self, data):
    sum = reduce(lambda x, y: x + y, map(lambda x, y: x * y, self.weights, data))
    self.lastActivation = self._activationFunction(sum)
    return self.lastActivation

  def backwardPropagation(self, errorTerm, data):
    self.weights = map(lambda x, y: x + self.learningRate * errorTerm * y, self.weights, data)
  
  def _activationFunction(self, s):
    return (1 / (1 + exp(-s)))

class NeuralNetwork:
  def __init__(self, numInputs, numHiddenNeurons, numOutputs):
    self.learningRate = 0.1
    self.numHiddenNeurons = numHiddenNeurons
    self.hiddenNeurons = [Neuron(self.learningRate,numInputs) for r in range(numHiddenNeurons)]
    self.outputNeurons = [Neuron(self.learningRate,len(self.hiddenNeurons)) for r in range(2)]

  def test():
    pass

  def train():
    pass

  def findMax(self, activationValues):
    max = 0
    for i in range(1, len(activationValues)):
      if (activationValues[i] > activationValues[max]):
        max = i
    return max

  def forwardPropagation(self, data):
    self.hiddenActivations, self.outputActivations = [], []
    for neuron in self.hiddenNeurons:
      self.hiddenActivations.append(neuron.forwardPropagation(data))
    for neuron in self.outputNeurons:
      self.outputActivations.append(neuron.forwardPropagation(self.hiddenActivations))
    return self.findMax(self.outputActivations)

  def backwardPropagation(self, data, desiredAction):
    self.forwardPropagation(data)
    self.outputErrors, ErrorTerm, WeightDeltaH = [], 0, 0
    for i in range(len(self.outputNeurons)):
      fire = 1 if (i == desiredAction) else 0
      lastActivation = self.outputNeurons[i].lastActivation
      errorTerm = (fire - lastActivation) * lastActivation * (1 - lastActivation)
      self.outputErrors.append(errorTerm)
      self.outputNeurons[i].backwardPropagation(errorTerm,self.hiddenActivations)
      #print desiredAction, errorTerm, self.hiddenActivations
    for i in range(len(self.hiddenNeurons)):
      weightDeltaH = 0
      for j in range(len(self.outputNeurons)):
        fire = 1 if (j == desiredAction) else 0
        lastActivation = self.hiddenNeurons[i].lastActivation
        weightDeltaH = weightDeltaH + (fire - lastActivation) * lastActivation * (1 - lastActivation) * self.outputNeurons[j].weights[i]
      errorTerm = weightDeltaH * lastActivation * (1 - lastActivation)
      self.hiddenNeurons[i].backwardPropagation(errorTerm,data)

def case(result):
  return "lowercase" if (0==result) else "uppercase"

if __name__ == "__main__":
  nn = NeuralNetwork(8,8,2)
  lowercase, uppercase = [], []
  lowercase.append([0,1,1,0,0,0,0,1]) # a
  uppercase.append([0,1,0,0,0,0,0,1]) # A
  lowercase.append([0,1,1,0,0,0,1,0]) # b
  uppercase.append([0,1,0,0,0,0,1,0]) # B
  lowercase.append([0,1,1,0,0,0,1,1]) # c
  uppercase.append([0,1,0,0,0,0,1,1]) # C
  lowercase.append([0,1,1,0,0,1,0,0]) # d
  uppercase.append([0,1,0,0,0,1,0,0]) # D
  lowercase.append([0,1,1,0,0,1,0,1]) # e
  uppercase.append([0,1,0,0,0,1,0,1]) # E
  lowercase.append([0,1,1,0,0,1,1,0]) # f
  uppercase.append([0,1,0,0,0,1,1,0]) # F
  lowercase.append([0,1,1,0,0,1,1,1]) # g
  uppercase.append([0,1,0,0,0,1,1,1]) # G
  lowercase.append([0,1,1,0,1,0,0,0]) # h
  uppercase.append([0,1,0,0,1,0,0,0]) # H
  lowercase.append([0,1,1,0,1,0,0,1]) # i
  uppercase.append([0,1,0,0,1,0,0,1]) # I
  lowercase.append([0,1,1,0,1,0,1,0]) # j
  uppercase.append([0,1,0,0,1,0,1,0]) # J
  lowercase.append([0,1,1,0,1,0,1,1]) # k
  uppercase.append([0,1,0,0,1,0,1,1]) # K
  lowercase.append([0,1,1,0,1,1,0,0]) # l
  uppercase.append([0,1,0,0,1,1,0,0]) # L
  lowercase.append([0,1,1,0,1,1,0,1]) # m
  uppercase.append([0,1,0,0,1,1,0,1]) # M
  lowercase.append([0,1,1,0,1,1,1,0]) # n
  uppercase.append([0,1,0,0,1,1,1,0]) # N
  lowercase.append([0,1,1,0,1,1,1,1]) # o
  uppercase.append([0,1,0,0,1,1,1,1]) # O
  lowercase.append([0,1,1,1,0,0,0,0]) # p
  uppercase.append([0,1,0,1,0,0,0,0]) # P
  lowercase.append([0,1,1,1,0,0,0,1]) # q
  uppercase.append([0,1,0,1,0,0,0,1]) # Q
  lowercase.append([0,1,1,1,0,0,1,0]) # r
  uppercase.append([0,1,0,1,0,0,1,0]) # R
  lowercase.append([0,1,1,1,0,0,1,1]) # s
  uppercase.append([0,1,0,1,0,0,1,1]) # S
  lowercase.append([0,1,1,1,0,1,0,0]) # t
  uppercase.append([0,1,0,1,0,1,0,0]) # T
  lowercase.append([0,1,1,1,0,1,0,1]) # u
  uppercase.append([0,1,0,1,0,1,0,1]) # U
  lowercase.append([0,1,1,1,0,1,1,0]) # v
  uppercase.append([0,1,0,1,0,1,1,0]) # V
  lowercase.append([0,1,1,1,0,1,1,1]) # w
  uppercase.append([0,1,0,1,0,1,1,1]) # W
  lowercase.append([0,1,1,1,1,0,0,0]) # x
  uppercase.append([0,1,0,1,1,0,0,0]) # X
  lowercase.append([0,1,1,1,1,0,0,1]) # y
  uppercase.append([0,1,0,1,1,0,0,1]) # Y
  lowercase.append([0,1,1,1,1,0,1,0]) # z
  uppercase.append([0,1,0,1,1,0,1,0]) # Z
  print "Untrained:"
  print "a is", case(nn.forwardPropagation(lowercase[0]))
  print "A is", case(nn.forwardPropagation(uppercase[0]))
  for i in range(100):
    for letter in lowercase:
      nn.backwardPropagation(letter,0)
    for letter in uppercase:
      nn.backwardPropagation(letter,1)
  print "Trained:"
  print "a is", case(nn.forwardPropagation(lowercase[0]))
  print "A is", case(nn.forwardPropagation(uppercase[0]))
  # print pickle.dumps(nn.outputNeurons)
  # print pickle.dumps(NeuralNetwork(8,8,2))
