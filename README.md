# Recognizing Handwritten Digits using a Two-layer Perceptron

In course of a seminar on “Selected Topics in Human Language Technology and Pattern Recognition”, I wrote a seminar paper on neural networks: "Introduction to Neural Networks". The seminar paper and the slides of the corresponding talk can be found in my blog article: [Seminar Paper “Introduction to Neural Networks”](http://davidstutz.de/seminar-paper-introduction-neural-networks/). Background on neural networks and the two-layer perceptron can be found in my seminar paper.

**Update:** The code can be adapted to allow mini-batch training as done in [this fork](https://github.com/Myasuka/matlab-mnist-two-layer-perceptron).

## MNIST Dataset

The [MNIST dataset](http://yann.lecun.com/exdb/mnist/) provides a training set of 60,000 handwritten digits and a validation set of 10,000 handwritten digits. The images have size 28 x 28 pixels. Therefore, when using a two-layer perceptron, we need 28 x 28 = 784 input units and 10 output units (representing the 10 different digits).

The methods `loadMNISTImages` and `loadMNISTLaels` are used to load the MNIST dataset as it is stored in a special file format. The methods can be found online at [http://ufldl.stanford.edu/wiki/index.php/Using_the_MNIST_Dataset](http://ufldl.stanford.edu/wiki/index.php/Using_the_MNIST_Dataset).

## Methods and Usage

The main method to train the two-layer perceptron is `trainStochasticSquaredErrorTwoLayerPerceptron`. The method applies stochastic training (or to be precise a stochastic variant of mini-batch training) using the sum-of-squared error function and the error backpropagation algorithm.

	function [hiddenWeights, outputWeights, error] = trainStochasticSquaredErrorTwoLayerPerceptron(activationFunction, dActivationFunction, numberOfHiddenUnits, inputValues, targetValues, epochs, batchSize, learningRate)
	% trainStochasticSquaredErrorTwoLayerPerceptron Creates a two-layer perceptron
	% and trains it on the MNIST dataset.
	%
	% INPUT:
	% activationFunction             : Activation function used in both layers.
	% dActivationFunction            : Derivative of the activation
	% function used in both layers.
	% numberOfHiddenUnits            : Number of hidden units.
	% inputValues                    : Input values for training (784 x 60000)
	% targetValues                   : Target values for training (1 x 60000)
	% epochs                         : Number of epochs to train.
	% batchSize                      : Plot error after batchSize images.
	% learningRate                   : Learning rate to apply.
	%
	% OUTPUT:
	% hiddenWeights                  : Weights of the hidden layer.
	% outputWeights                  : Weights of the output layer.
	% 

The above method requires the activation function used for both the hidden and the output layer to be given as parameter. I used the logistic sigmoid activation function:

	function y = logisticSigmoid(x)
	% simpleLogisticSigmoid Logistic sigmoid activation function
	% 
	% INPUT:
	% x     : Input vector.
	%
	% OUTPUT:
	% y     : Output vector where the logistic sigmoid was applied element by
	% element.
	%
	
In addition, the error backpropagation algorithm needs the derivative of the used activation function:

	function y = dLogisticSigmoid(x)
	% dLogisticSigmoid Derivative of the logistic sigmoid.
	% 
	% INPUT:
	% x     : Input vector.
	%
	% OUTPUT:
	% y     : Output vector where the derivative of the logistic sigmoid was
	% applied element by element.
	%
	
The method `applyStochasticSquaredErrorTwoLayerPerceptronMNIST` uses both the training method seen above and the method `validateTwoLayerPerceptron` to evaluate the performance of the two-layer perceptron:

	function [correctlyClassified, classificationErrors] = validateTwoLayerPerceptron(activationFunction, hiddenWeights, outputWeights, inputValues, labels)
	% validateTwoLayerPerceptron Validate the twolayer perceptron using the
	% validation set.
	%
	% INPUT:
	% activationFunction             : Activation function used in both layers.
	% hiddenWeights                  : Weights of the hidden layer.
	% outputWeights                  : Weights of the output layer.
	% inputValues                    : Input values for training (784 x 10000).
	% labels                         : Labels for validation (1 x 10000).
	%
	% OUTPUT:
	% correctlyClassified            : Number of correctly classified values.
	% classificationErrors           : Number of classification errors.
	% 
	
## License 

Copyright 2013 - 2014 David Stutz

The application is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

The application is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

See <http://www.gnu.org/licenses/>.
