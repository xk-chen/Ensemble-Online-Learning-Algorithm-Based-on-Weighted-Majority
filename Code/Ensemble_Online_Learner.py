# -*- coding: utf-8 -*-

"""
Ensemble Online Learner.
Copyright (c) 2020, Xingke Chen. All rights reserved.
@author:  Xingke Chen
@email:   chenxk1229@hotmail.com
@license: GPL-v3.

Warning: this code may contain factual errors, please check it out before using it.

"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

class BaseLearner:
    '''
    Toy example: the `BaseLearner` is just a classifier with random guess.
    For other learners, please implement by yourself.
    '''
    def __init__(self, prob):
        '''
        Initialize.
        
        Parameter:
        ----------
        prob: the probability of the learner predicts an instance as a positive one.
        
        Returns:
        ----------
        None
        '''
        self.prob = prob
    def predict(self, x):
        '''
        Predict the label of an instance `x`.
        
        Parameters
        ----------
        x: array-like, shape [num_features, 1]
            Data point to be classified.
        
        Returns
        ----------
        prediction: int, takes value in {-1, +1}
            The prediction of `x`.
        '''
        prediction = np.random.choice([-1,1], size = 1, p = [1-self.prob, self.prob])[0]
        return prediction
    
class ErrorDetector:
    '''
    Detect whether a data point is influential to `classifier` model.
    '''
    def __init__(self, X, y, classifier = "LogisticRegression", *args):
        '''
        Initialize the object.
        
        Parameters
        ----------
        X: array-like, shape [num_reliable_instances, num_features].
           The reliable annotation data.
        
        y: array-like, shape [num_reliable_instances, 1].
           The corresponding labels of `X`.
        
        classifier: string, 
           The name of the classifier to be constructed.
        
        *args: list, 
           Additional arguments for classifier (such as `C`, `class_weight`, etc.).
        
        Returns
        ----------
        None
        '''
        classifier_collection = {"LogisticRegression": LogisticRegression}
        self.classifier = classifier_collection[classifier](*args)
        self.X = X
        self.y = y
        
    def sample_append(self, x, y):
        '''
        Append new training data into the training set.
        
        Parameters
        ----------
        x: array-like, shape [num_features, 1].
           Single reliable annotation data.
        
        y: int, takes value in {-1, +1}.
           The corresponding label of x.
        
        Returns
        ----------
        None
        '''
        self.X = np.vstack((self.X, x))
        self.y = np.hstack((self.y, y))
        
    def fit(self):
        '''
        Fit the classifier using the training set.
        
        Parameters
        ----------
        None
        
        Returns
        ----------
        None
        
        '''
        self.classifier.fit(self.X, self.y)
        self.parameters = self.classifier.coef_.flatten()
        
    def get_influence_function(self, target_point, test_point, loss_function = "CrossEntropy"):
        '''
        Compute the influence function (for Logistic Regression so far).
        
        Parameters
        ----------
        target_point: tuple, consists of the instance (x, y)
                      shape of x: [num_features, 1]
                      y: int, takes value in {-1, +1}
                      
        test_point: tuple, consists of the instance (x_test, y_test)
                    shape of x_test: [num_features, 1]
                    y_test: int, takes value in {-1, +1}
                    
        loss_function: string, the name of loss function
        
        Returns
        ----------
        I: float, the influence (change) of upweighting `target_point` on the loss at the test point `test_point`
        '''
        if loss_function != "CrossEntropy":
            raise NotImplementatedError, "Influence function of " + loss_function + "has not been implemented yet."
        x, y = target_point
        x_test, y_test = test_point
        gradient_target = self._gradient(x, y)
        gradient_test = self._gradient(x_test, y_test)
        Hessian = self._Hessian()
        I = -np.dot(gradient_target, np.dot(np.linalg.pinv(Hessian), gradient_test))
        return I 
    
    def _gradient(self, x, y):
        '''
        Compute the gradient w.r.t. the loss of `classifier`
        at the point (`x`, `y`, `self.parameters`)
        
        Parameters
        ----------
        x: array-like, shape [num_features, 1].
           an instance in feature space.
        
        y: int, takes value in {-1, +1}.
           The corresponding label of x.
        
        Returns
        ----------
        gradient: array-like, shape [num_features, 1]
           the gradient at (`x`, `y`, `self.parameters`).
        
        '''
        gradient = -self._sigmoid(-y * np.dot(x, self.parameters)) * y * x
        return gradient
    
    def _Hessian(self):
        '''
        Compute the Hessian matrix w.r.t. the loss of `classifier`.
        
        Parameters
        ----------
        None
        
        Returns
        ----------
        Hessian: array-like, shape [num_features, num_features]
           the Hessian matrix of the loss of `classifier`.
        
        '''
        Hessian = np.matmul(self._sigmoid(np.dot(self.X, self.parameters)) * self.X.T, 
                           (self._sigmoid(-np.dot(self.X, self.parameters)) * self.X.T).T)
        return Hessian
    
    def _sigmoid(self, x):
        '''
        Compute the sigmoid trasform of `x`.
        
        Parameters
        ----------
        x: array-like, shape [arbitrary, 1]
        
        Returns
        ----------
        s_transform: array_like, the shape is the same as `x`.
        
        '''
        s_transform = 0.5 + 0.5 * np.tanh(0.5 * x)
        return s_transform
    
class EnsembleOnlineLearner:
    '''
    The implementation of ensemble online algorithm.
    '''
    def __init__(self, X_reliable, y_reliable, base_learners_collection, correction_parameter):
        '''
        Initialize the object. This method creats an instance of `ErrorDetector` object and fit the classifier
        in `ErrorDetector` with reliable dataset (`X_reliable`, `y_reliable`).
        
        Parameters
        ----------
        X_reliable: array-like, shape [num_reliable_instances, num_features].
            The reliable annotation data.
        
        y_reliable: array-like, shape [num_reliable_instances, 1].
            The corresponding labels of `X_reliable`.
        
        base_learners_collection: list.
            The list of base learner objects.
            
        correction_parameter: float.
            The hyperparameter that determines whether an output is valid.
            
        Returns
        ----------
        None
        '''
        self.base_learners_collection = base_learners_collection
        self.num_base_learners = len(self.base_learners_collection)
        self.weight = 1.0 / self.num_base_learners * np.ones(self.num_base_learners)
        self.correction_parameter = correction_parameter
        self.num_error = 0
        self.error_rate_collection = []
        self.step_counter = 0
        self.error_detector = ErrorDetector(X_reliable, y_reliable)
        self.error_detector.fit()
        
    def sampling(self):
        '''
        Sample the base learner according to the multinomial distribution Prob(`self.weight`).
        
        Parameters
        ----------
        None
        
        Returns
        ----------
        outcome: int, the value is in the set {0, 1, ..., self.num_base_learners - 1}.
        
        '''
        outcome = np.argmax(np.random.multinomial(1, self.weight, size = 1))
        return outcome
    
    def predict(self, x):
        '''
        Predict the label of current instance `x`.
        
        Parameters
        ----------
        x: array-like, shape [num_features, 1].
            The instance needs to be classified.
           
        Returns
        ----------
        output: array-like, shape [`self.num_base_learners`, 1], each element takes value in {-1, +1}.
            The prediction labels of `x` given by all classifiers.
        '''
        predict_collection = []
        for base_learner in self.base_learners_collection:
            predict_collection.append(base_learner.predict(x))
        selection = self.sampling()
        output = predict_collection[selection]
        return output, predict_collection
    
    def training(self, X, expert_database, learning_rate):
        '''
        Learning the model via expert advice algorithm.
        
        Parameters
        ----------
        X: array-like, shape [num_observations, num_features]
            Design matrix.
            
        expert_database: array-like, shape [num_observations, 1]
            The simulator of human experts, collecting true labels corresponding to `X`.
            
        learning_rate: float.
        
        '''
        self.X = X
        self.learning_rate = learning_rate
        self.expert_database = expert_database
        num_observations = len(self.X)
        for t in range(num_observations):
            output, predict_collection = self.predict(self.X[t])
            if output != self.expert_database[t]:
                self.num_error += 1
            target_point = (self.X[t], output)
            if -self.error_detector.get_influence_function(target_point, target_point) > self.correction_parameter:
                reliable_label = self.fetch_reliable_labels()
                self.error_detector.sample_append(self.X[t], reliable_label)
                self.error_detector.fit()
                V = (predict_collection == reliable_label)
                numerator = self.weight * np.exp(-self.learning_rate * V)
                self.weight = numerator * 1.0 / np.sum(numerator)
            self.step_counter += 1
            self.error_rate_collection.append(self._get_error_rate())
    def fetch_reliable_labels(self):
        '''
        The interface of fetching human experts' results, here we replace it with a database with true labels.
        
        Parameters
        ----------
        None
        
        Returns
        ----------
        ground_truth: int, takes value in {-1, +1}.
            The true label of current data point.
        '''
        ground_truth = self.expert_database[self.step_counter]
        return ground_truth
    
    def _get_error_rate(self): 
        '''
        Compute the error rate of the classifier at current step.
        
        Parameters
        ----------
        None
        
        Returns
        ----------
        error_rate: float.
            Current error rate.
        
        '''
        return self.num_error * 1.0 / self.step_counter
    
    def plot_error_rate_curve(self, x_label = "Number of samples", y_label = "Misclassification rate"):

        '''
        Plot the error rate curve, where x-axis represents the
        number of samples, y-axis represents the overall
        accuracy from observation 1 to current observation

        Parameters
        ----------
        x_label : string, 
            The name of x-axis.

        y_label = string,
            The name of y-axis.

        Returns
        -------
        The plot of accuracy curve.
            
        '''
        
        plt.plot(self.error_rate_collection)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.show()
