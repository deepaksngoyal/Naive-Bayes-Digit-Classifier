# Naive-Bayes-HandWritten-Digit-Classifier


INTRODUCTION:
In this assignment, we have implemented Naïve Bayes Classifier for digits reorganization. First of all, it will train the classifier with training data, and then we can recognize handwritten digits using this classifier. It will precisely calculate the probabilities of “white pixel”, “black pixel” and “grey pixel” while training. Then depending upon the probabilities of each pixel at each point it will give approximate digit during testing.

Naïve Bayes Classifier:
In naïve bayes classifier for digit recognizer, first it calculates the prior probability for each label in the given data. Then it calculates probabilities of “white”, “black” and “grey” values for each pixel for each digit in the training data. Then for a given test sample we check each pixel value in sample and find the probability of that pixel value in stored data. We multiply all these probabilities with the prior probability of each label and do it for each label and return the Vmap.

Laplace Smoothing:
Laplace Smoothing is used to compute the prior probabilities of  each pixel values  to avoid overfitting problem and erronous training data.
