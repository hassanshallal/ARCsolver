from __future__ import division
import numpy as np


def logProd(x):
    logs = np.log(x)
    result = np.sum(logs)
    return result


def NBYPrior(ytrain):
    priors = np.transpose([float(x)/ytrain.shape[0]
                           for x in list(np.bincount(ytrain))])
    return priors


# def NBYPrior(ytrain):
#    uniquesValues = np.unique(ytrain)
#    priors = []
#    for n in range(uniquesValues.shape[0] - 1):
#        priors.append((np.sum(ytrain == uniquesValues[n])/ytrain.shape[0]))

#    priors.append(1 - sum(priors))
    # print(len(priors)) # number of classes
#    return np.array(priors)


# This is to get the conditional probability of x|y for each feature x against each class y
# This is MLE sufficient statistic for categorical binary features
def B_NBXGivenY(XTrain, yTrain):
    overall = np.column_stack((XTrain, yTrain))
    uniques_classes = np.unique(overall[:, -1])

    thetaHat = []

    for n in range(uniques_classes.shape[0]):
        current_class = overall[overall[:, -1] == uniques_classes[n]]
        thetaHat.append((current_class.sum(axis=0)[
                        0:(current_class.shape[1]-1):]+1)/(current_class.shape[0]+2))

    # print(len(thetaHat)) : number of classes
    # print(thetaHat[0].shape): number of features

    # We convert the list into a numpy array in order to facilitate vecorized calculations latter
    return np.array(thetaHat)


def B_LSNBXGivenY(XTrain, yTrain):
    overall = np.column_stack((XTrain, yTrain))
    uniques_classes = np.unique(overall[:, -1])

    thetaHat = []

    for n in range(uniques_classes.shape[0]):
        current_class = overall[overall[:, -1] == uniques_classes[n]]
        thetaHat.append((current_class.sum(axis=0)[
                        0:(current_class.shape[1]-1):]+1)/(current_class.shape[0]+2))

    # print(len(thetaHat)) : number of classes
    # print(thetaHat[0].shape): number of features
    # We convert the list into a numpy array in order to facilitate vecorized calculations latter
    thetaHat = np.array(thetaHat)
    # print(thetaHat)
    b = 1 - thetaHat[0]
    d = 1 - thetaHat[1]
    # print(b)
    # print(d)
    bdoverbplusd = (b*d)/(b + d)
    acoveraplusc = (thetaHat[0]*thetaHat[1])/(thetaHat[0] + thetaHat[1])
    # print(bdoverbplusd)
    # print(acoveraplusc)
    thetaHat = (thetaHat + bdoverbplusd) / (1 + acoveraplusc + bdoverbplusd)
    # print(thetaHat) # these are probabilities and can't exceed 1! This is why all the predictions are 0!

    # print("=============")
    return thetaHat


# Helper function to compute the bernoulli likelihood of feature xi conditional on class
def B_multiple_feat_log_likelihood(xi, thetahat):
    # xi is an instance
    num_classes = thetahat.shape[0]
    num_features = thetahat.shape[1]
    features_classes_likelihood = np.ones(num_classes)
    for i in range(0, num_classes):
        product = 1
        for j in range(0, num_features):
            product *= (np.power(thetahat[i][j], xi.T[j])) * \
                (np.power((1-thetahat[i][j]), (1-xi.T[j])))

        features_classes_likelihood[i] = product

    return features_classes_likelihood


def B_NBgetcond(XTrain, yTrain, XTest):
    # Estimate class conditional means and conditional variances
    likelihoods = B_NBXGivenY(XTrain, yTrain)

    # Compute the likelihood of the features conditional on each class
    class_likelihood = np.zeros([XTest.shape[0], likelihoods.shape[0]])

    # Non-vectorized, pretty slow
    # for m in range(0, XTest.shape[0]): ## 0:144
    # for n in range(0, likelihoods.shape[0]): ## 0:1
    # for l in range(0, likelihoods.shape[1]): ## 0:26047
    ##class_sum_log_likelihood[m, n] += single_feat_log_likelihood((XTest[m, l]), (likelihoods[n, l]))

    # Vectorized implementation, way much faster
    for m in range(0, XTest.shape[0]):  # 0:144
        class_likelihood[[m], ] = B_multiple_feat_log_likelihood((XTest[[m], ]), likelihoods)

    return class_likelihood


# for every test instance, you compute the likelihood or p(xi|yk) that instance i belongs to
# class k by summing the log of each feature likelihood then you multiply with by the prior
# of this specific class
# then take the max among all the classes and predict this class for this instance
def B_NBClassify(XTrain, yTrain, XTest):

    # Estimate prior
    prior_y_train = NBYPrior(yTrain)

    # Estimate class conditional means and conditional variances
    likelihoods = B_NBXGivenY(XTrain, yTrain)

    # Compute the likelihood of the features conditional on each class
    class_likelihood = np.zeros([XTest.shape[0], likelihoods.shape[0]])

    # Non-vectorized, pretty slow
    # for m in range(0, XTest.shape[0]): ## 0:144
    # for n in range(0, likelihoods.shape[0]): ## 0:1
    # for l in range(0, likelihoods.shape[1]): ## 0:26047
    ##class_sum_log_likelihood[m, n] += single_feat_log_likelihood((XTest[m, l]), (likelihoods[n, l]))

    # Vectorized implementation, way much faster
    for m in range(0, XTest.shape[0]):  # 0:144
        class_likelihood[[m], ] = B_multiple_feat_log_likelihood(
            (XTest[[m], ]), likelihoods)

    # Multiply with priors
    for n in range(class_likelihood.shape[1]):
        class_likelihood[:n] *= prior_y_train[n]

    # Obtain and return predictions
    ypred = np.argmax(class_likelihood, axis=1)
    return ypred


def B_NBgetcond(XTrain, yTrain, XTest):

    # Estimate class conditional means and conditional variances
    likelihoods = B_NBXGivenY(XTrain, yTrain)

    # Compute the likelihood of the features conditional on each class
    class_likelihood = np.zeros([XTest.shape[0], likelihoods.shape[0]])

    # Non-vectorized, pretty slow
    # for m in range(0, XTest.shape[0]): ## 0:144
    # for n in range(0, likelihoods.shape[0]): ## 0:1
    # for l in range(0, likelihoods.shape[1]): ## 0:26047
    ##class_sum_log_likelihood[m, n] += single_feat_log_likelihood((XTest[m, l]), (likelihoods[n, l]))

    # Vectorized implementation, way much faster
    for m in range(0, XTest.shape[0]):  # 0:144
        class_likelihood[[m], ] = B_multiple_feat_log_likelihood(
            (XTest[[m], ]), likelihoods)

    return class_likelihood


# for every test instance, you compute the likelihood or p(xi|yk) that instance i belongs to
# class k by summing the log of each feature likelihood then you multiply with by the prior
# of this specific class
# then take the max among all the classes and predict this class for this instance
def B_LSNBClassify(XTrain, yTrain, XTest):

    # Estimate prior
    prior_y_train = NBYPrior(yTrain)

    # Estimate class conditional means and conditional variances
    likelihoods = B_LSNBXGivenY(XTrain, yTrain)

    # Compute the likelihood of the features conditional on each class
    class_likelihood = np.zeros([XTest.shape[0], likelihoods.shape[0]])

    # Non-vectorized, pretty slow
    # for m in range(0, XTest.shape[0]): ## 0:144
    # for n in range(0, likelihoods.shape[0]): ## 0:1
    # for l in range(0, likelihoods.shape[1]): ## 0:26047
    ##class_sum_log_likelihood[m, n] += single_feat_log_likelihood((XTest[m, l]), (likelihoods[n, l]))

    # Vectorized implementation, way much faster
    for m in range(0, XTest.shape[0]):  # 0:144
        class_likelihood[[m], ] = B_multiple_feat_log_likelihood(
            (XTest[[m], ]), likelihoods)

    # Multiply with priors
    for n in range(class_likelihood.shape[1]):
        class_likelihood[:n] *= prior_y_train[n]

    # Obtain and return predictions
    ypred = np.argmax(class_likelihood, axis=1)
    return ypred
