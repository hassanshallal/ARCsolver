# This is an attempted implementation of mixed NB
# We base this on the implementations of bnb and gnb
# get class likelihoods of test cases using B_NBgetcond, G_NBgetcond
# we multiply  class likelihoods of categorical and continious variables
# we multiply with prior
# we predict using arggmax

from bnb import *
from gnb import *


def M_NBClassify(XTrain_cat, XTrain_cont, yTrain, XTest_cat, XTest_cont):

    # Estimate prior
    prior_y_train = NBYPrior(yTrain)

    # get your sufficient statistics for both cat and cond variables
    # Estimate class conditional means and conditional variances
    likelihoods = B_NBXGivenY(XTrain_cat, yTrain)
    # Estimate class conditional means and conditional variances
    cond_means, cond_vars = G_NBXGivenY(XTrain_cont, yTrain)

    # Compute the likelihood of the features conditional on each class
    class_likelihood_cat = np.zeros([XTest_cat.shape[0], likelihoods.shape[0]])
    class_likelihood_cont = np.zeros(
        [XTest_cont.shape[0], cond_means.shape[0]])

    # get conditionals
    for m in range(0, XTest_cat.shape[0]):
        class_likelihood_cat[[m], ] = B_multiple_feat_log_likelihood(
            (XTest_cat[[m], ]), likelihoods)

    for m in range(0, XTest_cont.shape[0]):
        class_likelihood_cont[[m], ] = G_multiple_feat_log_likelihood(
            XTest_cont[[m], ], cond_means, cond_vars)

    # assert both conditional probability arrays have the same shape
    assert class_likelihood_cat.shape == class_likelihood_cont.shape

    # To avoid underflow, add the log of both cat and cont
    class_likelihood = np.log(class_likelihood_cat) + \
        np.log(class_likelihood_cont)

    # add log priors
    for n in range(class_likelihood.shape[1]):
        class_likelihood[:n] += np.log(prior_y_train[n])

    # Obtain and return predictions
    ypred = np.argmax(class_likelihood, axis=1)
    return ypred


def M_LSNBClassify(XTrain_cat, XTrain_cont, yTrain, XTest_cat, XTest_cont):

    # Estimate prior
    prior_y_train = NBYPrior(yTrain)

    # get your sufficient statistics for both cat and cond variables
    # Estimate class conditional means and conditional variances
    likelihoods = B_LSNBXGivenY(XTrain_cat, yTrain)
    # Estimate class conditional means and conditional variances
    cond_means, cond_vars = G_NBXGivenY(XTrain_cont, yTrain)

    # Compute the likelihood of the features conditional on each class
    class_likelihood_cat = np.zeros([XTest_cat.shape[0], likelihoods.shape[0]])
    class_likelihood_cont = np.zeros(
        [XTest_cont.shape[0], cond_means.shape[0]])

    # get conditionals
    for m in range(0, XTest_cat.shape[0]):
        class_likelihood_cat[[m], ] = B_multiple_feat_log_likelihood(
            (XTest_cat[[m], ]), likelihoods)

    for m in range(0, XTest_cont.shape[0]):
        class_likelihood_cont[[m], ] = G_multiple_feat_log_likelihood(
            XTest_cont[[m], ], cond_means, cond_vars)

    # assert both conditional probability arrays have the same shape
    assert class_likelihood_cat.shape == class_likelihood_cont.shape

    # To avoid underflow, add the log of both cat and cont
    class_likelihood = np.log(class_likelihood_cat) + \
        np.log(class_likelihood_cont)

    # add log priors
    for n in range(class_likelihood.shape[1]):
        class_likelihood[:n] += np.log(prior_y_train[n])

    # Obtain and return predictions
    ypred = np.argmax(class_likelihood, axis=1)
    return ypred
