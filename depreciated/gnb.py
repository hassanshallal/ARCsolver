
import numpy as np


def NBYPrior(ytrain):
    priors = np.transpose([float(x)/ytrain.shape[0]
                           for x in list(np.bincount(ytrain))])
    return priors


def G_NBXGivenY(Xtrain, ytrain):
    overall = np.column_stack((Xtrain, ytrain))
    uniques_classes = np.unique(overall[:, -1])

    cond_means = []
    cond_vars = []
    for n in range(uniques_classes.shape[0]):
        current_class = overall[overall[:, -1] == uniques_classes[n]]
        cond_means.append(current_class.mean(axis=0)[
                          0:(current_class.shape[1]-1):])
        cond_vars.append(current_class.var(axis=0)[
                         0:(current_class.shape[1]-1):])

    var_smoothing = 1e-9
    epsilon_ = var_smoothing * np.var(Xtrain, axis=0).max()
    #print("cond_vars before somoothing")
    # print(cond_vars)
    # print("epsilon_")
    # print(epsilon_)
    cond_vars += epsilon_
    #print("cond_vars after somoothing")
    # print(cond_vars)

    return np.array(cond_means), np.array(cond_vars)

#


def G_multiple_feat_log_likelihood(xi, cond_means, cond_vars):
    # xi is an instance
    num_classes = cond_means.shape[0]
    num_features = cond_means.shape[1]
    features_classes_likelihood = np.ones(num_classes)
    for i in range(0, num_classes):
        product = 1
        for j in range(0, num_features):
            product = product * (1/np.sqrt(2*np.pi*cond_vars[i][j])) * np.exp(-0.5 * pow(
                (xi.T[j] - cond_means[i][j]), 2)/cond_vars[i][j])
        features_classes_likelihood[i] = product
    return features_classes_likelihood


def G_NBgetcond(XTrain, yTrain, XTest):
    # Estimate prior
    prior_y_train = NBYPrior(yTrain)

    # Estimate class conditional means and conditional variances
    cond_means, cond_vars = G_NBXGivenY(XTrain, yTrain)

    class_likelihood = np.ones([len(XTest), cond_means.shape[0]])
    for m in range(0, len(XTest)):  # 0:49
        class_likelihood[[m], ] = G_multiple_feat_log_likelihood(
            XTest[[m], ], cond_means, cond_vars)

    return class_likelihood

# for every test instance, you compute the likelihood or p(xi|yk) that instance i belongs to
# class k by summing the log of each feature then you multiply by the prior of this specific class
# then take the max among all the classes and predict this class for this instance


def G_NBClassify(XTrain, yTrain, XTest):

    # Estimate prior
    prior_y_train = NBYPrior(yTrain)

    # Estimate class conditional means and conditional variances
    cond_means, cond_vars = G_NBXGivenY(XTrain, yTrain)

    class_likelihood = np.zeros([XTest.shape[0], cond_means.shape[0]])

    for m in range(0, len(XTest)):  # 0:49
        class_likelihood[[m], ] = G_multiple_feat_log_likelihood(
            XTest[[m], ], cond_means, cond_vars)

    # Multiply with priors
    for n in range(class_likelihood.shape[1]):
        class_likelihood[:n] *= prior_y_train[n]

    # Obtain and return predictions
    ypred = np.argmax(class_likelihood, axis=1)
    return ypred
