"""
Author      : Yi-Chieh Wu, Sriram Sankararaman
Description : Titanic
"""

# Use only the provided packages!
import math
import csv
from util import *
from collections import Counter

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import train_test_split
from sklearn import metrics

######################################################################
# classes
######################################################################

class Classifier(object) :
    """
    Classifier interface.
    """
    
    def fit(self, X, y):
        raise NotImplementedError()
        
    def predict(self, X):
        raise NotImplementedError()


class MajorityVoteClassifier(Classifier) :
    
    def __init__(self) :
        """
        A classifier that always predicts the majority class.
        
        Attributes
        --------------------
            prediction_ -- majority class
        """
        self.prediction_ = None
    
    def fit(self, X, y) :
        """
        Build a majority vote classifier from the training set (X, y).
        
        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
            y    -- numpy array of shape (n,), target classes
        
        Returns
        --------------------
            self -- an instance of self
        """
        majority_val = Counter(y).most_common(1)[0][0]
        self.prediction_ = majority_val
        return self
    
    def predict(self, X) :
        """
        Predict class values.
        
        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
        
        Returns
        --------------------
            y    -- numpy array of shape (n,), predicted classes
        """
        if self.prediction_ is None :
            raise Exception("Classifier not initialized. Perform a fit first.")
        
        n,d = X.shape
        y = [self.prediction_] * n 
        return y


class RandomClassifier(Classifier) :
    
    def __init__(self) :
        """
        A classifier that predicts according to the distribution of the classes.
        
        Attributes
        --------------------
            probabilities_ -- class distribution dict (key = class, val = probability of class)
        """
        self.probabilities_ = None
    
    def fit(self, X, y) :
        """
        Build a random classifier from the training set (X, y).
        
        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
            y    -- numpy array of shape (n,), target classes
        
        Returns
        --------------------
            self -- an instance of self
        """
        
        ### ========== TODO : START ========== ###
        # part b: set self.probabilities_ according to the training set
        most_common_2 = Counter(y).most_common(2)
        maj_feature = most_common_2[0][0]
        maj_num = most_common_2[0][1]
        min_feature = most_common_2[1][0]
        min_num = most_common_2[1][1]
        total = maj_num + min_num
        prob = {}
        prob[maj_feature] = float(maj_num)/float(total)
        prob[min_feature] = float(min_num)/float(total)
        self.probabilities_ = prob
        ### ========== TODO : END ========== ###
        
        return self
    
    def predict(self, X, seed=1234) :
        """
        Predict class values.
        
        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
            seed -- integer, random seed
        
        Returns
        --------------------
            y    -- numpy array of shape (n,), predicted classes
        """
        if self.probabilities_ is None :
            raise Exception("Classifier not initialized. Perform a fit first.")
        np.random.seed(seed)
        
        ### ========== TODO : START ========== ###
        # part b: predict the class for each test example
        # hint: use np.random.choice (be careful of the parameters)
        y = np.random.choice([0,1], X.shape[0], p = [self.probabilities_[0],self.probabilities_[1]])
        ### ========== TODO : END ========== ###
        
        return y


######################################################################
# functions
######################################################################
def plot_histograms(X, y, Xnames, yname) :
    n,d = X.shape  # n = number of examples, d =  number of features
    fig = plt.figure(figsize=(20,15))
    nrow = 3; ncol = 3
    for i in range(d) :
        fig.add_subplot (3,3,i)  
        data, bins, align, labels = plot_histogram(X[:,i], y, Xname=Xnames[i], yname=yname, show = False)
        n, bins, patches = plt.hist(data, bins=bins, align=align, alpha=0.5, label=labels)
        plt.xlabel(Xnames[i])
        plt.ylabel('Frequency')
        plt.legend() #plt.legend(loc='upper left')
 
    plt.savefig ('histograms.pdf')


def plot_histogram(X, y, Xname, yname, show = True) :
    """
    Plots histogram of values in X grouped by y.
    
    Parameters
    --------------------
        X     -- numpy array of shape (n,d), feature values
        y     -- numpy array of shape (n,), target classes
        Xname -- string, name of feature
        yname -- string, name of target
    """
    
    # set up data for plotting
    targets = sorted(set(y))
    data = []; labels = []
    for target in targets :
        features = [X[i] for i in range(len(y)) if y[i] == target]
        data.append(features)
        labels.append('%s = %s' % (yname, target))
    
    # set up histogram bins
    features = set(X)
    nfeatures = len(features)
    test_range = list(range(int(math.floor(min(features))), int(math.ceil(max(features)))+1))
    if nfeatures < 10 and sorted(features) == test_range:
        bins = test_range + [test_range[-1] + 1] # add last bin
        align = 'left'
    else :
        bins = 10
        align = 'mid'
    
    # plot
    if show == True:
        plt.figure()
        n, bins, patches = plt.hist(data, bins=bins, align=align, alpha=0.5, label=labels)
        plt.xlabel(Xname)
        plt.ylabel('Frequency')
        plt.legend() #plt.legend(loc='upper left')
        plt.show()

    return data, bins, align, labels


def error(clf, X, y, ntrials=100, test_size=0.2) :
    """
    Computes the classifier error over a random split of the data,
    averaged over ntrials runs.
    
    Parameters
    --------------------
        clf         -- classifier
        X           -- numpy array of shape (n,d), features values
        y           -- numpy array of shape (n,), target classes
        ntrials     -- integer, number of trials
    
    Returns
    --------------------
        train_error -- float, training error
        test_error  -- float, test error
    """
    
    ### ========== TODO : START ========== ###
    # part e: compute cross-validation error over ntrials
    # hint: use train_test_split (be careful of the parameters)
    train_error = 0
    test_error = 0
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = ntrials)
    for i in range (1, 101):
        clf.fit(x_train, y_train)
        y_predTrain = clf.predict(x_train)
        y_predTest = clf.predict(x_test)
        train_error += 1 - metrics.accuracy_score(y_train, y_predTrain, normalize = True)
        test_error += 1 - metrics.accuracy_score(y_test, y_predTest, normalize = True)
    train_error = train_error * 0.01
    test_error = test_error * 0.01
    ### ========== TODO : END ========== ###
    
    return train_error, test_error


def write_predictions(y_pred, filename, yname=None) :
    """Write out predictions to csv file."""
    out = open(filename, 'wb')
    f = csv.writer(out)
    if yname :
        f.writerow([yname])
    f.writerows(list(zip(y_pred)))
    out.close()


######################################################################
# main
######################################################################

def main():
    # load Titanic dataset
    titanic = load_data("titanic_train.csv", header=1, predict_col=0)
    X = titanic.X; Xnames = titanic.Xnames
    y = titanic.y; yname = titanic.yname
    n,d = X.shape  # n = number of examples, d =  number of features
    
    
    
    #========================================
    # part a: plot histograms of each feature
    print('Plotting...')
    for i in range(d) :
        plot_histogram(X[:,i], y, Xname=Xnames[i], yname=yname)
    #========================================
    # train Majority Vote classifier on data
    print('Classifying using Majority Vote...')
    clf = MajorityVoteClassifier() # create MajorityVote classifier, which includes all model parameters
    clf.fit(X, y)                  # fit training data using the classifier
    y_pred = clf.predict(X)        # take the classifier and run it on the training data
    train_error = 1 - metrics.accuracy_score(y, y_pred, normalize=True)
    print('\t-- training error: %.3f' % train_error)
    
    
    
    ### ========== TODO : START ========== ###
    # part b: evaluate training error of Random classifier
    print('Classifying using Random...')
    clfRandom = RandomClassifier()
    clfRandom.fit(X,y)
    y_pred = clfRandom.predict(X)
    train_error = 1 - metrics.accuracy_score(y, y_pred, normalize = True)
    print('\t-- training error: %.3f' % train_error)
    ### ========== TODO : END ========== ###
    
    
    
    ### ========== TODO : START ========== ###
    # part c: evaluate training error of Decision Tree classifier
    # use criterion of "entropy" for Information gain 
    print('Classifying using Decision Tree...')
    clfDecisionTree = DecisionTreeClassifier(criterion = 'entropy')
    clfDecisionTree.fit(X,y)
    y_pred = clfDecisionTree.predict(X)
    train_error = 1 - metrics.accuracy_score(y, y_pred, normalize = True)
    print('\t-- training error: %.3f' % train_error)
    ### ========== TODO : END ========== ###

    

    # note: uncomment out the following lines to output the Decision Tree graph
    """
    # save the classifier -- requires GraphViz and pydot
    import StringIO, pydot
    from sklearn import tree
    dot_data = StringIO.StringIO()
    tree.export_graphviz(clf, out_file=dot_data,
                         feature_names=Xnames)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf("dtree.pdf") 
    """



    ### ========== TODO : START ========== ###
    # part d: evaluate training error of k-Nearest Neighbors classifier
    # use k = 3, 5, 7 for n_neighbors 
    print('Classifying using k-Nearest Neighbors...')
    clf3KNeighbors = KNeighborsClassifier(n_neighbors=3)
    clf3KNeighbors.fit(X,y)
    y_pred = clf3KNeighbors.predict(X)
    train_error = 1 - metrics.accuracy_score(y, y_pred, normalize = True)
    print('\t-- training error when k = 3: %.3f' % train_error)
    
    clf5KNeighbors = KNeighborsClassifier(n_neighbors=5)
    clf5KNeighbors.fit(X,y)
    y_pred = clf5KNeighbors.predict(X)
    train_error = 1 - metrics.accuracy_score(y, y_pred, normalize = True)
    print('\t-- training error when k = 5: %.3f' % train_error)
    
    clf7KNeighbors = KNeighborsClassifier(n_neighbors=7)
    clf7KNeighbors.fit(X,y)
    y_pred = clf7KNeighbors.predict(X)
    train_error = 1 - metrics.accuracy_score(y, y_pred, normalize = True)
    print('\t-- training error when k = 7: %.3f' % train_error)
    ### ========== TODO : END ========== ###
    
    
    
    ### ========== TODO : START ========== ###
    # part e: use cross-validation to compute average training and test error of classifiers
    print('Investigating various classifiers...')
    train_error, test_error = error(clf, X, y, ntrials = 100, test_size = 0.2)
    print('\t-- training error for MajorityVoteClassifier: %.3f' % train_error)
    print('\t-- test error for MajorityVoteClassifier: %.3f' % test_error)
    
    train_error, test_error = error(clfRandom, X, y, ntrials = 100, test_size = 0.2)
    print('\t-- training error for RandomClassifier: %.3f' % train_error)
    print('\t-- test error for RandomClassifier: %.3f' % test_error)
    
    train_error, test_error = error(clfDecisionTree, X, y, ntrials = 100, test_size = 0.2)
    print('\t-- training error for DecisionTreeClassifier: %.3f' % train_error)
    print('\t-- test error for DecisionTreeClassifier: %.3f' % test_error)
    
    train_error, test_error = error(clf5KNeighbors, X, y, ntrials = 100, test_size = 0.2)
    print('\t-- training error for KNeighborsClassifier: %.3f' % train_error)
    print('\t-- test error for KNeighborsClassifier: %.3f' % test_error)
    ### ========== TODO : END ========== ###



    ### ========== TODO : START ========== ###
    # part f: use 10-fold cross-validation to find the best value of k for k-Nearest Neighbors classifier
    print('Finding the best k for KNeighbors classifier...')
    KNeighbors = []
    for i in range(1, 50, 2):
        KNeighbors.append(i)
    validation_error = []
    for k in KNeighbors:
        clfKNeighbors = KNeighborsClassifier(n_neighbors = k)
        validation_error_point = 1 - cross_val_score(clfKNeighbors, X, y, cv = 10).mean()
        validation_error.append(validation_error_point)
    plt.plot(KNeighbors, validation_error)
    plt.xlabel('Number of neighbors, k')
    plt.ylabel('Validation error')
    plt.show()
    ### ========== TODO : END ========== ###
    
    
    
    ### ========== TODO : START ========== ###
    # part g: investigate decision tree classifier with various depths
    print('Investigating depths...')
    depths = []
    trainError = []
    testError = []
    for i in range(1,21):
        depths.append(i)
        clfTreeDepths = DecisionTreeClassifier(criterion = 'entropy', max_depth=i)
        train_error, test_error = error(clfTreeDepths, X, y, ntrials=100, test_size=0.2)
        trainError.append(train_error)
        testError.append(test_error)
    trainingLine, = plt.plot(depths,trainError,label='Training Error')
    testLine, = plt.plot(depths,testError,label='Test Error')
    plt.xticks(np.arange(0, 21, 1))
    plt.xlabel('Depth Limit')
    plt.ylabel('Validation error')
    plt.legend(handles=[trainingLine, testLine])
    plt.show()
    ### ========== TODO : END ========== ###
    
    
    
    ### ========== TODO : START ========== ###
    # part h: investigate Decision Tree and k-Nearest Neighbors classifier with various training set sizes
    print('Investigating training set sizes...')
    trainD7 = []
    testD7 = []
    trainK7 = []
    testK7 = []
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)
    clf7DTrees = DecisionTreeClassifier(criterion = 'entropy', max_depth=7)
    clf7DTrees.fit(X_train,y_train)
    clf7KNeighbors.fit(X_train,y_train)
    percentageSplit = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    for i in percentageSplit:
        trainD7points, testD7points = error(clf7DTrees, X, y, ntrials = 100, test_size = i)
        trainD7.append(trainD7points)
        testD7.append(testD7points)
        trainK7points, testK7points = error(clf7KNeighbors, X, y, ntrials = 100, test_size = i)
        trainK7.append(trainK7points)
        testK7.append(testK7points)
    D7trainLine, = plt.plot(percentageSplit, trainD7, label = 'DecisionTree training error (depth = 7)')
    D7testLine, = plt.plot(percentageSplit, testD7, label = 'DecisionTree test error (depth = 7)')
    K7trainLine, = plt.plot(percentageSplit, trainK7, label = 'KNeighbors training error (k = 7)')
    K7testLine, = plt.plot(percentageSplit, testK7, label = 'KNeighbors test error (k = 7)')
    plt.xlabel('Amount of training data (experience)')
    plt.ylabel('Error (classifier performance)')
    plt.legend(handles=[D7trainLine, D7testLine, K7trainLine, K7testLine])
    plt.show()
    ### ========== TODO : END ========== ###
    
       
    print('Done')


if __name__ == "__main__":
    main()
