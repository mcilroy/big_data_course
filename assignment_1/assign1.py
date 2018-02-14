from sklearn import linear_model
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np


def main():
    """
    Assignment code. Performs the work specified in questions a, b and c. Most of the code is self explanatory.
    """
    df = pd.read_csv('animals.csv')
    # shuffle data
    df = df.sample(frac=1, random_state=3)
    X = df.iloc[:, 0:25]
    Y = df["class"].astype('category')
    X = np.array(X)
    Y = np.array(Y)

    # QUESTION A)
    # prepare data
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=42)
    y_train_elk, y_train_deer, y_train_cattle = binarize(y_train, "ELK"), binarize(y_train, "DEER"), binarize(y_train, "CATTLE")
    y_test_elk, y_test_deer, y_test_cattle = binarize(y_test, "ELK"), binarize(y_test, "DEER"), binarize(y_test, "CATTLE")

    # Logistic Regression
    logistic_regression = linear_model.LogisticRegression()
    print("logistic regression")
    calc_and_show_scores(logistic_regression, X_train, X_test, y_train_elk, y_train_deer, y_train_cattle, y_test_elk, y_test_deer, y_test_cattle)

    # Naive Bayes
    NB = GaussianNB()
    print("Naive Bayes")
    calc_and_show_scores(NB, X_train, X_test, y_train_elk, y_train_deer, y_train_cattle, y_test_elk, y_test_deer, y_test_cattle)

    # Decision Tree
    dec_tree = tree.DecisionTreeClassifier()
    print("Decision Tree")
    calc_and_show_scores(dec_tree, X_train, X_test, y_train_elk, y_train_deer, y_train_cattle, y_test_elk, y_test_deer, y_test_cattle)

    # Random Forest
    random_forest = RandomForestClassifier()
    print("Random Forest")
    calc_and_show_scores(random_forest, X_train, X_test, y_train_elk, y_train_deer, y_train_cattle, y_test_elk, y_test_deer, y_test_cattle)
    print("")

    # QUESTION B)
    print("10-fold Cross validation. Per instructions use all data")
    y_all_elk, y_all_deer, y_all_cattle = binarize(Y, "ELK"), binarize(Y, "DEER"), binarize(Y, "CATTLE")
    scores_logistic_regression = test_cv(logistic_regression, X, y_all_elk, y_all_deer, y_all_cattle)
    scores_NB = test_cv(NB, X, y_all_elk, y_all_deer, y_all_cattle)
    scores_dec_tree = test_cv(dec_tree, X, y_all_elk, y_all_deer, y_all_cattle)
    scores_random_forest = test_cv(random_forest, X, y_all_elk, y_all_deer, y_all_cattle)
    print("")
    # compare random forest with the rest
    paired_t_test(scores_random_forest, scores_logistic_regression, "random forest", "logistic regression")
    paired_t_test(scores_random_forest, scores_NB, "random forest", "Naive Bayes")
    paired_t_test(scores_random_forest, scores_dec_tree, "random forest", "decision tree")

    # display figure
    mpl_fig = plt.figure()
    ax = mpl_fig.add_subplot(111)
    ax.set_xlabel("Algorithms")
    ax.set_ylabel("Accuracy")
    ax.set_title("30 accuracies from 10-fold CV on three different classes")
    plt.boxplot([scores_logistic_regression, scores_NB, scores_dec_tree, scores_random_forest])
    plt.xticks([1, 2, 3, 4], ['logistic regression', 'naive Bayes', 'decision tree', 'random forest (default)'])
    print("")

    # QUESTION C)
    print("compare random forest n_estimators")
    rf_scores = []
    for x in [10, 20, 50, 100]:
        random_forest = RandomForestClassifier(n_estimators=x)
        print(str(x) + " of trees")
        rf_scores.append(test_cv(random_forest, X, y_all_elk, y_all_deer, y_all_cattle))
    best_rf_100 = rf_scores[3]

    # display figure comparing random forests
    mpl3_fig = plt.figure()
    ax = mpl3_fig.add_subplot(111)
    ax.set_xlabel("Algorithms")
    ax.set_ylabel("Accuracy")
    ax.set_title("30 accuracies from 10-fold CV on three different classes")
    plt.boxplot([rf_scores[0], rf_scores[1], rf_scores[2], rf_scores[3]])
    plt.xticks([1, 2, 3, 4], ['random forest sized 10', 'random forest sized 20', 'random forest sized 50', 'random forest sized 100'])
    paired_t_test(best_rf_100, rf_scores[0], "random forest sized 100", "random forest sized 10")
    paired_t_test(best_rf_100, rf_scores[1], "random forest sized 100", "random forest sized 20")
    paired_t_test(best_rf_100, rf_scores[2], "random forest sized 100", "random forest sized 50")

    # display figure comparing 100 tree random forest with other algorithms
    mpl2_fig = plt.figure()
    ax = mpl2_fig.add_subplot(111)
    ax.set_xlabel("Algorithms")
    ax.set_ylabel("Accuracy")
    ax.set_title("30 accuracies from 10-fold CV on three different classes")
    plt.boxplot([scores_logistic_regression, scores_NB, scores_dec_tree, best_rf_100])
    plt.xticks([1, 2, 3, 4], ['logistic regression', 'naive Bayes', 'decision tree', 'random forest 100 trees'])

    # compare random forest 100 vs. others using paired_t_test
    paired_t_test(best_rf_100, scores_logistic_regression, "random forest sized 100", "logistic regression")
    paired_t_test(best_rf_100, scores_NB, "random forest sized 100", "Naive Bayes")
    paired_t_test(best_rf_100, scores_dec_tree, "random forest sized 100", "decision tree")
    plt.show()


def paired_t_test(score1, score2, name1, name2):
    """ paired student's t-test """
    t_stats = stats.ttest_rel(score1, score2)
    print(t_stats)
    if t_stats[1] < 0.05:
        print("Rejected null hypothesis! p<0.05. Using " + name1 + " was statistically better than " + name2)
    else:
        print("p >= 0.05. Can't say anything about " + name1 + " vs. " + name2)
    print("")


def binarize(labels, name):
    """ binarizes the data"""
    new_Y = np.zeros(labels.shape)
    for x in range(labels.shape[0]):
        if labels[x] == name:
            new_Y[x] = 1
        else:
            new_Y[x] = 0
    return new_Y


def test_cv(algo, X, y_all_elk, y_all_deer, y_all_cattle):
    """ perform cross validation on the three classes and returns the 30 accuracy values """
    scores_elk = cv(algo, X, y_all_elk)
    scores_deer = cv(algo, X, y_all_deer)
    scores_cattle = cv(algo, X, y_all_cattle)
    all_scores = np.hstack((scores_cattle, scores_deer, scores_elk))
    print("mean of 30 values from 3 one-vs-all models: " + str(np.mean(all_scores)))
    print("standard deviation of 30 values from 3 one-vs-all models: " + str(all_scores.std()))
    return all_scores


def cv(algo, X, y):
    """ cross validation"""
    scores = cross_val_score(algo, X, y, cv=10)
    print("One-vs-all: 10-fold CV Accuracy: %0.2f Std. Dev.: %0.2f, 95%% within (+/- %0.2f)" % (scores.mean(), scores.std(), scores.std() * 2))
    print("")
    return scores


def calc_and_show_scores(algo, X_train, X_test, y_train_elk, y_train_deer, y_train_cattle, y_test_elk, y_test_deer, y_test_cattle):
    """  train 3 models on each set of one-vs-all data, print averages. """
    train_accuracy_elk, test_accuracy_elk = run_algorithm(algo, X_train, y_train_elk, X_test, y_test_elk)
    train_accuracy_deer, test_accuracy_deer = run_algorithm(algo, X_train, y_train_deer, X_test, y_test_deer)
    train_accuracy_cattle, test_accuracy_cattle = run_algorithm(algo, X_train, y_train_cattle, X_test, y_test_cattle)
    print("mean of scores from each class for train data: " + str(np.mean([train_accuracy_elk, train_accuracy_deer, train_accuracy_cattle])))
    print("mean of scores from each class for testing data: " + str(np.mean([test_accuracy_elk, test_accuracy_deer, test_accuracy_cattle])))


def run_algorithm(algo, X_train, y_train, X_test, y_test):
    """ fit a model, print confusion matrix and return the accuracy score """
    algo.fit(X_train, y_train)
    print("training confusion matrix: ")
    y_pred_on_training = algo.predict(X_train)
    print(confusion_matrix(y_train, y_pred_on_training))
    print("training accuracy: " + str(accuracy_score(y_train, y_pred_on_training)))
    y_pred_on_testing = algo.predict(X_test)
    print("testing confusion matrix: ")
    print(confusion_matrix(y_test, y_pred_on_testing))
    print("testing accuracy score: " + str(accuracy_score(y_test, y_pred_on_testing)))
    print("")
    return accuracy_score(y_train, y_pred_on_training), accuracy_score(y_test, y_pred_on_testing)

main()
