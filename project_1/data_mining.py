import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from scipy import stats
import numpy as np
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt

# constants
random_forest = "random_forest"
naive_bayes = "naive_bayes"
n_estimators = 50


def data_mining():
    """" 
    In this code, I build 4 models, a hierarchical model and a flat model on Random Forest and Naive Bayes to learn to
    predict labeled sub-trajectories' transportation methods.
    
    For each model: I split the data up into 10 stratified folds and performed classification on each fold.
    I do this for random forest (flat vs. hierarchical) and naive bayes (flat vs. hierarchical)
    I output a confusion matrix for all 4 and compare the accuracies with a paired t-test.
    
    I proposed a 3 model approach: 
        model 1: train vs. walk vs. (bus, car, subway, taxi)
        model 2: bus vs. subway vs. (car, taxi)
        model 3: car vs. taxi
    """
    df = pd.read_csv('traj_samples_v3.csv')

    # remove run and motorcycle classes
    df = df[df.transportation_mode != 'run']
    df = df[df.transportation_mode != 'motorcycle']
    # shuffle data
    df = df.sample(frac=1, random_state=3)

    # create numpy array of x and y
    df_all_x = df.iloc[:, 0:20]
    arr_x = np.array(df_all_x)
    df_y = df["transportation_mode"].astype('category')
    arr_y = binarize(np.array(df_y))

    skf = StratifiedKFold(n_splits=10)

    conf_mat_flat_rf = []
    conf_mat_flat_nb = []
    conf_mat_hi_rf = []
    conf_mat_hi_nb = []

    accuracies_flat_rf = []
    accuracies_flat_nb = []
    accuracies_hi_rf = []
    accuracies_hi_nb = []

    # for each stratified split:
    #   get training and testing fold and then create a flat_structure and hierarchical structure for RF and NB.
    # report confusion matrix and t-test results
    for train_idx, test_idx in skf.split(arr_x, arr_y):
        x_train, y_train = arr_x[train_idx], arr_y[train_idx]
        x_test, y_test = arr_x[test_idx], arr_y[test_idx]
        df_train, df_test = df.iloc[train_idx], df.iloc[test_idx]
        single_conf_mat_flat_rf, accuracy_flat_rf = flat_structure_algorithm(random_forest, x_train, y_train, x_test, y_test)
        single_conf_mat_flat_nb, accuracy_flat_nb = flat_structure_algorithm(naive_bayes, x_train, y_train, x_test, y_test)
        single_conf_mat_hi_rf, accuracy_hi_rf = hierarchical_structure_algorithm(random_forest, df_train, df_test)
        single_conf_mat_hi_nb, accuracy_hi_nb = hierarchical_structure_algorithm(naive_bayes, df_train, df_test)

        # appending results
        accuracies_flat_rf.append(accuracy_flat_rf)
        accuracies_flat_nb.append(accuracy_flat_nb)
        accuracies_hi_rf.append(accuracy_hi_rf)
        accuracies_hi_nb.append(accuracy_hi_nb)
        # appending results
        if conf_mat_flat_rf == []:
            conf_mat_flat_rf = single_conf_mat_flat_rf
        else:
            conf_mat_flat_rf += single_conf_mat_flat_rf
        if conf_mat_flat_nb == []:
            conf_mat_flat_nb = single_conf_mat_flat_nb
        else:
            conf_mat_flat_nb += single_conf_mat_flat_nb
        if conf_mat_hi_rf == []:
            conf_mat_hi_rf = single_conf_mat_hi_rf
        else:
            conf_mat_hi_rf += single_conf_mat_hi_rf
        if conf_mat_hi_nb == []:
            conf_mat_hi_nb = single_conf_mat_hi_nb
        else:
            conf_mat_hi_nb += single_conf_mat_hi_nb

    print("Compare hierarchical vs flat structure using Random Forest: ")
    print("Confusion matrix: Flat Random Forest")
    print(conf_mat_flat_rf)
    print("Confusion matrix: Hierarchical Random Forest")
    print(conf_mat_hi_rf)
    print("Accuracies: flat: " + str(np.mean(accuracies_flat_rf)))
    print("Accuracies: hierarchical: " + str(np.mean(accuracies_hi_rf)))
    paired_t_test(accuracies_flat_rf, accuracies_hi_rf, "flat RF", "hierarchical RF")
    print("")

    print("Compare hierarchical vs flat structure using Naive Bayes: ")
    print("Confusion matrix: Flat Naive Bayes")
    print(conf_mat_flat_nb)
    print("Confusion matrix: Hierarchical Naive Bayes")
    print(conf_mat_hi_nb)
    print("Accuracies: flat: " + str(np.mean(accuracies_flat_nb)))
    print("Accuracies: hierarchical: " + str(np.mean(accuracies_hi_nb)))
    paired_t_test(accuracies_flat_nb, accuracies_hi_nb, "flat NB", "hierarchical NB")

    mpl_fig = plt.figure()
    ax = mpl_fig.add_subplot(111)
    ax.set_xlabel("Algorithms")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracies from 10-fold stratified CV on 4 models")
    plt.boxplot([accuracies_flat_nb, accuracies_hi_nb, accuracies_flat_rf, accuracies_hi_rf])
    plt.xticks([1, 2, 3, 4], ['Flat Naive Bayes', 'Hier. Naive Bayes', 'Flat Random Forest', 'Hier. Random Forest'])
    plt.show()


def flat_structure_algorithm(algo_type, x_train, y_train, x_test, y_test):
    """ trains flat model on a fold of training and runs on fold of test."""
    if algo_type == random_forest:
        model_all = RandomForestClassifier(n_estimators=n_estimators)
    elif algo_type == naive_bayes:
        model_all = GaussianNB()
    else:
        raise Exception("Model not found!")
    model_all.fit(x_train, y_train)
    y_pred_on_testing = model_all.predict(x_test)
    return confusion_matrix(y_test, y_pred_on_testing), accuracy_score(y_test, y_pred_on_testing)


def hierarchical_structure_algorithm(algo_type, df_train, df_test):
    """ Train three models on training fold and test on test fold. 
    1. Separate data into respective models
    1b. Remove labeled data that isn't part of model.
    2. binarize data properly
    3. train models
    4. classify test data on 3 models using hierarchical evaluation
    """
    df_train_x = df_train.iloc[:, 0:20]
    arr_train_x = np.array(df_train_x)
    df_train_y = df_train["transportation_mode"].astype('category')
    y_T_W_SV = binarize_T_W_SV(np.array(df_train_y))
    if algo_type == random_forest:
        model_T_W_SV = RandomForestClassifier(n_estimators=n_estimators)
    elif algo_type == naive_bayes:
        model_T_W_SV = GaussianNB()
    else:
        raise Exception("Model not found!")
    model_T_W_SV.fit(arr_train_x, y_T_W_SV)

    # remove train and walk from original y and x
    SV_df = df_train.ix[df_train['transportation_mode'] != 'walk']
    SV_df = SV_df.ix[SV_df['transportation_mode'] != 'train']
    arr_x_SV = SV_df.iloc[:, 0:20]
    arr_x_SV = np.array(arr_x_SV)
    y_B_S_SSV = SV_df["transportation_mode"].astype('category')
    # create y class with bus, subway and small_street_vehicle
    y_B_S_SSV = binarize_B_S_SSV(np.array(y_B_S_SSV))
    # train algorithm on three classes
    if algo_type == random_forest:
        model_B_S_SSV = RandomForestClassifier(n_estimators=n_estimators)
    elif algo_type == naive_bayes:
        model_B_S_SSV = GaussianNB()
    else:
        raise Exception("Model not found!")
    model_B_S_SSV.fit(arr_x_SV, y_B_S_SSV)

    # remove walk train subway and bus from original y and x
    SSV_df = df_train.ix[df_train['transportation_mode'] != 'walk']
    SSV_df = SSV_df.ix[SSV_df['transportation_mode'] != 'train']
    SSV_df = SSV_df.ix[SSV_df['transportation_mode'] != 'subway']
    SSV_df = SSV_df.ix[SSV_df['transportation_mode'] != 'bus']
    x_SSV = SSV_df.iloc[:, 0:20]
    x_SSV = np.array(x_SSV)
    y_SSV = SSV_df["transportation_mode"].astype('category')
    y_SSV = binarize_C_T(np.array(y_SSV))

    # train algorithm on two classes
    if algo_type == random_forest:
        model_C_T = RandomForestClassifier(n_estimators=n_estimators)
    elif algo_type == naive_bayes:
        model_C_T = GaussianNB()
    else:
        raise Exception("Model not found!")

    model_C_T.fit(x_SSV, y_SSV)
    df_test_x = df_test.iloc[:, 0:20]
    test_x = np.array(df_test_x)

    # classify test data
    hierarchical_predictions = get_hierarchical_predictions(test_x, model_T_W_SV, model_B_S_SSV, model_C_T)
    df_test_y = df_test["transportation_mode"].astype('category')
    test_y = binarize(np.array(df_test_y))
    return confusion_matrix(test_y, hierarchical_predictions), accuracy_score(test_y, hierarchical_predictions)


def get_hierarchical_predictions(all_x, model_T_W_SV, model_B_S_SSV, model_C_T):
    """ Given a set of unlabeled data, make a prediction using the hierarchical model
    First checks model (Train, Walk, Street Vehicle), if model predicts Street Vehicle then the next model is used
    to predict the data into (Bus, Subway, or Small Street Vehicle). If the model predicts Small Street Vehicle, then
    the next model predicts either (Tax, or Car). """
    final_predictions = []
    for i in range(len(all_x)):
        pred_y_T_W_SV = model_T_W_SV.predict(all_x[i].reshape(1, -1))
        pred_y_T_W_SV = pred_y_T_W_SV[0]
        pred = -1
        if pred_y_T_W_SV == 0:
            pred = 4
        elif pred_y_T_W_SV == 1:
            pred = 5
        elif pred_y_T_W_SV == 2:
            pred_y_B_S_SSV = model_B_S_SSV.predict(all_x[i].reshape(1, -1))
            pred_y_B_S_SSV = pred_y_B_S_SSV[0]
            if pred_y_B_S_SSV == 0:
                pred = 0
            elif pred_y_B_S_SSV == 1:
                pred = 3
            elif pred_y_B_S_SSV == 2:
                pred_y_C_T = model_C_T.predict(all_x[i].reshape(1, -1))
                pred_y_C_T = pred_y_C_T[0]
                if pred_y_C_T == 0:
                    pred = 2
                elif pred_y_C_T == 1:
                    pred = 1
        if pred == -1:
            raise Exception("something went wrong with prediction!")
        final_predictions.append(pred)
    return np.array(final_predictions)


def binarize_C_T(labels):
    """ binarizes the data given tax or car into labels 0 or 1 (taxi, car)"""
    new_Y = np.zeros(labels.shape)
    for x in range(labels.shape[0]):
        if labels[x] == "bus":
            raise Exception("bus should not be in this data!")
        elif labels[x] == "taxi":
            new_Y[x] = 1
        elif labels[x] == "car":
            new_Y[x] = 0
        elif labels[x] == "subway":
            raise Exception("subway should not be in this data!")
        elif labels[x] == "train":
            raise Exception("train should not be in this data!")
        elif labels[x] == "walk":
            raise Exception("walk should not be in this data!")
    return new_Y


def binarize_B_S_SSV(labels):
    """ binarizes the data given bus, taxi, car or subway and label 0,1,2 (Bus, Taxi, Small Vehicle)"""
    new_Y = np.zeros(labels.shape)
    for x in range(labels.shape[0]):
        if labels[x] == "bus":
            new_Y[x] = 0
        elif labels[x] == "taxi":
            new_Y[x] = 2
        elif labels[x] == "car":
            new_Y[x] = 2
        elif labels[x] == "subway":
            new_Y[x] = 1
        elif labels[x] == "train":
            raise Exception("train should not be in this data!")
        elif labels[x] == "walk":
            raise Exception("walk should not be in this data!")
    return new_Y


def binarize_T_W_SV(labels):
    """ binarizes the data given walk, train, subway, car, taxi, bus into label 0,1,2 (Train, Walk, Street Vehicle)"""
    new_Y = np.zeros(labels.shape)
    for x in range(labels.shape[0]):
        if labels[x] == "bus":
            new_Y[x] = 2
        elif labels[x] == "taxi":
            new_Y[x] = 2
        elif labels[x] == "car":
            new_Y[x] = 2
        elif labels[x] == "subway":
            new_Y[x] = 2
        elif labels[x] == "train":
            new_Y[x] = 0
        elif labels[x] == "walk":
            new_Y[x] = 1
    return new_Y


def binarize(labels):
    """ binarizes the six labels into 0-5"""
    new_Y = np.zeros(labels.shape)
    for x in range(labels.shape[0]):
        if labels[x] == "bus":
            new_Y[x] = 0
        elif labels[x] == "taxi":
            new_Y[x] = 1
        elif labels[x] == "car":
            new_Y[x] = 2
        elif labels[x] == "subway":
            new_Y[x] = 3
        elif labels[x] == "train":
            new_Y[x] = 4
        elif labels[x] == "walk":
            new_Y[x] = 5
    return new_Y


def paired_t_test(score1, score2, name1, name2):
    """ paired student's t-test """
    t_stats = stats.ttest_rel(score1, score2)
    print(t_stats)
    if t_stats[1] < 0.05:
        if score1 > score2:
            print("Rejected null hypothesis! p<0.05. Using " + name1 + " was statistically better than " + name2)
        else:
            print("Rejected null hypothesis! p<0.05. Using " + name2 + " was statistically better than " + name1)
    else:
        print("p >= 0.05. Can't say anything about " + name1 + " vs. " + name2)
    print("")

data_mining()
