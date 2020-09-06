import numpy as np
import ProblemSet.ps1.src.util as util
import matplotlib.pyplot as plt

from ProblemSet.ps1.src.p01b_logreg import LogisticRegression

# Character to replace with sub-problem letter in plot_path/pred_path
WILDCARD = 'X'


def main(train_path='/Users/leeseungjoon/realest/CS229/ProblemSet/ps1/data/ds3_train.csv',
         valid_path='/Users/leeseungjoon/realest/CS229/ProblemSet/ps1/data/ds3_valid.csv',
         test_path='/Users/leeseungjoon/realest/CS229/ProblemSet/ps1/data/ds3_train.csv',
         pred_path='/Users/leeseungjoon/realest/CS229/ProblemSet/ps1/src/output/p02X_pred.txt'):
    """Problem 2: Logistic regression for incomplete, positive-only labels.

    Run under the following conditions:
        1. on y-labels,
        2. on l-labels,
        3. on l-labels with correction factor alpha.

    Args:
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    pred_path_c = pred_path.replace(WILDCARD, 'c')
    pred_path_d = pred_path.replace(WILDCARD, 'd')
    pred_path_e = pred_path.replace(WILDCARD, 'e')

    # *** START CODE HERE ***
    # Part (c): Train and test on true labels
    # Make sure to save outputs to pred_path_c
    x_train, t_train = util.load_dataset(train_path, label_col='t', add_intercept=True)
    x_test, t_test = util.load_dataset(test_path, label_col='t', add_intercept=True)

    theta = np.zeros(x_train.shape[1])
    model_c = LogisticRegression(theta_0=theta)
    model_c.fit(x_train, t_train)

    h = model_c.predict(x_test)
    t_pred = np.where(h >= 0.5, 1, 0)
    util.plot(x_test, t_test, model_c.theta, '{}.png'.format(pred_path_c))
    np.savetxt(pred_path_c, t_pred, delimiter=',')

    # Part (d): Train on y-labels and test on true labels
    # Make sure to save outputs to pred_path_d
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_test, t_test = util.load_dataset(test_path, label_col='t', add_intercept=True)

    theta = np.zeros(x_train.shape[1])
    model_d = LogisticRegression(theta_0=theta)
    model_d.fit(x_train, y_train)

    h = model_d.predict(x_test)
    y_pred = np.where(h >= 0.5, 1, 0)
    util.plot(x_test, t_test, model_d.theta, '{}.png'.format(pred_path_d))
    np.savetxt(pred_path_d, y_pred, delimiter=',')

    # Part (e): Apply correction factor using validation set and test on true labels
    # Plot and use np.savetxt to save outputs to pred_path_e
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_val, y_val = util.load_dataset(valid_path, add_intercept=True)
    x_test, t_test = util.load_dataset(test_path, label_col='t', add_intercept=True)

    theta = np.zeros(x_train.shape[1])
    model_e = LogisticRegression(theta_0=theta)
    model_e.fit(x_train, y_train)

    h_val = model_e.predict(x_val)
    num_ones = (y_val == 1).sum()
    alpha = h_val[y_val == 1].sum() / num_ones

    h_test = model_e.predict(x_test) / alpha
    t_pred = np.where(h_test >= 0.5, 1, 0)

    correction = 1 + (np.log(2 / alpha - 1) / model_e.theta[0])  # TODO: Why??
    # TODO: Check https://github.com/zhixuan-lin/cs229-ps-2018/blob/master/ps1/src/p02cde_posonly.py
    util.plot(x_test, t_test, model_e.theta, '{}.png'.format(pred_path_e), correction=correction)  # TODO: Why this col?
    np.savetxt(pred_path_e, t_pred, delimiter=',')

    # # Plot dataset
    # x = x_test
    # t = t_test
    #
    # plt.figure()
    # plt.plot(x[t == 1, -2], x[t == 1, -1], 'bx', linewidth=1)
    # plt.plot(x[t == 0, -2], x[t == 0, -1], 'gx', linewidth=1)
    #
    # # Plot decision boundary (found by solving for theta^T x = 0)
    # x1 = np.arange(min(x[:, -2]), max(x[:, -2]), 0.01)
    #
    # theta = model_c.theta
    # x2_c = -(theta[0] / theta[2] + theta[1] / theta[2] * x1)
    # # plt.plot(x1, x2_c, c='red', label='problemC', linewidth=2)
    #
    # theta = model_d.theta
    # x2_d = -(theta[0] / theta[2] + theta[1] / theta[2] * x1)
    # # plt.plot(x1, x2_d, c='red', label='problemD', linewidth=2)
    #
    # theta = model_e.theta / alpha
    # x2_e = -(theta[0] / theta[2] + theta[1] / theta[2] * x1)
    # plt.plot(x1, x2_e, c='red', label='problemE', linewidth=2)
    #
    # # Add labels and save to disk
    # plt.xlabel('x1')
    # plt.ylabel('x2')
    # plt.show()
    #
    # *** END CODER HERE


if __name__ == '__main__':
    main()
