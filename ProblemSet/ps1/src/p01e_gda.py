import numpy as np
import ProblemSet.ps1.src.util as util

from ProblemSet.ps1.src.linear_model import LinearModel


def main(train_path='/Users/leeseungjoon/realest/CS229/ProblemSet/ps1/data/ds1_train.csv',
         eval_path='/Users/leeseungjoon/realest/CS229/ProblemSet/ps1/data/ds1_valid.csv',
         pred_path='/Users/leeseungjoon/realest/CS229/ProblemSet/ps1/src/output/p01e_pred_1.txt'):
    """Problem 1(e): Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # *** START CODE HERE ***
    # Train a GDA classifier
    model = GDA()
    model.fit(x_train, y_train)

    # Plot decision boundary on validation set
    x_val, y_val = util.load_dataset(eval_path, add_intercept=False)
    util.plot(x_val, y_val, model.theta, f'{pred_path}.png')

    # Use np.savetxt to save outputs from validation set to pred_path
    h = model.predict(x_val)
    pred = np.where(h >= 0.5, 1, 0)
    np.savetxt(pred_path, pred, delimiter=',')

    print('Number of wrong predictions')
    print((y_val != pred).sum())
    # *** END CODE HERE ***


class GDA(LinearModel):
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    # def __init__(self):
    #     super().__init__()
    #     self.theta0 = 0

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).

        Returns:
            theta: GDA model parameters.
        """
        # *** START CODE HERE ***
        # Find phi, mu_0, mu_1, and sigma
        m, n = x.shape
        phi = (y == 1).sum() / m

        mu_0 = x[y == 0].sum(axis=0) / (y == 0).sum()
        mu_1 = x[y == 1].sum(axis=0) / (y == 1).sum()

        temp = np.zeros((m, n))
        temp[y == 0] = mu_0
        temp[y == 1] = mu_1

        vec = x - temp
        sigma = np.zeros((n,n))
        for k in vec:
            tmp = np.matmul(k.reshape(-1, 1), k.reshape(-1, 1).T)
            sigma += tmp

        # Write theta in terms of the parameters
        inv = np.linalg.inv(sigma)
        theta = (mu_1 - mu_0).dot(inv)
        theta0 = ((np.matmul(mu_0.dot(inv), mu_0) - np.matmul(mu_1.dot(inv), mu_1)) / 2) - np.log(1/phi - 1)

        theta0 = np.array([theta0])
        theta = np.hstack([theta0, theta])
        self.theta = theta
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        x = util.add_intercept(x)

        z = x.dot(self.theta)
        gz = 1 / (1 + np.exp(-z))

        return gz
        # *** END CODE HERE


if __name__ == '__main__':
    main(train_path='/Users/leeseungjoon/realest/CS229/ProblemSet/ps1/data/ds1_train.csv',
         eval_path='/Users/leeseungjoon/realest/CS229/ProblemSet/ps1/data/ds1_valid.csv',
         pred_path='/Users/leeseungjoon/realest/CS229/ProblemSet/ps1/src/output/p01e_pred_1.txt')

    main(train_path='/Users/leeseungjoon/realest/CS229/ProblemSet/ps1/data/ds2_train.csv',
         eval_path='/Users/leeseungjoon/realest/CS229/ProblemSet/ps1/data/ds2_train.csv',
         pred_path='/Users/leeseungjoon/realest/CS229/ProblemSet/ps1/src/output/p01e_pred_2.txt')
