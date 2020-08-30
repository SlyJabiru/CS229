import numpy as np
import ProblemSet.ps1.src.util as util

from ProblemSet.ps1.src.linear_model import LinearModel


def main(train_path='/Users/leeseungjoon/realest/CS229/ProblemSet/ps1/data/ds1_train.csv',
         eval_path='/Users/leeseungjoon/realest/CS229/ProblemSet/ps1/data/ds1_valid.csv',
         pred_path='/Users/leeseungjoon/realest/CS229/ProblemSet/ps1/src/output/p01b_pred_1.txt'):
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    # x_train.shape: (800, 3) X0 X1 X2
    # y_train.shape (800, )

    # *** START CODE HERE ***
    theta = np.zeros(x_train.shape[1])
    clf = LogisticRegression(theta_0=theta)

    # Train a logistic regression classifier
    clf.fit(x_train, y_train)

    # Plot decision boundary on top of validation set set
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)
    util.plot(x_eval, y_eval, clf.theta, '/Users/leeseungjoon/realest/CS229/ProblemSet/ps1/data/ds1_plot.png')

    # Use np.savetxt to save predictions on eval set to pred_path
    pred = clf.predict(x_eval)
    np.savetxt(pred_path, pred, delimiter=',')
    # *** END CODE HERE ***


class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***

        tol = 1e9
        n_iters = 0

        while tol > 1e-6:
            z = x.dot(self.theta)
            gz = 1 / (1 + np.exp(-z))  # == h(x)

            diff = gz - y  # h(x) -y
            grad = np.matmul(x.T, diff)  # (3,) grad(J(theta))

            temp = gz * (1 - gz)
            for_multiply = np.tile(temp.transpose(), (3, 1)).T
            tmp = for_multiply * x  # h(x) * (1 - h(x)) * x

            hessian = np.matmul(x.T, tmp)
            inverse = np.linalg.inv(hessian)

            sub = np.matmul(inverse, grad)
            new_theta = self.theta - sub
            tol = np.sum(np.abs(new_theta - self.theta))
            print(tol)
            self.theta = new_theta
            n_iters += 1

        print(f'Converged after {n_iters} iterations')
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        z = x.dot(self.theta)
        gz = 1 / (1 + np.exp(-z))  # == h(x)

        return np.where(gz >= 0.5, 1, 0)
        # *** END CODE HERE ***


if __name__ == '__main__':
    main()
