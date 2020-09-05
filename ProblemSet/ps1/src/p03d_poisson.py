import numpy as np
import ProblemSet.ps1.src.util as util

from ProblemSet.ps1.src.linear_model import LinearModel


def main(lr=1e-8,
         train_path='/Users/leeseungjoon/realest/CS229/ProblemSet/ps1/data/ds4_train.csv',
         eval_path='/Users/leeseungjoon/realest/CS229/ProblemSet/ps1/data/ds4_valid.csv',
         pred_path='/Users/leeseungjoon/realest/CS229/ProblemSet/ps1/src/output/p03d_pred.txt'):
    """Problem 3(d): Poisson regression with gradient ascent.

    Args:
        lr: Learning rate for gradient ascent.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)
    # x_train.shape: (2500, 4)
    # y_train.shape: (2500,)

    # *** START CODE HERE ***

    # Fit a Poisson Regression model
    model = PoissonRegression(max_iter=1000000, step_size=lr, eps=1e-5)
    model.fit(x_train, y_train)

    # Run on the validation set, and use np.savetxt to save outputs to pred_path
    x_val, y_val = util.load_dataset(eval_path, add_intercept=False)
    pred = model.predict(x_val)

    np.savetxt(pred_path, pred, delimiter=',')
    # *** END CODE HERE ***


class PoissonRegression(LinearModel):
    """Poisson Regression.

    Example usage:
        > clf = PoissonRegression(step_size=lr)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Run gradient ascent to maximize likelihood for Poisson regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        m, n = x.shape
        if self.theta is None:
            self.theta = np.zeros(n)

        # g = lambda k: np.exp(k)
        # for i in range(self.max_iter):
        #     theta = self.theta
        #     grad = (1 / m) * (y - g(x.dot(theta))).dot(x)
        #
        #     cost_vector = g(x.dot(theta)) - y * x.dot(theta)
        #     cost = - np.sum(cost_vector)
        #
        #     self.theta = theta + self.step_size * grad
        #     print(f'i = {i}, theta = {theta}, delta of theta = {np.linalg.norm(self.theta - theta, ord=1)}, cost = {cost}')
        #     if np.linalg.norm(self.theta - theta, ord=1) < self.eps:
        #         break

        iter_num = 0
        cost = 1e9

        while iter_num <= self.max_iter:
            natural = np.matmul(x, self.theta)
            h = np.exp(natural)

            to_sub = y * natural
            cost_vector = h - to_sub

            cost = - np.sum(cost_vector)
            # print(f'Iter Num: {iter_num}, cost: {cost}')

            grad = (1 / m) * np.matmul(x.T, y-h)
            new_theta = self.theta + self.step_size * grad

            print(f'i = {iter_num}, theta = {self.theta}, delta of theta = {np.linalg.norm(self.theta - new_theta, ord=1)}, cost = {cost}')
            if np.linalg.norm(self.theta - new_theta, ord=1) < self.eps:
                break

            self.theta = new_theta
            iter_num += 1

        # print(f'Converged after {iter_num} iterations, and cost is {cost}')
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Floating-point prediction for each input, shape (m,).
        """
        # *** START CODE HERE ***
        natural = np.matmul(x, self.theta)
        h = np.exp(natural)

        return h
        # *** END CODE HERE ***


if __name__ == '__main__':
    main()
