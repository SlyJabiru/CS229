import matplotlib.pyplot as plt
import numpy as np
import ProblemSet.ps1.src.util as util

from ProblemSet.ps1.src.linear_model import LinearModel


def main(tau=0.5,
         train_path='/Users/leeseungjoon/realest/CS229/ProblemSet/ps1/data/ds5_train.csv',
         eval_path='/Users/leeseungjoon/realest/CS229/ProblemSet/ps1/data/ds5_valid.csv'):
    """Problem 5(b): Locally weighted regression (LWR)

    Args:
        tau: Bandwidth parameter for LWR.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # Fit a LWR model
    model = LocallyWeightedLinearRegression(tau)
    model.fit(x_train, y_train)

    # # Get MSE value on the validation set
    x_val, y_val = util.load_dataset(eval_path, add_intercept=True)
    y_pred = model.predict(x_val)
    mse = ((y_pred - y_val) ** 2).mean()
    print(mse)

    # Plot validation predictions on top of training set
    # No need to save predictions
    # Plot data
    plt.figure()
    plt.plot(x_val, y_val, 'bx')
    plt.plot(x_val, y_pred, 'rx')
    plt.show()
    # *** END CODE HERE ***


class LocallyWeightedLinearRegression(LinearModel):
    """Locally Weighted Regression (LWR).

    Example usage:
        > clf = LocallyWeightedLinearRegression(tau)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self, tau):
        super(LocallyWeightedLinearRegression, self).__init__()
        self.tau = tau
        self.x = None
        self.y = None

    def fit(self, x, y):
        """Fit LWR by saving the training set.

        """
        # *** START CODE HERE ***
        self.x = x
        self.y = y
        # *** END CODE HERE ***

    def predict(self, x):
        """Make predictions given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        y_pred = np.zeros(x.shape[0])

        for i, x_val_i in enumerate(x):
            # x_val_i [123, 312] 즉, (1, 2) 행렬. 즉, (2,)
            sub = self.x - x_val_i  # train_x 의 모든 row 에 대해, x_val_i 를 뺀 것.
            numerator = np.linalg.norm(sub, axis=1)
            denominator = 2 * (self.tau ** 2)
            wi = np.exp(- numerator / denominator)

            W = np.diag(wi)
            to_inverse = np.matmul(np.matmul(self.x.T, W), self.x)  # (X^T)WX
            inv = np.linalg.inv(to_inverse)
            temp = np.matmul(np.matmul(self.x.T, W), self.y)

            theta = np.matmul(inv, temp)  # (n,)
            yi = theta.dot(x_val_i)
            y_pred[i] = yi

        return y_pred
        # *** END CODE HERE ***


if __name__ == '__main__':
    main()
