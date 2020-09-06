import matplotlib.pyplot as plt
import numpy as np
import ProblemSet.ps1.src.util as util

from ProblemSet.ps1.src.p05b_lwr import LocallyWeightedLinearRegression


def main(tau_values=[3e-2, 5e-2, 1e-1, 5e-1, 1e0, 1e1],
         train_path='/Users/leeseungjoon/realest/CS229/ProblemSet/ps1/data/ds5_train.csv',
         valid_path='/Users/leeseungjoon/realest/CS229/ProblemSet/ps1/data/ds5_valid.csv',
         test_path='/Users/leeseungjoon/realest/CS229/ProblemSet/ps1/data/ds5_test.csv',
         pred_path='/Users/leeseungjoon/realest/CS229/ProblemSet/ps1/src/output/p05c_pred.txt'):
    """Problem 5(b): Tune the bandwidth paramater tau for LWR.

    Args:
        tau_values: List of tau values to try.
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # Search tau_values for the best tau (lowest MSE on the validation set)
    x_val, y_val = util.load_dataset(valid_path, add_intercept=True)

    best_tau = tau_values[0]
    mse = 1e9

    for tau in tau_values:
        model = LocallyWeightedLinearRegression(tau=tau)
        model.fit(x_train, y_train)

        y_pred = model.predict(x_val)
        temp = ((y_pred - y_val) ** 2).mean()
        print(f'tau: {tau}, mse: {temp}')

        if temp < mse:
            mse = temp
            best_tau = tau

        plt.figure()
        plt.plot(x_val, y_val, 'bx')
        plt.plot(x_val, y_pred, 'rx')
        plt.show()

    # Fit a LWR model with the best tau value
    model = LocallyWeightedLinearRegression(tau=best_tau)

    # Run on the test set to get the MSE value
    x_test, y_test = util.load_dataset(test_path, add_intercept=True)
    model.fit(x_test, y_test)

    y_pred = model.predict(x_test)
    mse = ((y_pred - y_test) ** 2).mean()
    print()
    print(f'Best tau: {best_tau}, Test mse: {mse}')

    # Save predictions to pred_path
    np.savetxt(pred_path, y_pred, delimiter=',')

    # Plot data
    plt.figure()
    plt.plot(x_test, y_test, 'bx')
    plt.plot(x_test, y_pred, 'rx')
    plt.show()
    # *** END CODE HERE ***


if __name__ == '__main__':
    main()
