import numpy as np

### Functions for you to fill in ###
def IdentityMatrix(n):
    imat = []
    for r in range(n):
        row = []
        for c in range(n):
            if r==c:
                row.append(1)
            else:
                row.append(0)
        imat.append(row)
    return imat
    
def closed_form(X, Y, lambda_factor):
    """
    Computes the closed form solution of linear regression with L2 regularization

    Args:
        X - (n, d + 1) NumPy array (n datapoints each with d features plus the bias feature in the first dimension)
        Y - (n, ) NumPy array containing the labels (a number from 0-9) for each
            data point
        lambda_factor - the regularization constant (scalar)
    Returns:
        theta - (d + 1, ) NumPy array containing the weights of linear regression. Note that theta[0]
        represents the y-axis intercept of the model and therefore X[0] = 1
    """
    I = IdentityMatrix(X.shape[1])    
    step1 = np.dot(X.T,X)
    step2 = np.add(step1, np.dot(lambda_factor, I))
    step3 = np.linalg.inv(step2)
    theta = np.dot(step3, np.dot(X.T,Y))
    return theta


def compute_test_error_linear(test_x, Y, theta):
    test_y_predict = np.round(np.dot(test_x, theta))
    test_y_predict[test_y_predict < 0] = 0
    test_y_predict[test_y_predict > 9] = 9
    return 1 - np.mean(test_y_predict == Y)
