import numpy as np
import pandas as pd
from loguru import logger
import matplotlib.pyplot as plt
# from sklearn.linear_model import LinearRegression


class LinearRegressionBase:
    def __init__(self):
        self.weights = None
        self.intercept = None

    def fit(self):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

# def scale_features(X):
#    return (X - np.mean(X)) / np.std(X)


class LinearRegressionCloseform(LinearRegressionBase):
    def fit(self, X, y):
        X_with_intercept = np.c_[np.ones(X.shape[0]), X]
        self.weights = np.linalg.inv(X_with_intercept.T @ X_with_intercept) @ X_with_intercept.T @ y
        self.intercept = self.weights[0]
        self.weights = self.weights[1:]

    def predict(self, X):
        return X @ self.weights + self.intercept


class LinearRegressionGradientdescent(LinearRegressionBase):
    def __init__(self):
        super().__init__()
        self.feature_mean = None
        self.feature_std = None

    def scale_features(self, X, is_training=False):
        if is_training:
            self.feature_mean = np.mean(X, axis=0)
            self.feature_std = np.std(X, axis=0)
        return (X - self.feature_mean) / self.feature_std

    def fit(self, X, y, learning_rate: float = 0.001, epochs: int = 1000):
        # X = scale_features(X)
        X = self.scale_features(X, is_training=True)
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.intercept = 0
        losses = []

        for _ in range(epochs):
            y_pred = self.predict(X, is_training=True)
            error = y_pred - y

            gradient = (2 / n_samples) * X.T @ error
            intercept_gradient = (2 / n_samples) * np.sum(error)

            self.weights -= learning_rate * gradient
            self.intercept -= learning_rate * intercept_gradient

            loss = compute_mse(y_pred, y)
            losses.append(loss)

        return losses

    def predict(self, X, is_training=False):
        if not is_training:
            X = self.scale_features(X)
        return X @ self.weights + self.intercept

    def plot_learning_curve(self, losses):
        plt.plot(losses)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.show()


def compute_mse(prediction, ground_truth):
    return np.average(np.square(prediction - ground_truth))


def main():
    train_df = pd.read_csv('./train.csv')
    train_x = train_df.drop(["Performance Index"], axis=1).to_numpy()
    train_y = train_df["Performance Index"].to_numpy()

#    model = LinearRegression()
#    model.fit(train_x, train_y)
#    print("weights: ", model.coef_)
#    print("Intercept: ", model.intercept_)

    LR_CF = LinearRegressionCloseform()
    LR_CF.fit(train_x, train_y)
    logger.info(f'{LR_CF.weights=}, {LR_CF.intercept=:.4f}')

    LR_GD = LinearRegressionGradientdescent()
    losses = LR_GD.fit(train_x, train_y, learning_rate=1e-4, epochs=40000)
    LR_GD.plot_learning_curve(losses)
    logger.info(f'{LR_GD.weights=}, {LR_GD.intercept=:.4f}')

    test_df = pd.read_csv('./test.csv')
    test_x = test_df.drop(["Performance Index"], axis=1).to_numpy()
    test_y = test_df["Performance Index"].to_numpy()

    y_preds_cf = LR_CF.predict(test_x)
    y_preds_gd = LR_GD.predict(test_x)
    y_preds_diff = np.abs(y_preds_gd - y_preds_cf).sum()
    logger.info(f'Prediction difference: {y_preds_diff:.4f}')

    mse_cf = compute_mse(y_preds_cf, test_y)
    mse_gd = compute_mse(y_preds_gd, test_y)
    diff = ((mse_gd - mse_cf) / mse_cf) * 100
    logger.info(f'{mse_cf=:.4f}, {mse_gd=:.4f}. Difference: {diff:.3f}%')


if __name__ == '__main__':
    main()
