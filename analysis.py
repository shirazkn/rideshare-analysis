import numpy as np
import pandas as pd
import torch

from plotting import line_xy
from pre_analysis import minutes_in_day
from sklearn.preprocessing import StandardScaler


# Loss function used for training
criterion = torch.nn.MSELoss()
torch.manual_seed(0)


# Linear Regression Model
class RegModel(torch.nn.Module):
    """
    Neural network which stores
    """

    def __init__(self, inputs, outputs, learning_rate=0.1, scale_features=True, linear=True):
        super(RegModel, self).__init__()

        self.inputs = inputs
        self.outputs = outputs
        self.input_scaler = StandardScaler() if scale_features else NoScaler()
        self.output_scaler = StandardScaler() if scale_features else NoScaler()

        n_inputs = len(inputs)
        n_outputs = len(outputs)

        # Linear Regression Model
        if linear:
            self.Linear = torch.nn.Linear(n_inputs, n_outputs)
            self.Linear.weight.data.uniform_(-1, 1)
            self.forward = self.linear_forward

        # Non-linear Regression Model
        else:
            n_hidden = int(np.max([n_inputs + n_outputs - 5, 3]))  # No. of neurons in the hidden layer
            self.Input = torch.nn.Linear(n_inputs, n_hidden)
            self.Output = torch.nn.Linear(n_hidden, n_outputs)
            self.forward = self.nonlinear_forward

        self.optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)
        self.fitted = "Not Calculated"

        self.sse = None
        self.sst = None

    def linear_forward(self, x):
        x = self.Linear(x)
        return x

    def nonlinear_forward(self, x):
        x = self.Input(x)
        x = torch.tanh(x)  # Non-linear Activation
        x = self.Output(x)
        return x

    def fit_data(self, data, epochs=100, outliers=None):
        """
        Learn from each row in data (Pandas DataFrame)
        :param data: Training Data
        :param epochs: Training epochs (int)
        :param outliers: Indices of outlier points
        """
        self.train()
        X, Y = self.XY_from_data(data, outliers)
        losses = []
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            Y_pred = self(X)
            loss = criterion(Y_pred, Y)
            loss.backward()
            self.optimizer.step()
            losses.append(loss)
        self.eval()
        return losses

    def XY_from_data(self, data, outliers=None):
        """
        :param data: Pandas DataFrame
        :param outliers: Data points to be omitted from the returned lists
        :return: list(Tensor), list(Tensor)
        """
        if outliers:
            data = data.drop(data.index[outliers])

        X_array = self.input_scaler.fit_transform(np.array(data[self.inputs], dtype=float))
        Y_array = self.output_scaler.fit_transform(np.array(data[self.outputs], dtype=float))
        X = torch.from_numpy(X_array).float()
        Y = torch.from_numpy(Y_array).float()
        for x, y in zip(X, Y):
            x.requires_grad = True
            y.requires_grad = False

        return X, Y

    def predict(self, x):
        """
        Uses NN to make prediction
        :param x: Float, unscaled input
        :return: Float, unscaled output
        """
        x = self.input_scaler.transform([x])[0]
        y = self(torch.tensor(x, requires_grad=False).float()).detach().numpy()
        return self.output_scaler.inverse_transform([y])[0]

    def regression_line_1(self, label="Fitted Line", color="blue", **kwargs):
        """
        Plot regression line for the single predictor case
        """
        plot_data = pd.DataFrame()
        plot_data["Start Time in Minutes"] = minutes_in_day(points=4000)
        plot_data["Total Fare per Minute"] = [self.predict([x])[0]
                                              for x in plot_data["Start Time in Minutes"]]
        return line_xy(x="Start Time in Minutes", y="Total Fare per Minute", data=plot_data,
                       label=label, color=color, **kwargs)

    def regression_lines_2(self, weekdays, palette, **kwargs):
        """
        Plot regression line for the single predictor case
        """
        plot_data = pd.DataFrame()
        plot_data["Start Time in Minutes"] = minutes_in_day(points=4000)
        for weekday, color in zip(weekdays, palette):
            inputs = [[min, 0, 0, 0, 0, 0, 0, 0] for min in plot_data["Start Time in Minutes"]]
            for x in inputs:
                x[weekday + 1] = 1

            plot_data["Total Fare per Minute"] = [self.predict(x)[0]
                                                  for x in inputs]
            line_xy(x="Start Time in Minutes", y="Total Fare per Minute", data=plot_data, color=color, **kwargs)

    def compute_residuals(self, data, outliers=None):
        self.eval()
        X, Y = self.XY_from_data(data, outliers)
        Y_pred = self(X)

        self.fitted = data.copy()
        if outliers:
            self.fitted = data.drop(data.index[outliers])

        self.fitted["Predictions"] = [t.detach().numpy()[0] for t in Y_pred]
        self.fitted["Residuals"] = [t.detach().numpy()[0]*(-1) for t in Y_pred - Y]
        self.fitted["Residuals_Sq"] = [t**2 for t in self.fitted["Residuals"]]
        self.sse = self.fitted["Residuals_Sq"].sum()
        total_errors = np.array([t.detach().numpy()[0] - Y.mean().detach().numpy() for t in Y])
        self.sst = np.sum(total_errors**2)

    def get_outliers_from_residuals(self, limits):
        indices = []
        for i, e in enumerate(self.fitted["Residuals"]):
            if e < limits[0] or e > limits[1]:
                indices.append(i)
        return indices


class NoScaler:
    """
    Dummy 'scaler' class, just does identity transformation for all elements
    Using this to compare effect of scaling variables for neural nets
    """
    def __init__(self):
        pass

    def fit_transform(self, x):
        return x

    def transform(self, x):
        return x

    def inverse_transform(self, x):
        return x