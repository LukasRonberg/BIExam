import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error


def train_linear_regression(
    df: pd.DataFrame,
    feature_cols: list,
    target_col: str,
    test_size: float = None,
    random_state: int = 42
) -> tuple:
    """
    Train a LinearRegression model.

    Parameters:
    - df: pandas DataFrame containing the dataset
    - feature_cols: list of column names to use as features
    - target_col: name of the target column
    - test_size: fraction of data to reserve for testing (None for no split)
    - random_state: random seed for reproducibility

    Returns:
    - model: trained LinearRegression
    - X_eval: features used for evaluation (either full X or X_test)
    - y_eval: target values used for evaluation
    - y_pred: model predictions on X_eval
    - metrics: dict with 'r2' and 'mse'
    """
    X = df[feature_cols]
    y = df[target_col]

    if test_size is not None and test_size > 0:
        X_train, X_eval, y_train, y_eval = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        model = LinearRegression()
        model.fit(X_train, y_train)
    else:
        X_eval = X
        y_eval = y
        model = LinearRegression()
        model.fit(X, y)

    y_pred = model.predict(X_eval)
    metrics = {
        'r2': r2_score(y_eval, y_pred),
        'mse': mean_squared_error(y_eval, y_pred)
    }
    return model, X_eval, y_eval, y_pred, metrics


def plot_regression_line(
    model,
    df: pd.DataFrame,
    feature_col: str,
    target_col: str,
    figsize: tuple = (8, 5)
) -> plt.Figure:
    """
    Plot scatter and regression line for a single feature linear model.

    Parameters:
    - model: trained LinearRegression
    - df: DataFrame containing feature_col and target_col
    - feature_col: name of the feature column
    - target_col: name of the target column
    - figsize: size of the figure

    Returns:
    - fig: matplotlib Figure object
    """
    X = df[[feature_col]]
    y = df[target_col]
    y_pred = model.predict(X)

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(X, y, alpha=0.3, label='Observations')
    ax.plot(X, y_pred, color='red', label='Regression line')
    ax.set_xlabel(feature_col)
    ax.set_ylabel(target_col)
    ax.set_title(f'Linear Regression: {target_col} vs. {feature_col}')
    ax.legend()
    fig.tight_layout()
    return fig


def plot_actual_vs_predicted(
    y_true,
    y_pred,
    title: str = 'Actual vs. Predicted',
    figsize: tuple = (8, 5)
) -> plt.Figure:
    """
    Plot actual vs. predicted values for evaluation of linear models.

    Parameters:
    - y_true: true target values (array-like)
    - y_pred: predicted target values (array-like)
    - title: title of the plot
    - figsize: size of the figure

    Returns:
    - fig: matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.3, ax=ax)
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    ax.plot([min_val, max_val], [min_val, max_val], 'r--')
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_title(title)
    fig.tight_layout()
    return fig
