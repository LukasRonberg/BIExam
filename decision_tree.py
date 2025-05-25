import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def train_decision_tree(
    df: pd.DataFrame,
    feature_cols: list,
    target_col: str,
    test_size: float = 0.2,
    random_state: int = 42,
    max_depth: int = 4
) -> tuple:
    """
    Train a DecisionTreeRegressor and return model components and metrics.
    Returns: (model, X_train, X_test, y_train, y_test, y_pred, metrics_dict)
    """
    X = df[feature_cols]
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    model = DecisionTreeRegressor(max_depth=max_depth, random_state=random_state)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = {
        'mse': mean_squared_error(y_test, y_pred),
        'mae': mean_absolute_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred)
    }
    return model, X_train, X_test, y_train, y_test, y_pred, metrics


def plot_decision_tree_model(
    model,
    feature_names: list,
    figsize: tuple = (20, 10)
) -> plt.Figure:
    """
    Return a matplotlib Figure of the decision tree.
    """
    fig = plt.figure(figsize=figsize)
    plot_tree(model, feature_names=feature_names, filled=True, rounded=True)
    fig.tight_layout()
    return fig


def get_feature_importance(
    model,
    feature_names: list
) -> pd.DataFrame:
    """
    Return a DataFrame of feature importances sorted descending.
    """
    importances = model.feature_importances_
    df_imp = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    return df_imp


def plot_feature_importance(
    importance_df: pd.DataFrame,
    title: str,
    figsize: tuple = (8, 4)
) -> plt.Figure:
    """
    Return a horizontal bar plot of feature importances.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(importance_df['Feature'], importance_df['Importance'])
    ax.set_title(title)
    ax.set_xlabel('Importance')
    ax.invert_yaxis()
    fig.tight_layout()
    return fig


def plot_actual_vs_predicted(
    y_test,
    y_pred,
    xlabel: str = 'Actual',
    ylabel: str = 'Predicted',
    title: str = 'Actual vs. Predicted',
    figsize: tuple = (10, 6)
) -> plt.Figure:
    """
    Return a scatter plot comparing actual and predicted values.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(y_test, y_pred, alpha=0.3)
    min_val, max_val = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True)
    fig.tight_layout()
    return fig


def train_random_forest(
    df: pd.DataFrame,
    feature_cols: list,
    target_col: str,
    test_size: float = 0.2,
    random_state: int = 42,
    n_estimators: int = 100,
    max_depth: int = None
) -> tuple:
    """
    Train a RandomForestRegressor and return model components and metrics.
    Returns: (model, X_train, X_test, y_train, y_test, y_pred, metrics_dict)
    """
    X = df[feature_cols]
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = {
        'mse': mean_squared_error(y_test, y_pred),
        'mae': mean_absolute_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred)
    }
    return model, X_train, X_test, y_train, y_test, y_pred, metrics


def predict_suicides_for_population(
    tree_model,
    populations: pd.DataFrame,
    age_order: list,
    gdp_per_capita: float
) -> pd.DataFrame:
    """
    Given a decision tree model and a DataFrame with columns ['sex','age','population'],
    return a DataFrame including predictions per 100k and expected counts.
    """
    df_in = populations.copy()
    df_in['age'] = pd.Categorical(
        df_in['age'],
        categories=age_order,
        ordered=True
    )
    df_in['age_encoded'] = df_in['age'].cat.codes
    df_in['sex_numeric'] = df_in['sex'].map({'male': 1, 'female': 2})
    df_in['gdp_per_capita ($)'] = gdp_per_capita
    X = df_in[['age_encoded', 'sex_numeric', 'gdp_per_capita ($)']]
    df_in['suicides_per_100k'] = tree_model.predict(X)
    df_in['expected_suicides'] = (
        df_in['suicides_per_100k'] / 100000
    ) * df_in['population']
    return df_in.sort_values(by='suicides_per_100k', ascending=False)
