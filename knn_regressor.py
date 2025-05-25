import pandas as pd

from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def train_knn_regressor(
    df: pd.DataFrame,
    feature_cols: list,
    target_col: str,
    test_size: float = 0.25,
    random_state: int = 42,
    n_neighbors: int = 5
) -> tuple:
    """
    Train a KNeighborsRegressor and return:
      - trained model
      - X_train, X_test, y_train, y_test
      - model predictions (y_pred)
      - metrics dict {'mse', 'mae', 'r2'}

    Parameters:
    - df: pandas DataFrame containing the dataset
    - feature_cols: list of column names to use as features
    - target_col: name of the target column
    - test_size: fraction of data to reserve for testing (default 0.25)
    - random_state: random seed for reproducibility (default 42)
    - n_neighbors: number of neighbors for KNN (default 5)
    """
    # Prepare feature matrix X and target vector y
    X = df[feature_cols]
    y = df[target_col]

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Initialize and train the KNN regressor
    model = KNeighborsRegressor(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)

    # Generate predictions on the test set
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics
    metrics = {
        'mse': mean_squared_error(y_test, y_pred),
        'mae': mean_absolute_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred)
    }

    return model, X_train, X_test, y_train, y_test, y_pred, metrics
