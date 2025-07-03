from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

def random_forest_regressor(X, y):
    # Create an instance of Random Forest Regressor
    RFR = RandomForestRegressor(random_state=100)

    # Create an instance of KFold with 10 folds
    kfold = KFold(n_splits=10, shuffle=True, random_state=100)

    # Find the predicted values of each instance using cross_val_predict
    y_pred = cross_val_predict(RFR, X, y, cv=kfold)

    # Calculate Mean Absolute Error (MAE)
    mae = mean_absolute_error(y, y_pred)

    # Calculate Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mean_squared_error(y, y_pred))

    # Calculate Normalized Root Mean Squared Error (NRMSE)
    nrmse = rmse / (np.max(y) - np.min(y))

    # Return results
    return mae, nrmse, rmse, y_pred






from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

def regressor_evaluation(X, y, regressor_type='random_forest'):
    """
    Evaluate different regression models.

    Parameters:
    - X: Features
    - y: Target variable
    - regressor_type: Type of regressor ('random_forest', 'svr', 'elastic_net', 'ridge', 'adaboost')

    Returns:
    - mae: Mean Absolute Error
    - nrmse: Normalized Root Mean Squared Error
    - rmse: Root Mean Squared Error
    - y_pred: Predicted values
    """

    if regressor_type == 'random_forest':
        regressor = RandomForestRegressor(random_state=100)
    elif regressor_type == 'svr':
        regressor = SVR()
    elif regressor_type == 'elastic_net':
        regressor = ElasticNet(random_state=100)
    elif regressor_type == 'ridge':
        regressor = Ridge(random_state=100)
    elif regressor_type == 'adaboost':
        regressor = AdaBoostRegressor(random_state=100)
    else:
        raise ValueError("Invalid regressor_type. Supported types: 'random_forest', 'svr', 'elastic_net', 'ridge', 'adaboost'.")

    # Create an instance of KFold with 10 folds
    kfold = KFold(n_splits=10, shuffle=True, random_state=100)

    # Find the predicted values of each instance using cross_val_predict
    y_pred = cross_val_predict(regressor, X, y, cv=kfold)

    # Calculate Mean Absolute Error (MAE)
    mae = mean_absolute_error(y, y_pred)

    # Calculate Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mean_squared_error(y, y_pred))

    # Calculate Normalized Root Mean Squared Error (NRMSE)
    nrmse = rmse / (np.max(y) - np.min(y))

    # Return results
    return mae, nrmse, rmse, y_pred

"""# Example usage:
# Assuming X_train and y_train are your training features and target variable
mae_rf, nrmse_rf, rmse_rf, y_pred_rf = regressor_evaluation(X_train, y_train, regressor_type='random_forest')
mae_svr, nrmse_svr, rmse_svr, y_pred_svr = regressor_evaluation(X_train, y_train, regressor_type='svr')
mae_en, nrmse_en, rmse_en, y_pred_en = regressor_evaluation(X_train, y_train, regressor_type='elastic_net')
mae_rr, nrmse_rr, rmse_rr, y_pred_rr = regressor_evaluation(X_train, y_train, regressor_type='ridge')
mae_ab, nrmse_ab, rmse_ab, y_pred_ab = regressor_evaluation(X_train, y_train, regressor_type='adaboost')

print("Random Forest - MAE:", mae_rf, "NRMSE:", nrmse_rf, "RMSE:", rmse_rf)
print("SVR - MAE:", mae_svr, "NRMSE:", nrmse_svr, "RMSE:", rmse_svr)
print("Elastic Net - MAE:", mae_en, "NRMSE:", nrmse_en, "RMSE:", rmse_en)
print("Ridge Regression - MAE:", mae_rr, "NRMSE:", nrmse_rr, "RMSE:", rmse_rr)
print("AdaBoost - MAE:", mae_ab, "NRMSE:", nrmse_ab, "RMSE:", rmse_ab)
"""