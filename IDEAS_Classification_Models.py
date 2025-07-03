
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
"""from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense"""

def create_dbn_model(input_dim):
    model = Sequential()
    model.add(Dense(100, input_dim=input_dim, activation='sigmoid'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def classifier_evaluation(X, y, classifier_type='random_forest'):
    """
    Evaluate different classification models.

    Parameters:
    - X: Features
    - y: Target variable
    - classifier_type: Type of classifier ('knn', 'naive_bayes', 'svm', 'decision_tree', 'random_forest', 'xgboost', 'dbn')

    Returns:
    - accuracy: Accuracy
    - precision: Precision
    - recall: Recall
    - f1_score: F1 Score
    """

    if classifier_type == 'knn':
        classifier = KNeighborsClassifier()
    elif classifier_type == 'naive_bayes':
        classifier = GaussianNB()
    elif classifier_type == 'svm':
        classifier = SVC()
    elif classifier_type == 'decision_tree':
        classifier = DecisionTreeClassifier(random_state=100)
    elif classifier_type == 'random_forest':
        classifier = RandomForestClassifier(random_state=100)
    elif classifier_type == 'xgboost':
        classifier = XGBClassifier(random_state=100)
    elif classifier_type == 'dbn':
        # DBN-like architecture with TensorFlow and Keras
        model = create_dbn_model(X.shape[1])
        classifier = make_pipeline(StandardScaler(), model)
    else:
        raise ValueError("Invalid classifier_type. Supported types: 'knn', 'naive_bayes', 'svm', 'decision_tree', 'random_forest', 'xgboost', 'dbn'.")

    # Create an instance of StratifiedKFold with 10 folds
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=100)

    # Find the predicted values of each instance using cross_val_predict
    y_pred = cross_val_predict(classifier, X, y, cv=skf)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, average='weighted')
    recall = recall_score(y, y_pred, average='weighted')
    F1_score = f1_score(y, y_pred, average='weighted')

    # Return results
    return accuracy, precision, recall, F1_score, y_pred
