import os
import numpy as np
import pandas as pd
from flask import Flask, request, render_template
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from scipy.sparse import csr_matrix

# Custom implementation of M5Rules
class M5Rules(BaseEstimator, RegressorMixin):
    def __init__(self, min_samples_split=10, max_depth=5):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.model_tree = None
        self.linear_models = {}

    def fit(self, X, y):
        self.model_tree = DecisionTreeRegressor(min_samples_split=self.min_samples_split, max_depth=self.max_depth)
        self.model_tree.fit(X, y)
        self._fit_linear_models(X, y)
        return self

    def _fit_linear_models(self, X, y):
        # Fit linear models to the terminal nodes
        leaf_indices = self.model_tree.apply(X)
        for leaf in np.unique(leaf_indices):
            indices = np.where(leaf_indices == leaf)
            X_leaf, y_leaf = X[indices], y[indices]
            if len(y_leaf) > 1:
                linear_model = LinearRegression()
                linear_model.fit(X_leaf, y_leaf)
                self.linear_models[leaf] = linear_model

    def predict(self, X):
        if isinstance(X, csr_matrix):
            X = X.toarray()
        leaf_indices = self.model_tree.apply(X)
        predictions = []
        for i, leaf in enumerate(leaf_indices):
            if leaf in self.linear_models:
                linear_model = self.linear_models[leaf]
                predictions.append(linear_model.predict(X[i].reshape(1, -1))[0])
            else:
                predictions.append(self.model_tree.tree_.value[leaf][0][0])
        return np.array(predictions)

# Flask app setup
app = Flask(__name__)
app.template_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')

# Load the dataset
DATA_PATH = "electric_vehicle_sales_by_state.csv"

def load_data():
    data = pd.read_csv(DATA_PATH, sep='\t')
    # Assuming 'electric_vehicles_sold' is the target column
    categorical_features = ['state', 'vehicle_category']
    X = data.drop(columns=['electric_vehicles_sold', 'date'])
    y = data['electric_vehicles_sold'].values

    # Ensure categorical features are of type string for one-hot encoding
    for feature in categorical_features:
        X[feature] = X[feature].astype(str)

    # One-hot encode categorical features
    column_transformer = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(sparse_output=False), categorical_features)
        ],
        remainder='passthrough'
    )
    X_transformed = column_transformer.fit_transform(X)
    return X_transformed, y, data

@app.route('/')
def home():
    return render_template('index2.html')

# Power BI Dashboard Page Route
@app.route('/powerbi-dashboard')
def powerbi_dashboard():
    return render_template('powerbi_dashboard.html')


# About Us Page Route
@app.route('/about-us')
def about_us():
    return render_template('about_us.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
     # Get user input for the state
    if request.method == 'POST':
        state_input = request.form.get('state')
    else:
        return render_template('index.html', predictions=["Please enter a state."])
    
    # Load dataset
    X, y, data = load_data()
    
    # Filter data for the specified state
    state_data = data[data['state'] == state_input]
    if state_data.empty:
        return render_template('index.html', predictions=[f"No data available for state: {state_input}"])
    
    X_state = X[data['state'] == state_input]
    y_state = y[data['state'] == state_input]
    
    # Train M5Rules model
    model = M5Rules()
    model.fit(X, y)
    
    # Make predictions on the filtered data
    predictions = model.predict(X_state)
    total_abs_predictions = np.round(np.sum(np.abs(predictions)))
    
    return render_template('result.html', predictions=total_abs_predictions, state=state_input)

if __name__ == "__main__":
    app.run(debug=True)