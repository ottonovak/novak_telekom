import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the CSV data from file
data = pd.read_csv('housing.csv')

# Replace 'Null' strings with actual NaN values
data.replace('Null', pd.NA, inplace=True)

# Drop rows with any missing values
data.dropna(inplace=True)

# Split data into features and target variable
X = data.drop(['MEDIAN_HOUSE_VALUE', 'OCEAN_PROXIMITY'], axis=1)
y = data['MEDIAN_HOUSE_VALUE']

# Define the preprocessing steps for numeric and categorical features
numeric_features = ['LONGITUDE', 'LAT', 'MEDIAN_AGE', 'ROOMS', 'BEDROOMS', 'POP', 'HOUSEHOLDS', 'MEDIAN_INCOME']
categorical_features = ['OCEAN_PROXIMITY']

numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
    ])

# Create a pipeline with preprocessing and linear regression model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Reserve the last 10% of data for testing
split_index = int(len(data) * 0.9)
X_train = X[:split_index]
y_train = y[:split_index]
X_test = X[split_index:]
y_test = y[split_index:]

# Train the model
model.fit(X_train, y_train)

# Save the trained model to a file
joblib.dump(model, 'new_model.joblib')

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model performance
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Model performance on test data:")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R²): {r2}")

# Save the testing dataset to a CSV file
testing_data = pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)
testing_data.to_csv('data_testing_testing.csv', index=False)

print("Model training and saving complete. Testing data saved to data_testing_testing.csv.")
