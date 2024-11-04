import sys
import pandas as pd
from sklearn.compose import ColumnTransformer
import joblib
import logging
import sqlite3
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

def predict(X, model):
    try:
        logging.info('Predicting data...')
        Y = model.predict(X)
        return Y
    except Exception as e:
        logging.error(f"Error predicting data: {e}")
        return None


def load_model(filename):
    try:
        logging.info(f'Loading the model from {filename}...')
        model = joblib.load(filename)
        return model
    except FileNotFoundError:
        logging.error(f"File '{filename}' does not exist.")
    except ValueError as e:
        logging.error(f"ValueError: {e}")
    except Exception as e:
        logging.error(f"Error loading model: {e}")
    return None


# Function for creation of SQLite database and tables
def create_db_and_tables():
    try:
        logging.info('Creating SQLite database and tables...')
        conn = sqlite3.connect('database.db')
        cursor = conn.cursor()

        cursor.execute('''DROP TABLE IF EXISTS transformed_data''')
        cursor.execute('''DROP TABLE IF EXISTS predictions''')

        # Create table for transformed data if it doesn't already exist
        cursor.execute('''CREATE TABLE IF NOT EXISTS transformed_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            LONGITUDE REAL,
            LAT REAL,
            MEDIAN_AGE REAL,
            ROOMS REAL,
            BEDROOMS REAL,
            POP REAL,
            HOUSEHOLDS REAL,
            MEDIAN_INCOME REAL,
            AGENCY TEXT
        )''')

        # Create table for predictions
        cursor.execute('''CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            MEDIAN_HOUSE_VALUE REAL
        )''')

        conn.commit()
        conn.close()
    except Exception as e:
        logging.error(f"Error creating SQLite database and tables: {e}")


def save_transformed_data_to_db(df, table_name):
    try:
        logging.info(f'Saving transformed data to {table_name} table...')
        conn = sqlite3.connect('database.db')
        df.to_sql(table_name, conn, if_exists='replace', index=False)
        conn.close()
    except Exception as e:
        logging.error(f"Error saving transformed data: {e}")


def clean_data_and_split_features_and_target(df):
    try:
        logging.info('Cleaning data and splitting into features & target...')

        # Drop rows with any missing values
        df.dropna(inplace=True)

        # Define the preprocessing steps for numeric and categorical features
        numeric_features = ['LONGITUDE', 'LAT', 'MEDIAN_AGE', 'ROOMS', 'BEDROOMS', 'POP', 'HOUSEHOLDS', 'MEDIAN_INCOME']

        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
            ])

        # Separate features and target variable
        df_features = df.drop(['MEDIAN_HOUSE_VALUE', 'AGENCY'], axis=1)
        y = df['MEDIAN_HOUSE_VALUE'].values

        # Apply the preprocessing pipeline
        df_features_transformed = preprocessor.fit_transform(df_features)

        # Get column names after transformation
        all_columns = numeric_features

        # Create DataFrame with named columns
        df_features_transformed = pd.DataFrame(df_features_transformed, columns=all_columns)

        return df_features_transformed, y
    except Exception as e:
        logging.error(f"Error cleaning data and splitting features and target: {e}")
        return None, None


def load_data_from_db(table_name='transformed_data', db_name='database.db'):
    try:
        logging.info(f'Loading data from {table_name} table in {db_name} database...')
        conn = sqlite3.connect(db_name)
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
        conn.close()
        return df
    except Exception as e:
        logging.error(f"Error loading data from database: {e}")
        return None


def save_predictions_to_db(y_pred, param):
    try:
        logging.info(f'Saving predictions to {param} table...')
        conn = sqlite3.connect('database.db')
        y_pred_df = pd.DataFrame(y_pred, columns=['MEDIAN_HOUSE_VALUE'])
        y_pred_df.to_sql(param, conn, if_exists='replace', index=False)
        conn.close()
    except Exception as e:
        logging.error(f"Error saving predictions to database: {e}")


if __name__ == '__main__':
    try:
        df = pd.read_csv('data_testing_testing.csv')

        # Creating SQLite database and tables
        create_db_and_tables()

        # Cleaning data and splitting features and target
        df_features, y = clean_data_and_split_features_and_target(df)

        # Saving transformed data to SQLite database
        if df_features is not None:
            save_transformed_data_to_db(pd.DataFrame(df_features), 'transformed_data')

            # Loading transformed data from SQLite database
            X_train = load_data_from_db('transformed_data')

            # Load the model
            model = load_model('new_model.joblib')

            if model is not None:
                # Predict
                y_pred = predict(X_train, model)
                if y_pred is not None:
                    logging.info('First 5 predictions:')
                    logging.info(y_pred[:5])
                    save_predictions_to_db(y_pred, 'predictions')

            else:
                logging.error('Model loading failed.')
        else:
            logging.error('Feature transformation failed.')
    except Exception as e:
        logging.error(f"An unexpected error occurred in the main workflow: {e}")



