# Overview of Otto's data engineer assignment for Telekom

Due to issues with the provided model, I developed a new model in `new_training.py`, which generates the file `new_model.joblib`. Although this new model does not provide accurate predictions,the project coordinator advised me to create a simple model for generating predictions for further processing. The effectiveness of the model is not crucial for the tasks.

---

### Running the Main Application

To initiate the process, execute the `main.py` file. Here’s what it does:

1. Reads data into a DataFrame from the file `data_testing_testing.csv`. In this file are stored 10% of the original data not used for training or testing previously.
2. Creates a SQLite database named `database.db`.
3. Creates two tables in DB: `transformed_data` and `predictions`.
4. Cleans and splits the data into features and targets using the custom function `clean_data_and_split_features_and_target`.
5. Loads the transformed (cleaned) data into the SQLite database. Features are stored in the `transformed_data` table.
4. Extracts data from the database (from the `transformed_data` table) and loads it into the `X_train` DF.
5. Loads the model from the file `new_model.joblib`.
6. Generates predictions by passing `X_train` through the model and stores the results in DF `y_pred`.
5. Saves the predictions to the database using the custom function `save_predictions_to_db`, specifically in the single-column table `predictions`.

---

### Training the Model


If you need to retrain the model, run `new_training.py`. Executing this file will generate a new `new_model.joblib` file containing the updated model. The training process retains 10% of the original dataset from the file `data_testing.csv` for testing purposes. This step is optional, as you can use the pre-trained model in `new_model.joblib`.

---

### Preprocessing the Data

The script preprocesses data by first loading and cleaning it, dropping any rows with missing values. Numeric features are standardized using StandardScaler within a pipeline to ensure consistent scaling across features. The transformed data is then saved into an SQLite database, making it readily accessible for model predictions. Finally, predictions are saved back to the database for further use

---

### API Access Points with Flask

I created an API access point using Flask, located in `flask_api.py`, which includes a simple GET endpoint that returns JSON data for houses with a specified median income, because we don't have an ID column in the dataset.

The Flask application retrieves data from the file `data_testing_testing.csv`. The data in this file is accessible through the access point `get_house_info`. 

![image](https://github.com/user-attachments/assets/054ea9ca-c548-43d9-9e18-fecee2e6b250)

This is just a demonstration, but a better usage example would be a case where I retrieve information from an SQLite database and distribute it to other systems via an API. Since the SQLite database is actually just a single file, it is not recommended to perform extractions from multiple devices. However, an alternative solution like this could allow us to retrieve information from multiple devices.

To run the Flask application:

1. Execute the `flask_api.py` file.
2. The Flask server will run locally on `localhost`.
3. Use a web browser or `curl` to access the endpoint, for example: `http://127.0.0.1:5000/get_house_info?median_income=2.4615`


---

### Logging and Exception Handling

The logging and exception handling mechanisms are used to capture and record any errors, warnings, or informational messages that occur during the execution of the code. This allows for better debugging and troubleshooting of issues, as well as providing a record of system events and errors for auditing and compliance purposes. Additionally, logging and exception handling can help to prevent unexpected behavior and crashes by catching and handling errors before they cause the program to terminate.

![image](https://github.com/user-attachments/assets/2ad5b0d4-625d-48ab-921e-1acb9d57fd49)

