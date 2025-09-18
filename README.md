IMDb Movie Rating Prediction
This project aims to predict the rating of movies from the 'IMDb Movies India' dataset using machine learning. The model analyzes various features of a movie, such as its genre, director, actors, and duration, to estimate its rating.
📜 Description
The goal is to analyze historical movie data and develop a regression model that accurately predicts movie ratings. This project involves data cleaning, preprocessing, feature engineering, and training a machine learning model to understand the factors that influence movie ratings.
💾 Dataset
The dataset used is IMDb Movies India.csv, sourced from Kaggle. It contains information about thousands of Indian movies.
Features include:
Name: The name of the movie.
Year: The year of release.
Duration: The duration in minutes.
Genre: The primary genre of the movie.
Rating: The IMDb rating (target variable).
Votes: The number of votes the movie received.
Director: The director's name.
Actor 1, Actor 2, Actor 3: The names of the main actors.
⚙️ Methodology
Data Loading: The dataset is loaded using pandas.
Data Cleaning and Preprocessing:
The Name column is dropped as it is a unique identifier.
Missing values in categorical columns are filled with 'Unknown'.
Rows with missing Rating, Duration, Year, or Votes are dropped.
Year, Duration, and Votes columns are cleaned and converted to a numerical format.
Feature Engineering:
Categorical features (Genre, Director, Actors) are converted into numerical format using LabelEncoder from scikit-learn.
Model Training:
The data is split into training (80%) and testing (20%) sets.
A RandomForestRegressor model is trained on the training data.
Model Evaluation:
The model's performance is evaluated on the test set using the following metrics:
Mean Absolute Error (MAE)
Mean Squared Error (MSE)
R-squared (R²)
Visualization:
A bar chart is generated to show the importance of each feature in the model's predictions.
A scatter plot compares the actual ratings to the predicted ratings.



📈 Results
The model's performance is printed to the console upon execution. The feature importance analysis reveals that the number of Votes, Year of release, and Duration are the most significant factors in predicting a movie's rating. The visualization plots provide a clear picture of the model's accuracy and key drivers.
