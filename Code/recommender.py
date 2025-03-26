import pandas as pd
from surprise import Dataset, Reader, dump
from surprise.model_selection import train_test_split, GridSearchCV
from surprise.prediction_algorithms.baseline_only import BaselineOnly
from surprise.prediction_algorithms import SVD, CoClustering, BaselineOnly, SlopeOne
from surprise import accuracy
import numpy as np
import os

np.random.seed(42)

# Load data
def load_data(data_path, filter_date=True):
    df = pd.read_csv(data_path, parse_dates=['start_time'])
    # filter on recent data only. Use data from the last 6 months
    last_date = df['start_time'].max()
    
    if filter_date:
        six_months_ago = last_date - pd.DateOffset(months=6)
        df = df[df['start_time'] >= six_months_ago]
    
    # filter out exercises that have not been done more than once by the same user
    df = df.groupby(['username', 'exercise_title']).filter(lambda x: len(x) > 3)
    
    # Use only exercises that at least 10 users have done
    df = df.groupby('exercise_title').filter(lambda x: len(x) >= 10)
    return df

# Create a user-exercise matrix
def create_user_exercise_matrix(df):
    user_exercise_matrix = df.pivot_table(index='username', columns='exercise_title', values='estimated_1rm', aggfunc='max')
    return user_exercise_matrix

# Normalize the estimated 1RM values
def normalize_estimated_1rm(df):
    df.loc[:, 'estimated_1rm'] = df['estimated_1rm'].apply(lambda x: np.log1p(x))
    return df

# Prepare the data for the recommender system
def prepare_recommender_data(df):
    reader = Reader(rating_scale=(df['estimated_1rm'].min(), df['estimated_1rm'].max()))
    data = Dataset.load_from_df(df[['username', 'exercise_title', 'estimated_1rm']], reader)
    return data

# Train the recommender system with hyperparameter tuning
def train_recommender(data, model):
    # define the parameter grid
    if model == CoClustering:
        param_grid = {
            'n_cltr_u': [3, 5, 10],
            'n_cltr_i': [3, 5, 10],
            'n_epochs': [10, 20, 30]
        }
    elif model == SVD:
        param_grid = {
            'n_factors': [50, 100, 150],
            'lr_all': [0.002, 0.005, 0.01],
            'reg_all': [0.02, 0.05, 0.1]
        }
    else:
        param_grid = {}
        
    # perform grid search
    gs = GridSearchCV(model, param_grid, measures=['rmse', 'mae', 'fcp'], cv=5, 
                      n_jobs=-1, refit=True, joblib_verbose=0)
    gs.fit(data)
    
    best_params = gs.best_params['rmse']
    print(f"Best parameters: {best_params}")
        
    scores = gs.best_score
    
    # convert rmse and mae back to original scale
    scores['rmse'] = np.expm1(scores['rmse'])
    scores['mae'] = np.expm1(scores['mae'])
    
    # round all scores to 3 decimal places
    scores = {k: round(v, 3) for k, v in scores.items()}
    print(f"Scores: {scores}")
    
    # save the best model    
    algo = gs.best_estimator['rmse']
    os.makedirs('../Models', exist_ok=True)
    dump.dump(f'../Models/{model.__name__}.pkl', algo=algo)
    return algo, gs.best_score['rmse']

# Generate recommendations
def generate_recommendations(algo, user_id, user_exercise_matrix):
    user_data = user_exercise_matrix.loc[user_id]
    user_unrated_exercises = user_data[user_data.isna()].index
    recommendations = []
    for exercise in user_unrated_exercises:
        est_rating = algo.predict(user_id, exercise).est
        recommendations.append((exercise, est_rating))
    recommendations.sort(key=lambda x: x[1], reverse=True)
    return recommendations

def generate_full_matrix(algo, user_exercise_matrix): 
    # Fill in missing values
    for user in user_exercise_matrix.index:
        for exercise in user_exercise_matrix.columns:
            if pd.isna(user_exercise_matrix.at[user, exercise]):
                est_rating = algo.predict(user, exercise).est
                user_exercise_matrix.at[user, exercise] = np.expm1(est_rating)  # Convert back from log scale
                
    return user_exercise_matrix

# Main function
def main(data_path):
    df = load_data(data_path)
    user_id = df['username'].sample(1).values[0]
    print(f"Generating recommendations for user {user_id}...")

    df = normalize_estimated_1rm(df)
    user_exercise_matrix = create_user_exercise_matrix(df)
    
    recommender_data = prepare_recommender_data(df)
    models = [BaselineOnly, SlopeOne, SVD, CoClustering]
    algorithms = []
    rmse_scores = []
    for model in models:
        print(f"Training the recommender system using {model.__name__}...")
        algo, rmse = train_recommender(recommender_data, model)
        rmse_scores.append(rmse)
        algorithms.append(algo)
        
    best_model = models[np.argmin(rmse_scores)]
    algo = algorithms[np.argmin(rmse_scores)]
    print(f"Best model: {best_model.__name__}")


    
    recommendations = generate_recommendations(algo, user_id, user_exercise_matrix)
    
    print(f"Top recommendations for user {user_id}:")
    for exercise, rating in recommendations:
        print(f"Exercise: {exercise}, Estimated Rating: {np.expm1(rating):.2f}")

if __name__ == "__main__":
    data_path = "../Data/workout_data.csv"
    main(data_path)
