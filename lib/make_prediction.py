# This file contains functions to make a prediction for game the information about that you entered in GUI
import numpy as np
import pickle
import get_features_for_prediction as get_f
from create_df import TrainingDataFrame, get_names
from fit_clf import FittedLogit


def open_scaler(path_to_file):
	''' open scaler from pickle file '''
	with open(path_to_file, 'rb') as f:
		scaler = pickle.load(f)
	return scaler


def scale_features(X, fitted_scaler):
	''' Scales list of input numeric features '''
	scaler = fitted_scaler
	return scaler.transform(X)


def concatenate_features(bool_f, num_f):
	''' Concatenate array of boolean features
	and scaled numeric features to one array for further prediction '''
	return np.concatenate((bool_f, num_f), axis=1)


def extract_clf_from_file(path_to_file):
	''' Extract fitted estimator from pickle file. Returns estimator '''
	with open(path_to_file, 'rb') as f:
		clf = pickle.load(f)
	return clf


def get_predicted_results(clf, game):
	''' Predict probability of my win for this game. Returns probability in percents '''
	y_predict_proba = clf.predict_proba(game)
	return round(y_predict_proba[:, 1][0] * 100, 1)


def predict_result(game_for_pred):
	''' Main function that makes all steps to predict result of the game. To realize that, we need saved
	estimator (clf.pkl), saved scaler (scaler.pkl) 
	1 - checking input data, 2 - scaling features, 3 - concatenating scaled and boolean features to one array,
	4 - extracting estimator keeping in file, 5 - predicting probability of my win by estimator
	Probability of my win in percents will be returned '''
	clf = extract_clf_from_file(r'../work/clf.pkl')
	get_f.check_input_data(game_for_pred)
	scaled_X = scale_features(get_f.make_numeric_features(game_for_pred), open_scaler(r'../work/scaler.pkl'))
	X_pred = concatenate_features(get_f.make_boolean_features(game_for_pred), scaled_X)
	
	return get_predicted_results(clf, X_pred)


if __name__ == '__main__':
	predict_result(data_)
