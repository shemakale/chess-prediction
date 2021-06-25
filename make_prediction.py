#import pandas as pd
#import numpy as np
import pickle
#import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import get_features_for_prediction as get_f


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


def logit(input_data):
	''' Fit logistic regression with whole dataframe '''
	X = input_data.drop(columns=['Date_Time', 'date_', 'my_result'])
	y = input_data.my_result
	clf = LogisticRegression()
	clf.fit(X, y)
	return clf


def predict_result(data_):
	''' Main function that makes all steps to predict result of the game
	1 - checking input data, 2 - scaling features, 3 - concatenating scaled and boolean features to one array,
	4 - extracting estimator keeping in file, 5 - predicting probability of my win by estimator
	Probabilit of my win in percents will return '''
	get_f.check_input_data(data_)
	#print(get_f.make_boolean_features(data_))
	#print(get_f.make_numeric_features(data_))
	#print(scale_features(make_numeric_features(data_), open_scaler('scaler.pkl')))
	scaled_X = scale_features(get_f.make_numeric_features(data_), open_scaler('scaler.pkl'))
	X_pred = concatenate_features(get_f.make_boolean_features(data_), scaled_X)
	#print(X_pred)
	clf = extract_clf_from_file(r'clf.pkl')
	#print(get_predicted_results(clf, X_pred))
	#print(predict_result(logit(open_games_base()), make_features(data_)))
	return get_predicted_results(clf, X_pred)


#data_ = ['Обычная игра', 'shahmatpatblog', 'shahmatpatblo1', '2021', '04', '18', '02', '14', '2239', '2366', 'Блиц']
if __name__ == '__main__':
	predict_result(data_)



"""
	column_names = ['Date_Time', 'date_', 'my_color', 'event', 'my_rating', 'rating_diff', \
	'time_control', 'part_of_day_afternoon', 'part_of_day_evening', 'part_of_day_morning', \
	'part_of_day_night', 'its_weekend', 'last_7days_games', 'last_7days_wins', \
	'last_7days_win_rate', 'count_today_games', 'win_today_games', 'win_rate_today_games']
	df = pd.DataFrame(columns=[column_names])

def create_df_from_pgn(file_path, col_names):
	'''
	creating dataframe from chess base file by .pgn format
	'''
	with open(file_path, 'r', encoding='utf-8') as f:
		text_file = f.read()
		data_dict = {} #создаю словарь признаков и через цикл заполняю его
		for col_name in col_names:
			pattern = r'(?<=\[' + col_name +  r' \")[^\"]+'
			data_dict[col_name] = re.findall(pattern, text_file)
	return pd.DataFrame(data=data_dict)

def count_rate(winned, played):
	'''
	Считает отношение количества побед к общему количеству игр, в случае, если игр не было сыграно, отношение принимается = 0
	'''
	try:
		answer = winned / played
	except ZeroDivisionError:
		answer = 0
	return answer.fillna(0).round(3)

def create_my_result(df):
	'''
	создание целевой переменной my_result, а также переменной y_color (цвет игры)
	'''
	target = pd.DataFrame()
	target['my_color'] = df.White.replace('shahmatpatblog', 1)
	target.loc[target.my_color != 1, 'my_color'] = 0
	target['my_color'] = target['my_color'].astype('int32')

	target['result'] = df['Result'].replace('1-0', 1).replace('0-1', 0).replace('1/2-1/2', 0.5)

	target['my_result'] = target.my_color + target.result
	target.my_result = target.my_result.replace([0, 2], 'win').replace(1, 'not_win').replace([0.5, 1.5], 'not_win')
	target.my_result = target.my_result.replace('win', 1).replace('not_win', 0)
	target['my_result'] = target['my_result'].astype('int32')
	return target.drop(columns=['result'])

def plot_auc_roc(y_true, y_pr):
	'''
	показать график ROC-кривой и посчитать площадь под ней
	'''
	fpr, tpr, thresholds = roc_curve(y_true, y_pr[:,1])
	roc_auc= auc(fpr, tpr)
	plt.figure(figsize=(8, 6))
	lw = 5
	plt.plot(fpr, tpr, color='darkorange',
			 lw=lw, label='ROC curve (area = %0.3f)' % roc_auc)
	plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver operating characteristic example')
	plt.legend(loc="lower right")
	plt.show()

def fit_model(X, y, clf, GridSearch_params):
	'''
	найти наилучшую модель с помощью GridSearchCV и вернуть её
	'''
	grid_search_cv = GridSearchCV(clf, GridSearch_params, cv=3, n_jobs=-1)
	grid_search_cv.fit(X, y)
	print('Найден лучший классификатор с параметрами {0} и score = {1}'.format(grid_search_cv.best_params_, grid_search_cv.best_score_))
	return grid_search_cv.best_estimator_
	"""