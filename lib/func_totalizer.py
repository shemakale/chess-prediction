import re
import pandas as pd
import numpy as np
import pickle
#import matplotlib.pyplot as plt
import datetime
#from sklearn.model_selection import GridSearchCV
#from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import get_features_for_prediction as get_f

"""def check_input_data(input_data):
	'''	Checking data which i enter in totalizer GUI, counting empty fields '''
	empty_count = list(map(len, input_data)).count(0)
	if empty_count == 0:
		print('Пустых полей во входных данных нет')
	else:
		print('Заполните все поля входных данных!')


def open_games_base():
	df = pd.read_csv('all_games_df.csv', sep=',', index_col=0, parse_dates=['Date_Time'])
	df[['Date_Time']] = df[['Date_Time']].applymap(lambda x: x.date())
	return df
	#return '{:02}'.format(int(input_data[4]))


def last_7days_statistics(df, input_data):
	input_date = get_date_time(input_data)
	condition_1 = df['Date_Time'] <= input_date - datetime.timedelta(days=1)
	condition_2 = df['Date_Time'] >= input_date - datetime.timedelta(days=8)
	last_7days_games = df.loc[(condition_1) & (condition_2), ['Date_Time', 'my_result']]
	return last_7days_games


def today_statistics(df, input_data):
	input_date = get_date_time(input_data)
	today_games = df.loc[df['Date_Time'] == input_date, ['Date_Time', 'my_result']]
	return today_games


def get_date_time(input_data):
	''' Get date and time from input data in DateTime format '''
	game_date_time = '-'.join((input_data[3], input_data[4], input_data[5])) + ' '\
		+ ':'.join((input_data[6], input_data[7], '00'))
	return datetime.datetime.strptime(game_date_time, '%Y-%m-%d %H:%M:%S').date()

def get_color(input_data):
	''' Get my color in this game. 1 - WHITE, 0 - BLACK '''
	if 'shahmatpatblog' == input_data[1]:
		return 1
	elif 'shahmatpatblog' == input_data[2]:
		return 0

def get_event(input_data):
	if 'Обычная игра' == input_data[0]:
		return 0
	elif 'Турнирная игра' == input_data[0]:
		return 1

def get_rating(input_data):
	if get_color(input_data):
		return int(input_data[8])
	else:
		return int(input_data[9])

def get_rating_diff(input_data):
	diff = int(input_data[8]) - int(input_data[9])
	if get_color(input_data):
		return diff
	else:
		return -diff

def get_time_control(input_data):
	if 'Блиц' == input_data[10]:
		return 1
	elif 'Пуля' == input_data[10]:
		return 0

def get_its_afternoon(input_data):
	if int(input_data[6]) >= 8 and int(input_data[6]) <= 11:
		return 1
	else:
		return 0

def get_its_evening(input_data):
	if int(input_data[6]) >= 12 and int(input_data[6]) <= 15:
		return 1
	else:
		return 0

def get_its_morning(input_data):
	if int(input_data[6]) >= 0 and int(input_data[6]) <= 7:
		return 1
	else:
		return 0


def get_its_night(input_data):
	if int(input_data[6]) >= 16 and int(input_data[6]) <= 24:
		return 1
	else:
		return 0


def get_its_weekend(input_data):
	if get_date_time(input_data).isoweekday() in (6, 7):
		return 1
	else:
		return 0


def get_last_7days_games(df_7days):
	return df_7days.my_result.count()


def get_last_7days_wins(df_7days):
	return df_7days.my_result.sum()


def get_last_7days_win_rate(df_7days):
	if df_7days.my_result.sum():
		answer = np.divide(get_last_7days_wins(df_7days), get_last_7days_games(df_7days))
		return round(answer, 3)
	else:
		return 0.0


def get_today_games(today_games):
	return today_games.my_result.count()


def get_today_wins(today_games):
	return today_games.my_result.sum()


def get_today_win_rate(today_games):
	if today_games.my_result.sum():
		answer = np.divide(get_today_wins(today_games), get_today_games(today_games))
		return round(answer, 3)
	else:
		return 0.0


def make_boolean_features(input_data):
	''' Calculate features for input data. Returns a list of features'''
	bool_features = []
	bool_features.append(get_color(input_data))
	bool_features.append(get_event(input_data))
	bool_features.append(get_time_control(input_data))
	bool_features.append(get_its_afternoon(input_data))
	bool_features.append(get_its_evening(input_data))
	bool_features.append(get_its_morning(input_data))
	bool_features.append(get_its_night(input_data))
	bool_features.append(get_its_weekend(input_data))
	return np.reshape(bool_features, (1, len(bool_features)))


def make_numeric_features(input_data):
	''' Calculate numeric features for input data. 
	These features will be scaled in scale_features() function. 
	Returns a numpy array of numeric features with shape (1, len(num_features))'''
	num_features = []
	df_games = open_games_base()
	df_7days = last_7days_statistics(df_games, input_data)
	df_today = today_statistics(df_games, input_data)
	num_features.append(get_rating(input_data))
	num_features.append(get_rating_diff(input_data))
	num_features.append( get_last_7days_games(df_7days))
	num_features.append(get_last_7days_wins(df_7days))
	num_features.append( get_last_7days_win_rate(df_7days))
	num_features.append(get_today_games(df_today))
	num_features.append(get_today_wins(df_today))
	num_features.append(get_today_win_rate(df_today))
	return np.reshape(num_features, (1, len(num_features)))
"""

def open_scaler(path_to_file):
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
	''' Predict result and probability of my win. Returns list of 2 elements: predicted result and probability of my win '''
	#X_pred = np.reshape(game, (1, len(game)))
	#y_pred = clf.predict(game)
	y_predict_proba = clf.predict_proba(game)
	return round(y_predict_proba[:, 1][0] * 100, 1)


def logit(input_data):
	X = input_data.drop(columns=['Date_Time', 'date_', 'my_result'])
	y = input_data.my_result
	clf = LogisticRegression()
	clf.fit(X, y)
	return clf


def predict_result(data_):
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