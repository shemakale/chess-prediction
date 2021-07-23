# This file contains functions which work with input information about game to predict
import pandas as pd
import numpy as np
import datetime


def check_input_data(input_data):
	"""	Checking data which i enter in totalizer GUI, counting empty fields """
	empty_count = list(map(len, input_data)).count(0)
	if empty_count == 0:
		print('Пустых полей во входных данных нет')
	else:
		print('Заполните все поля входных данных!')


def open_games_base():
	''' Opening base of chess games further for making features '''
	df = pd.read_csv('all_games_df.csv', sep=',', index_col=0, parse_dates=['Date_Time'])
	df[['Date_Time']] = df[['Date_Time']].applymap(lambda x: x.date())
	return df


def last_7days_statistics(df, input_data):
	''' Making dataframe with statistics of last 7 days since every game moment. 
	This is for making following features: last_7days_games, last_7days_wins, last_7days_win_rate '''
	input_date = get_date_time(input_data)
	condition_1 = df['Date_Time'] <= input_date - datetime.timedelta(days=1)
	condition_2 = df['Date_Time'] >= input_date - datetime.timedelta(days=8)
	last_7days_games = df.loc[(condition_1) & (condition_2), ['Date_Time', 'my_result']]
	return last_7days_games


def today_statistics(df, input_data):
	''' Making dataframe with statistics of game day. 
	This is for making following features: today_games, today_wins, today_win_rate '''
	input_date = get_date_time(input_data)
	today_games = df.loc[df['Date_Time'] == input_date, ['Date_Time', 'my_result']]
	return today_games


def get_date_time(input_data):
	""" Get date and time from input data in DateTime format """
	game_date_time = '-'.join((input_data[3], input_data[4], input_data[5])) + ' '\
		+ ':'.join((input_data[6], input_data[7], '00'))
	return datetime.datetime.strptime(game_date_time, '%Y-%m-%d %H:%M:%S').date()


def get_color(input_data):
	""" Get my color in this game. 1 - WHITE, 0 - BLACK """
	if 'shahmatpatblog' == input_data[1]:
		return 1
	elif 'shahmatpatblog' == input_data[2]:
		return 0


def get_event(input_data):
	""" Get type of this game (usual or tournament) """
	if 'Обычная игра' == input_data[0]:
		return 0
	elif 'Турнирная игра' == input_data[0]:
		return 1


def get_rating(input_data):
	""" Get my current rating """
	if get_color(input_data):
		return int(input_data[8])
	else:
		return int(input_data[9])


def get_rating_diff(input_data):
	""" Get rating difference between me and opponent """
	diff = int(input_data[8]) - int(input_data[9])
	if get_color(input_data):
		return diff
	else:
		return -diff


def get_time_control(input_data):
	""" Get type control of this game (blitz or bullet) """
	if 'Блиц' == input_data[10]:
		return 1
	elif 'Пуля' == input_data[10]:
		return 0


def get_its_afternoon(input_data):
	""" Is this game played in the afternoon hours (since 8 to 12 UTC)? """
	if int(input_data[6]) >= 8 and int(input_data[6]) <= 11:
		return 1
	else:
		return 0


def get_its_evening(input_data):
	""" Is this game played in the evening hours (since 12 to 16 UTC)? """
	if int(input_data[6]) >= 12 and int(input_data[6]) <= 15:
		return 1
	else:
		return 0


def get_its_morning(input_data):
	""" Is this game played in the morning hours (since 0 to 8 UTC)? """
	if int(input_data[6]) >= 0 and int(input_data[6]) <= 7:
		return 1
	else:
		return 0


def get_its_night(input_data):
	""" Is this game played in the night hours (since 16 to 24)? """
	if int(input_data[6]) >= 16 and int(input_data[6]) <= 24:
		return 1
	else:
		return 0


def get_its_weekend(input_data):
	""" Is this weekend? """
	if get_date_time(input_data).isoweekday() in (6, 7):
		return 1
	else:
		return 0


def get_last_7days_games(df_7days):
	""" Get how many games i played last 7 days """
	return df_7days.my_result.count()


def get_last_7days_wins(df_7days):
	""" Get how many games i won last 7 days """
	return df_7days.my_result.sum()


def get_last_7days_win_ratio(df_7days):
	""" Get ratio between amount of wins and overall amount of games last 7 days """
	if df_7days.my_result.sum():
		answer = np.divide(get_last_7days_wins(df_7days), get_last_7days_games(df_7days))
		return round(answer, 3)
	else:
		return 0.0


def get_today_games(today_games):
	""" Get how many games i played this day before """
	return today_games.my_result.count()


def get_today_wins(today_games):
	""" Get how many games i won this day before """
	return today_games.my_result.sum()


def get_today_win_ratio(today_games):
	""" Get ratio between amount of wins and overall amount of games this day before """
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
	num_features.append( get_last_7days_win_ratio(df_7days))
	num_features.append(get_today_games(df_today))
	num_features.append(get_today_wins(df_today))
	num_features.append(get_today_win_ratio(df_today))
	return np.reshape(num_features, (1, len(num_features)))