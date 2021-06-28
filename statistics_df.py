#import pandas as pd
import datetime


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