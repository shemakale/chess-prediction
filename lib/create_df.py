# In this file a dataframe is creating from chessbase file (PGN format).
# Training dataframe is saving as .CSV file. Scaler is saving as PKL file
# They are used to make features for the game from GUI 
import re
import pandas as pd
import numpy as np
import datetime
from sklearn.preprocessing import StandardScaler
import pickle


def get_names():
	''' Parcing file with nick_name of user and path to chessbase.
	Returns tuple where 0th element - nickname of user (str), 1st element - path to chessbase (str) '''
	with open(r'../base/user_base_info.txt', 'r', encoding='utf-8') as f:
		text = f.readlines()
	nickname = re.match('.+(?=\\n)', text[0]).group()
	path_to_base = text[1]
	return nickname, path_to_base


class TrainingDataFrame:
	""" Dataframe that constructs from .PGN chessbase file. To create training dataframe use create_training_df() method """
	time_controls = {'bullet': ['60+0', '60+1', '120+0', '120+1'],
					'blitz': ['180+0', '180+1', '180+2', '300+0', '300+2', '300+3']}
	non_tournament_events = ['Rated Blitz game', 'Rated Bullet game', 'Casual Blitz game']


	def __init__(self, file_path):
		self.file_path = file_path
		self.raw_df = self.create_df_from_pgn(file_path)


	def create_df_from_pgn(self, file_path):
		'''	Creating dataframe from chess base file by .pgn format '''
		col_names = ['Event', 'White', 'Black', 'Result', 'UTCDate', \
					'UTCTime', 'WhiteElo', 'BlackElo', 'TimeControl']
		with open(file_path, 'r', encoding='utf-8') as f:
			text_file = f.read()
			data_dict = {} # dictionary with columns as keys
			for col_name in col_names:
				pattern = r'(?<=\[' + col_name +  r' \")[^\"]+'
				data_dict[col_name] = re.findall(pattern, text_file)
		return pd.DataFrame(data=data_dict)


	def calculate_ratio(self, winned, played):
		'''	Calculating ratio between count of winned games and count of all games. 
		If no games played then ratio = 0.0	'''
		try:
			answer = winned / played
		except ZeroDivisionError:
			answer = 0
		return answer.fillna(0).round(3)


	#Functions below create features for training dataframe
	def get_datetime(self, df):
		'''	Create feature of date and time'''
		Date_Time = pd.Series(data=(df['UTCDate'] + ' ' + df['UTCTime']), name='Date_Time')
		Date_Time = Date_Time.apply(lambda x: datetime.datetime.strptime(x, '%Y.%m.%d %H:%M:%S'))
		return Date_Time


	def get_date(self, df):
		'''	Create feature of date '''
		date_ = pd.Series(data=(df['UTCDate']), name='date_')
		date_ = date_.apply(lambda x: datetime.datetime.strptime(x, '%Y.%m.%d'))
		return date_


	def get_my_result(self, df):
		''' Create my_result - target variable. 0 - I lost, 1 - I won. This is target variable (y)! '''
		my_result = pd.Series(data=df['Result'].map({'1-0': 1, '0-1': 0, '1/2-1/2': 0.5}))
		my_result = my_result + self.get_my_color(df)
		my_result = my_result.replace([0.0, 2.0], 'win').replace([0.5, 1.0, 1.5], 'not_win')
		my_result = my_result.map({'win': 1, 'not_win': 0}).rename("my_result")
		return my_result.astype('uint8')


	def get_my_color(self, df):
		''' Color of my pieces. 0 - I play black, 1 - I play white '''
		name = get_names()[0]
		my_color = pd.Series(data=df.White.map({name: 1}), name='my_color')
		my_color = my_color.replace(np.NaN, 0)
		return my_color.astype('uint8')


	def get_event(self, df):
		''' This feature shows type of the game. 0 - not tournament game, 1 - tournament game '''
		event = df.Event.replace(TrainingDataFrame.non_tournament_events, 0)
		event.loc[event != 0] = 1
		return event.astype('uint8')


	def get_my_rating(self, df):
		''' My current rating in Lichess '''
		name = get_names()[0]
		my_rating = pd.Series(data=df['WhiteElo'].where(df['White'] == name, df['BlackElo']),
					name='my_rating')
		return my_rating.astype('uint16')


	def get_rating_diff(self, df):
		''' Rating difference between me and opponent '''
		diff = df['WhiteElo'].astype('int16') - df['BlackElo'].astype('int16')
		rating_diff = pd.Series(data=diff, name='rating_diff')
		rating_diff = rating_diff.where(self.get_my_color(df) == 1, -rating_diff)
		return rating_diff


	def get_time_control(self, df):
		''' Time control of the game. 0 - bullet, 1 - blitz. '''
		time_control = pd.Series(data=df['TimeControl'], name='time_control')
		time_control = time_control.replace(TrainingDataFrame.time_controls['bullet'], 0) \
                                    .replace(TrainingDataFrame.time_controls['blitz'], 1)
		return time_control.astype('uint8')


	def get_parts_of_day(self, df):
		''' Time of the day when game was played (afternoon, evening, morning, night)
		It returns a DataFrame with 4 columns '''
		hour_of_game = self.get_datetime(df).apply(lambda z: z.time().hour + 4).astype('uint8')
		hour_of_game.rename("part_of_day", inplace=True)
		parts_of_day = hour_of_game.to_frame().replace([4, 5, 6, 7, 8, 9, 10, 11], 'morning') \
                                    .replace([12, 13, 14, 15], 'afternoon').replace([16, 17, 18, 19], 'evening') \
                                    .replace([20, 21, 22, 23, 24, 25, 26], 'night')
		return pd.get_dummies(data=parts_of_day)


	def get_its_weekend(self, df):
		''' The game was played in weekend (1) or not (0) '''
		its_weekend = self.get_datetime(df).apply(lambda x: x.isoweekday()).astype('uint8')
		its_weekend.rename("its_weekend", inplace=True)
		its_weekend = its_weekend.replace([1, 2, 3, 4, 5], 0).replace([6, 7], 1).astype('uint8')
		return its_weekend


	def results_and_dates_df(self, df):
		''' Creates dataframe with 3 columns: Date_Time, date_, my_result.
		It is for creating dataframes with statistics for last 7 days and statistics for this day
		(create_stats_df_7days() and create_stats_df_today() functions) '''
		Date_Time, time_, my_result = self.get_datetime(df), self.get_date(df), self.get_my_result(df)
		results_and_dates = pd.DataFrame(data=np.array([Date_Time, time_, my_result]).T,
								columns=[Date_Time.name, time_.name, my_result.name])
		return results_and_dates


	def get_7days_stats(self, df):
		''' Creates dataframe with last 7 days statistics. 
		It is for creating following features: last_7days_games, last_7days_wins, last_7days_win_ratio'''
		sorted_df = self.results_and_dates_df(df).sort_values(by='Date_Time').set_index('date_')
		last_7d_stats = sorted_df[['my_result']].resample('d').count().rolling(8, min_periods=1).sum() -\
						sorted_df[['my_result']].resample('d').count()
		my_wins = sorted_df[['my_result']].loc[ sorted_df['my_result'] == 1]
		last_7d_wins = my_wins.resample('d').count().rolling(8, min_periods=1).sum() - my_wins.resample('d').count()
		last_7d_stats = last_7d_stats.merge(last_7d_wins, left_index=True, right_index=True, how='inner')
		last_7d_stats = last_7d_stats.reset_index().rename(columns={'my_result_x': 'last_7days_games',
																'my_result_y': 'last_7days_wins'})
		last_7d_stats['last_7days_win_ratio'] = self.calculate_ratio(last_7d_stats['last_7days_wins'], 
																	last_7d_stats['last_7days_games'])
		return last_7d_stats


	def get_this_day_stats(self, df):
		''' Creates dataframe with "day of game" statistics. 
		It is for creating following features: count_today_games, win_today_games, this_day_win_ratio'''
		dd = self.results_and_dates_df(df)
		list_count_today_games, list_count_today_wins= [], []
		for i, row in dd.iterrows():
			# Series with games for every day:
			condition_game = (dd['Date_Time'] < row[0]) & (dd['date_'] == row[1])
			subseries_games = dd[['Date_Time']].loc[condition_game]
			# Series with winned games for every day:
			condition_win = (dd['Date_Time'] < row[0]) & (dd['date_'] == row[1]) & (dd['my_result'] == 1)
			subseries_wins = dd[['Date_Time']].loc[condition_win]
			list_count_today_games.append(subseries_games.shape[0]) #count games for every day and put into a list
			list_count_today_wins.append(subseries_wins.shape[0]) #count wind for every day and put into an another list
		this_day_stats = pd.DataFrame(data=np.array((list_count_today_games, list_count_today_wins)).T,
									columns=['count_today_games', 'win_today_games']) #dataframe with stats for every day
		this_day_stats['this_day_win_ratio'] = self.calculate_ratio(this_day_stats['win_today_games'], 
																	this_day_stats['count_today_games'])
		return this_day_stats


	def scale_df(self, df):
		''' Get dataframe with scaled features. Scaling is not applied for boolean features (uint8 type) '''
		features_for_scaling = df.select_dtypes(exclude=['uint8', 'datetime64'])
		scaler = StandardScaler()
		scaled_arr = scaler.fit_transform(features_for_scaling)
		with open(r'../work/scaler.pkl', 'wb') as f:
			pickle.dump(scaler, f)
		scaled_df = pd.DataFrame(data=scaled_arr, columns=features_for_scaling.columns)
		resulted_df = df.drop(columns=features_for_scaling.columns)
		resulted_df = pd.merge(left=resulted_df, right=scaled_df, left_index=True, right_index=True, how='inner')
		return resulted_df


	def create_training_df(self, scale=True):
		''' Merges all features in one dataframe. If 'scale' flag is True dataframe will be scaling.
		Saves resulting dataframe as "all_games_df.csv" in the same directory '''
		full_df = pd.DataFrame() #dataframe for training
		full_df['Date_Time'] = self.get_datetime(self.raw_df)
		full_df['date_'] = self.get_date(self.raw_df)
		full_df['my_result'] = self.get_my_result(self.raw_df)
		full_df['my_color'] = self.get_my_color(self.raw_df)
		full_df['event'] = self.get_event(self.raw_df)
		full_df['my_rating'] = self.get_my_rating(self.raw_df)
		full_df['rating_diff'] = self.get_rating_diff(self.raw_df)
		full_df['time_control'] = self.get_time_control(self.raw_df)
		full_df = full_df.merge(right=self.get_parts_of_day(self.raw_df), left_index=True, right_index=True, how='inner')
		full_df['its_weekend'] = self.get_its_weekend(self.raw_df)
		full_df['rating_diff'] = self.get_rating_diff(self.raw_df)
		full_df['rating_diff'] = self.get_rating_diff(self.raw_df)
		full_df = full_df.merge(right=self.get_7days_stats(self.raw_df), on='date_', how='inner')
		full_df = full_df.merge(right=self.get_this_day_stats(self.raw_df), left_index=True, right_index=True, how='inner')
		if scale:
			full_df = self.scale_df(full_df)
		full_df.to_csv(r"../work/all_games_df.csv") #saving dataframe as .CSV
		full_df.drop(columns=['Date_Time', 'date_'], inplace=True)
		return full_df


if __name__ == '__main__':
	#data = TrainingDataFrame('lichess_shahmatpatblog_2021-04-20.pgn')
	#print(data.create_training_df())
	print(get_names())