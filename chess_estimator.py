import re
import pandas as pd
import numpy as np
import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pickle


class TrainingDataFrame:
	time_controls = {'bullet': ['60+0', '60+1', '120+0', '120+1'],
					'blitz': ['180+0', '180+1', '180+2', '300+0', '300+2', '300+3']}
	non_tournament_events = ['Rated Blitz game', 'Rated Bullet game', 'Casual Blitz game']
	def __init__(self, file_path):
		self.file_path = file_path
		self.all_games = self.create_df_from_pgn(file_path)


	def create_df_from_pgn(self, file_path):
		'''
		creating dataframe from chess base file by .pgn format
		'''
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
		'''
		Calculating ratio between count of winned games and count of all games. 
		If no games played then ratio = 0.0
		'''
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
		''' 0 - I lost, 1 - I won '''
		my_result = pd.Series(data=df['Result'].map({'1-0': 1, '0-1': 0, '1/2-1/2': 0.5}))
		my_result = my_result + self.get_my_color(df)
		my_result = my_result.replace([0.0, 2.0], 'win').replace([0.5, 1.0, 1.5], 'not_win')
		my_result = my_result.map({'win': 1, 'not_win': 0}).rename("my_result")
		return my_result.astype('uint8')


	def get_my_color(self, df):
		''' 0 - I play black, 1 - I play white '''
		my_color = pd.Series(data=df.White.map({'shahmatpatblog': 1}), name='my_color')
		my_color = my_color.replace(np.NaN, 0)
		return my_color.astype('uint8')


	def get_event(self, df):
		''' 0 - not tournament game, 1 - tournament game '''
		event = df.Event.replace(TrainingDataFrame.non_tournament_events, 0)
		event.loc[event != 0] = 1
		return event.astype('uint8')


	def get_my_rating(self, df):
		''' My current rating in Lichess '''
		my_rating = pd.Series(data=df['WhiteElo'].where(df['White'] == 'shahmatpatblog', df['BlackElo']),
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

	def get_7days_stats(self, df):
		return last_7days_statistics(df)


	def results_and_dates_df(self, df):
		''' Creates dataframe with 3 columns: Date_Time, date_, my_result.
		It is for creating dataframes with statistics for last 7 days and statistics for this day
		(create_stats_df_7days() and create_stats_df_today() functions) '''
		Date_Time, time_, my_result = self.get_datetime(df), self.get_date(df), self.get_my_result(df)
		results_and_dates = pd.DataFrame(data=np.array([Date_Time, time_, my_result]).T,
								columns=[Date_Time.name, time_.name, my_result.name])
		return results_and_dates


	def create_stats_df_7days(self, df):
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


	def create_training_df(self):
		return self.create_stats_df_7days(self.all_games)


aas = TrainingDataFrame('lichess_shahmatpatblog_2021-04-20.pgn')
#games = aas.create_df_from_pgn('lichess_shahmatpatblog_2021-04-20.pgn')
print(aas.create_training_df())






#all_games = create_df_from_pgn('lichess_shahmatpatblog_2021-04-20.pgn')
#print(get_rating_diff(all_games))
#print(pd.DataFrame(data=np.array([get_datetime(all_games), get_date(all_games)]).T))
#print(pd.merge(get_datetime(all_games), get_date(all_games), left_index=True, right_index=True))



def create_X(df_train, df_test, for_test=False):
	X = pd.DataFrame()
	if for_test:
		df = df_test.append(df_train, ignore_index=True)
	else:
		df = df_train #for_test var defines whether we prepare train data or test data

	#Date_Time, date_
	X['Date_Time'] = df['UTCDate'] + ' ' + df['UTCTime']
	X[['Date_Time']] = X[['Date_Time']].applymap(lambda x: datetime.datetime.strptime(x, '%Y.%m.%d %H:%M:%S'))
	X[['date_']] = df[['UTCDate']].applymap(lambda x: datetime.datetime.strptime(x, '%Y.%m.%d'))

	#my_result
	X['my_result'] = create_series_y(df)

	#my_color
	X['my_color'] = df.White.replace('shahmatpatblog', 1)
	X.loc[X.my_color != 1, 'my_color'] = 0
	X['my_color'] = X['my_color'].astype('int64')

	#elo_diff
	X['elo_diff'] = df['WhiteElo'].astype('int64') - df['BlackElo'].astype('int64')
	X['elo_diff'] = X['elo_diff'].where(X['my_color'] == 1, -X['elo_diff'])

	#event
	X['event'] = df.Event.replace(['Rated Blitz game', 'Rated Bullet game', 'Casual Blitz game'], 0)
	X.loc[X.event != 0, 'event'] = 1
	X['event'] = X['event'].astype('int64')

	#time_control (0 - bullet, 1 - blitz)
	X['time_control'] = df.TimeControl.replace(['60+0', '60+1', '120+0', '120+1'], 0) \
									.replace(['180+0', '180+1', '180+2', '300+0', '300+2', '300+3'], 1)

	#1-Monday, 2-Tuesday ... 7-Sunday
	X[['day_of_week']] = X[['Date_Time']].applymap(lambda x: x.isoweekday())

	# hour of game (UTC time)
	X[['hour_of_game']] = X[['Date_Time']].applymap(lambda z: z.time().hour)

	#how many games i played and win last 7 days
	dated_X = X.sort_values(by='Date_Time').set_index('date_')
	last_7d_stats = dated_X[['my_result']].resample('d').count().rolling(8, min_periods=1).sum() - dated_X[['my_result']].resample('d').count()
	my_wins = dated_X[['my_result']].loc[ dated_X['my_result'] == 1]
	last_7d_wins = my_wins.resample('d').count().rolling(8, min_periods=1).sum() - my_wins.resample('d').count()
	last_7d_stats = last_7d_stats.merge(last_7d_wins, left_index=True, right_index=True, how='inner')
	last_7d_stats = last_7d_stats.reset_index().rename(columns={'my_result_x': 'last_7days_games',
																'my_result_y': 'last_7days_wins'})
	try:
		last_7d_stats['last_7days_win_rate'] = last_7d_stats.last_7days_wins / last_7d_stats.last_7days_games
	except ZeroDivisionError:
		last_7d_stats['last_7days_win_rate'] = 0
	last_7d_stats['last_7days_win_rate'] = last_7d_stats['last_7days_win_rate'].fillna(0).round(3)

	X = pd.merge(left=X, right=last_7d_stats, on='date_', how='inner')

	#how many games i played that day before
	dd = X[['Date_Time', 'date_', 'my_result']]
	list_count_today_games, list_count_today_wins= [], []
	for i, row in dd.iterrows():
		#print(row)
		subseries_games = dd[['Date_Time']].loc[(dd['Date_Time'] < row[0]) & (dd['date_'] == row[1])]
		subseries_wins = dd[['Date_Time']].loc[(dd['Date_Time'] < row[0]) & (dd['date_'] == row[1]) & dd['my_result'] == 1]
		list_count_today_games.append(subseries_games.shape[0])
		list_count_today_wins.append(subseries_wins.shape[0])
	df_count_today = pd.DataFrame(data=np.array((list_count_today_games, list_count_today_wins)).T,
								 columns=['count_today_games', 'win_today_games'])
	try:
		df_count_today['win_rate_today_games'] = df_count_today.win_today_games / df_count_today.count_today_games
	except ZeroDivisionError:
		df_count_today['win_rate_today_games'] = 0
	df_count_today['win_rate_today_games'] = df_count_today['win_rate_today_games'].fillna(0).round(3)
	X = pd.merge(left=X, right=df_count_today, left_index=True, right_index=True, how='inner')

	#my_rating (my current ELO rating)
	X['my_rating'] = df['WhiteElo'].where(df['White'] == 'shahmatpatblog', df['BlackElo']).astype('int32')

	#my stats for this beginning (number of games in this ECO, how many wins i have in this ECO)
	X = X.merge(statistics_eco(df)[['ECO_games', 'ECO_win_rate']], left_index=True, right_index=True, how='inner')
	return X[:df_test.shape[0]] if for_test else X

def create_my_result(df):
	'''
	create dataframe with my_color and my_result features
	'''
	target = pd.DataFrame()
	#вначале создаю столбец каким цветом я играю
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
	grid_search_cv = GridSearchCV(clf, GridSearch_params, cv=3, n_jobs=-1, verbose=3)
	grid_search_cv.fit(X, y)
	print('Найден лучший классификатор с параметрами {0} и score = {1}'.format(grid_search_cv.best_params_, grid_search_cv.best_score_))
	return grid_search_cv.best_estimator_


'''
def statistics_eco(df):
	stats = pd.DataFrame()
	#Date_Time
	stats['Date_Time'] = df['UTCDate'] + ' ' + df['UTCTime']
	stats[['Date_Time']] = stats[['Date_Time']].applymap(lambda x: datetime.datetime.strptime(x, '%Y.%m.%d %H:%M:%S'))
	#my_result
	stats['my_result'] = create_series_y(df)
	#my_color
	stats['my_color'] = df.White.replace('shahmatpatblog', 1)
	stats.loc[stats.my_color != 1, 'my_color'] = 0
	stats['my_color'] = stats['my_color'].astype('int64')
	#ECO
	stats['ECO'] = df['ECO']
	#collecting stats before current game - how many games were, how many games i won (for that ECO)
	white_games, black_games, white_wins, black_wins = [], [], [], []
	for i, row in stats.iterrows():
		#print(row)
		stats_before_this = stats.loc[(stats['Date_Time'] < row[0]) & (stats['ECO'] == row[3])]
		white_games.append(stats_before_this.loc[stats_before_this['my_color'] == 1, 'my_result'].count())
		white_wins.append(stats_before_this.loc[stats_before_this['my_color'] == 1, 'my_result'].sum())
		black_games.append(stats_before_this.loc[stats_before_this['my_color'] == 0, 'my_result'].count())
		black_wins.append(stats_before_this.loc[stats_before_this['my_color'] == 0, 'my_result'].sum())
		#white_games.append(stats_before_this.loc[stats_before_this['color'] == 1, 'my_result'])
		#print(stats_before_this)
	w_b = pd.DataFrame(data=np.array((white_games, white_wins, black_games, black_wins)).T,
					   columns=['white_games', 'white_wins', 'black_games', 'black_wins'])
	w_b = stats[['ECO', 'my_color']].merge(w_b, left_index=True, right_index=True, how='inner')
	#white_win_rate
	try:
		w_b['white_win_rate'] =  w_b['white_wins'] / w_b['white_games']
	except ZeroDivisionError:
		w_b['white_win_rate'] = 0
	w_b['white_win_rate'] = w_b['white_win_rate'].fillna(0)
	#black_win_rate
	try:
		w_b['black_win_rate'] =  w_b['black_wins'] / w_b['black_games']
	except ZeroDivisionError:
		w_b['black_win_rate'] = 0
	w_b['black_win_rate'] = w_b['black_win_rate'].fillna(0)
	#creating columns of games number, win_rate according to color of the game
	w_b['ECO_games'] = w_b['white_games'].where(w_b['my_color'] == 1, w_b['black_games']).round(3)
	w_b['ECO_win_rate'] = w_b['white_win_rate'].where(w_b['my_color'] == 1, w_b['black_win_rate']).round(3)
	return w_b
'''