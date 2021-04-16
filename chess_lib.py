import re
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import datetime
from sklearn.model_selection import GridSearchCV

def create_df_from_pgn(file_path, col_names):
    with open(file_path, 'r', encoding='utf-8') as f:
        text_file = f.read()
        data_dict = {} #создаю словарь признаков и через цикл заполняю его
        for col_name in col_names:
            pattern = r'(?<=\[' + col_name +  r' \")[^\"]+'
            data_dict[col_name] = re.findall(pattern, text_file)
            #print(col_name, len(data_dict[col_name])) #проверка на одинаковое количество элементов у признака
    return pd.DataFrame(data=data_dict)

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

def create_series_y(df):
    '''
    create target variable y as pandas Series
    '''
    target = pd.DataFrame()
    #вначале создаю столбец каким цветом я играю
    target['my_color'] = df.White.replace('shahmatpatblog', 1)
    target.loc[target.my_color != 1, 'my_color'] = 0

    target['result'] = df['Result'].replace('1-0', 1).replace('0-1', 0).replace('1/2-1/2', 0.5)

    target['my_result'] = target.my_color + target.result
    target.my_result = target.my_result.replace([0, 2], 'win').replace(1, 'not_win').replace([0.5, 1.5], 'not_win')
    target.my_result = target.my_result.replace('win', 1).replace('not_win', 0)
    return target.my_result

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
