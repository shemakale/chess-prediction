import re
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import datetime
from sklearn.model_selection import GridSearchCV

def create_df_from_pgn(file_path, col_names):
	'''
	создаёт датафрейм из шахматной базы формата .pgn
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
