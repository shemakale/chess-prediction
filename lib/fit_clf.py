# In this file logistic regression is fitted and scaler for numeric features is fitted too.
# Classificator is saving as .PKL file
# It is needed for prediction of the game from GUI 
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle
from create_df import TrainingDataFrame, get_names


class FittedLogit:
	''' The best logistic regression classifier for input datarame '''
	def __init__(self, df):
		self.X, self.y = self.split_X_y(df)
		self.prediction_accuracy, self.c_parameter = self.find_best_clf(self.X, self.y)
		self.fit_clf()


	def split_X_y(self, df):
		''' Divides dataframe into X (features) and y (target variable)'''
		X = df.drop(columns='my_result')
		y = df['my_result']
		return X, y


	def find_best_clf(self, X, y):
		''' Searches the logistic regression classifier with best accuracy score for test data.
		Returns accuracy score for test data and C value '''
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, shuffle=True)
		kf = KFold(n_splits=5, random_state=28, shuffle=True)
		c_values = np.power(10, range(-4, 5), dtype = np.float) #C-parameter values of Logit
		accuracy_train, accuracy_test = [], []
		for c in c_values:
			clf = LogisticRegression(C=c, random_state=28)
			cross_val_sc = np.mean(cross_val_score(clf, X_train, y_train, cv=kf))
			clf.fit(X_train, y_train)
			accuracy_train.append(cross_val_sc)
			accuracy_test.append(accuracy_score(y_test, clf.predict(X_test)))
		max_test_score = round(max(accuracy_test), 3)
		best_C = c_values[accuracy_test.index(max(accuracy_test))]
		return max_test_score, best_C


	def fit_clf(self):
		''' Fits best estimator with found C value and saves it to PKL file'''
		clf = LogisticRegression(C=self.c_parameter, random_state=28)
		clf.fit(self.X, self.y)
		with open(r'../work/clf.pkl', 'wb') as f:
			pickle.dump(clf, f)
			return clf
		

if __name__ == '__main__':
	path_to_base = get_names()[1]
	data = TrainingDataFrame(path_to_base)
	clf = FittedLogit(data.create_training_df())
	print(clf.fit_clf())
	print('accuracy = ', clf.prediction_accuracy)