# -*- coding: utf-8 -*-
#GUI
import tkinter as tk
import tkinter.ttk as ttk
import tkinter.filedialog as fd
import make_prediction
from create_df import TrainingDataFrame, get_names
from fit_clf import FittedLogit


def fit_estimator():
	""" Function for button Обучить модель. It takes path name from user_base_info.txt (get_names()),
	then creates training dataframe (TrainingDataFrame class), and finally fits estimator (FittedLogit class) """
	path_to_base = get_names()[1]
	chessbase = TrainingDataFrame(path_to_base)
	FittedLogit(chessbase.create_training_df())
	print('Модель обучена и готова к предсказанию!')


class TopDescription(tk.Frame):
	""" Frame with description on the top of the window """
	def __init__(self, parent):
		tk.Frame.__init__(self, parent)
		self.parent = parent
		self.parent.title ("Chess Nostradamus")
		self.parent.geometry('500x100')
		self.pack()
		self.lbl_descrition()

	def lbl_descrition(self):
		self.instrucion_lbl = tk.Label(self, justify='left', font=("Calibri", 10, "bold"), bd=10,
			text="Добро пожаловать в \"Шахматный Нострадамус\"!\nЗдесь вы предугадаете результат своей партии!\nДля предсказания введите своё имя и выберите базу ваших партий (PGN файл).\nНажмите кнопку \"Обучить модель\".\nЗатем укажите данные партии, результат которой хотите предсказать\nи нажмите кнопку \"Предсказать результат!\"")
		self.instrucion_lbl.pack()


class UserData(tk.Frame):
	""" Frame where user enters his name, chooses chessbase file and fits classificator """
	def __init__(self, parent):
		tk.Frame.__init__(self, parent)
		self.parent = parent
		self.parent.geometry('500x200')
		self.pack()
		self.lbl_enter_name()
		self.entry_name()
		self.but_choose_base()
		self.but_fit_clf()


	def lbl_enter_name(self):
		self.enter_name_lbl = tk.Label(self, text="ВВЕДИТЕ СВОЁ ИМЯ (НИКНЕЙМ): ", justify='left')
		self.enter_name_lbl.grid(row=0, column=0, sticky='W')
	

	def entry_name(self):
		self.enter_name_entry = tk.Entry(self, width=24)
		self.enter_name_entry.grid(row=0, column=1, sticky='W', padx=10)
		self.enter_name_entry.insert(0, 'shahmatpatblog')


	def but_choose_base(self):
		self.choose_file_btn = tk.Button(self, text="Выберите файл базы данных (PGN)", command=self.choose_file, width=28)
		self.choose_file_btn.grid(row=1, column=0)


	def but_fit_clf(self):
		self.fit_clf_btn = tk.Button(self, text="Обучить модель", font=("Calibri", 10, "bold"), command=fit_estimator, width=20)
		self.fit_clf_btn.grid(row=1, column=1, padx=10)


	def choose_file(self):
		filetypes = (("База данных партий", "*.pgn"), ("Любой", "*"))
		filename = fd.askopenfilename(title="Открыть файл", initialdir="base/",
										filetypes=filetypes)
		if filename:
			with open(r'../base/user_base_info.txt', 'w') as f:
				f.write('{}\n{}'.format(self.enter_name_entry.get(), filename))


class GameData(tk.Frame):
	""" Frame where user enters data of the game to predict and prediction is shown """
	def __init__(self, parent):
		tk.Frame.__init__(self, parent)
		self.parent = parent
		self.parent.geometry('500x600')
		self.pack(side='left')
		self.create_widgets()


	def create_widgets(self):
		""" Creating interface to input game information for prediction """
		self.enter_info_lbl = tk.Label(self, text="ПОСЛЕ ТОГО, КАК МОДЕЛЬ ОБУЧИТСЯ,\nВВЕДИТЕ ДАННЫЕ ПАРТИИ:", justify='left')
		self.enter_info_lbl.grid(row=3, column=0, pady=1, sticky='W')

		self.event_lbl = tk.Label(self, text="Выберите тип события партии: ")
		self.event_lbl.grid(row=4, column=0, sticky='W')
		self.event_cmbox = ttk.Combobox(self, state='readonly')
		self.event_cmbox['values'] = ('Обычная игра', 'Турнирная игра')
		self.event_cmbox.current(0)
		self.event_cmbox.grid(row=4, column=1, sticky='W')

		self.white_lbl = tk.Label(self, text="Белые: ")
		self.white_lbl.grid(row=5, column=0, sticky='W')
		self.white_entry = tk.Entry(self, width=20)
		self.white_entry.grid(row=5, column=1, sticky='W')
		self.white_entry.insert(0, 'shahmatpatblog')

		self.black_lbl = tk.Label(self, text="Чёрные: ")
		self.black_lbl.grid(row=6, column=0, sticky='W')
		self.black_entry = tk.Entry(self, width=20)
		self.black_entry.grid(row=6, column=1, sticky='W')
		self.black_entry.insert(0, 'somebody')

		self.date_lbl = tk.Label(self, text="Дата (год, месяц, день): ")
		self.date_lbl.grid(row=7, column=0, sticky='W')
		self.year_box = ttk.Spinbox(self, from_=2010, to=2030, width=11)
		self.year_box.grid(row=7, column=1, sticky='W')
		self.year_box.insert(0, '2021')
		self.month_box = ttk.Spinbox(self, from_=1, to=12, width=6)
		self.month_box.grid(row=7, column=1, sticky='E')
		self.month_box.insert(0, '01')
		self.day_box = ttk.Spinbox(self, from_=1, to=31, width=6)
		self.day_box.grid(row=7, column=2, sticky='W')
		self.day_box.insert(0, '01')

		self.time_lbl = tk.Label(self, text="Время (ЧЧ:ММ): ")
		self.time_lbl.grid(row=9, column=0, sticky='W')
		self.hour_box = ttk.Spinbox(self, from_=0, to=23, width=6)
		self.hour_box.grid(row=9, column=1, sticky='W')
		self.hour_box.insert(0, '12')
		self.minute_box = ttk.Spinbox(self, from_=0, to=59, width=6)
		self.minute_box.grid(row=9, column=1, sticky='E')
		self.minute_box.insert(0, '00')

		self.elo_white_lbl = tk.Label(self, text="Рейтинг белых: ")
		self.elo_white_lbl.grid(row=10, column=0, sticky='W')
		self.elo_white_entry = tk.Entry(self, width=10)
		self.elo_white_entry.grid(row=10, column=1, sticky='W')
		self.elo_white_entry.insert(0, '2200')

		self.elo_black_lbl = tk.Label(self, text="Рейтинг чёрных: ")
		self.elo_black_lbl.grid(row=11, column=0, sticky='W')
		self.elo_black_entry = tk.Entry(self, width=10)
		self.elo_black_entry.grid(row=11, column=1, sticky='W')
		self.elo_black_entry.insert(0, '2200')

		self.timecontrol_lbl = tk.Label(self, text="Выберите контроль времени: ")
		self.timecontrol_lbl.grid(row=12, column=0, sticky='W')
		self.timecontrol_cmbox = ttk.Combobox(self, state='readonly')
		self.timecontrol_cmbox['values'] = ('Блиц', 'Пуля')
		self.timecontrol_cmbox.current(0)
		self.timecontrol_cmbox.grid(row=12, column=1, sticky='W')

		self.predict_btn = tk.Button(self, text="Предсказать\nрезультат!", height=3, width=13, command=self.get_prediction)
		self.predict_btn.grid(row=13, column=0, sticky='W', pady=10)

		self.output_field = tk.Label(self, text='', font=("Calibri", 12, "bold"), justify='left')
		self.output_field.grid(row=14, column=0, sticky='W')


	def get_prediction(self):
		""" Get result of prediction: probability of my win in this game """
		data_ = [self.event_cmbox.get(), self.white_entry.get(), self.black_entry.get(), self.year_box.get(),\
				self.month_box.get(), self.day_box.get(), self.hour_box.get(), self.minute_box.get(),\
				self.elo_white_entry.get(), self.elo_black_entry.get(), self.timecontrol_cmbox.get()]
		prediction = make_prediction.predict_result(data_)
		self.output_field.configure(text=self.print_prediction(prediction))
		print('Предсказание получено')


	def print_prediction(self, pred):
		''' This function makes formatting of the prediction result. pred - predicted probability of win class. '''
		if pred < 20:
			phrase = 'Прости, но я бы \nне поставил на тебя!..'
		elif pred >= 20 and pred < 40:
			phrase = 'Шансов немного, \nно я верю в тебя!'
		elif pred >= 40 and pred < 60:
			phrase = 'Шансы примерно равны. \nДерзай, и у тебя получится!'
		elif pred >= 60 and pred < 80:
			phrase = 'Все шансы твои. \nЯ жду твоей победы, чувак!'
		else:
			phrase = 'Ну, если тут не выигрывать, \nто я тогда не знаю где...'
		return '='*25 + '\nВероятность твоей победы:\n{} %\n\n{}\n'.format(pred, phrase) + '='*25


def main ():
	""" Open a window """
	root = tk.Tk () #main screen
	description_label = TopDescription(root)
	user_data = UserData(root)
	game_data = GameData(root)
	root.mainloop()


if __name__ == '__main__':
	main()

