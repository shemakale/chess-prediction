# -*- coding: utf-8 -*-
#GUI
import tkinter as tk
import tkinter.ttk as ttk
import tkinter.filedialog as fd
from PIL import ImageTk, Image
import make_prediction
#import os


class Totalizer_window(tk.Frame):
	"""Window with fields to fill and buttons to show predictions"""
	def __init__(self, parent):
		tk.Frame.__init__(self, parent)
		self.parent = parent
		self.parent.title ("Predictor chess results")
		self.parent.geometry('500x700')
		self.pack(side='top')
		self.create_widgets()


	def create_widgets(self):
		""" Creating interface to input game information for prediction """
		self.instrucion_lbl = tk.Label(self, justify='left', font=("Calibri", 10, "bold"), bd=10,
			text="Добро пожаловать в\n\"Шахматный Нострадамус\"!\nЗдесь вы предугадаете результат \nсвоей партии! Для предсказания\nвведите своё имя и выберите базу\nваших партий (PGN файл). Затем\nукажите данные партии, результат\nкоторой хотите предсказать и\nнажмите кнопку\n\"Предсказать результат!\"")
		self.instrucion_lbl.grid(row=0, column=0)
		#img = Image.open("cat.jpeg")
		#img = img.resize((100, 100), Image.ANTIALIAS)
		#img = ImageTk.PhotoImage(img)
		#self.avatar_img = tk.Label(self, image=img, compound='top', width=50, height=50)
		#self.avatar_img.grid(row=0, column=1)
		self.enter_name_lbl = tk.Label(self, text="Введите своё имя (никнейм): ")
		self.enter_name_lbl.grid(row=1, column=0, sticky='W')
		self.enter_name_entry = tk.Entry(self, width=20)
		self.enter_name_entry.grid(row=1, column=1, sticky='W')
		self.enter_name_entry.insert(0, 'shahmatpatblog')
		self.choose_file_btn = tk.Button(self, text="Выберите файл базы данных (PGN)", command=self.choose_file, width=30)
		self.choose_file_btn.grid(row=2, column=0, sticky='W')

		self.enter_info_lbl = tk.Label(self, text="Введите данные партии:", font=("Calibri", 12, "bold"))
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


	def choose_file(self):
		filetypes = (("База данных партий", "*.pgn"), ("Любой", "*"))
		filename = fd.askopenfilename(title="Открыть файл", initialdir="/",
										filetypes=filetypes)
		if filename:
			with open('user_base_info.txt', 'w') as f:
				f.write('{}\n{}'.format(self.enter_name_entry.get(), filename))


	def get_prediction(self):
		""" Get result of prediction: probability of my win in this game """
		data_ = [self.event_cmbox.get(), self.white_entry.get(), self.black_entry.get(), self.year_box.get(),\
				self.month_box.get(), self.day_box.get(), self.hour_box.get(), self.minute_box.get(),\
				self.elo_white_entry.get(), self.elo_black_entry.get(), self.timecontrol_cmbox.get()]
		prediction = make_prediction.predict_result(data_)
		self.output_field.configure(text=self.print_prediction(prediction))


	def print_prediction(self, pred):
		''' pred - predicted probability of win class. this function makes formatting of the prediction result '''
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
	my_window = Totalizer_window(root)
	my_window.mainloop()


if __name__ == '__main__':
	main()

