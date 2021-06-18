import tkinter as tk #Tk, canvas, CHORD, ARC, LAST, BOTH, X, Y, LEFT, PhotoImage, Label, Entry, Button, ttk, Frame
import tkinter.ttk as ttk
import func_totalizer as ft

class Totalizer_window(tk.Frame):
	"""Window with fields to fill and buttons to show predictions"""
	def __init__(self, parent):
		tk.Frame.__init__(self, parent)
		self.parent = parent
		self.parent.title ("Predictor chess results")
		self.parent.geometry('500x600')
		self.pack(side='top')
		self.create_widgets()
		self.event_var = ''

	def create_widgets(self):
		#self.event_var = ''
		self.top_lbl = tk.Label(self, text="Введите данные партии", font=("Calibri", 14, "bold"))
		self.top_lbl.grid(row=0, column=0)

		self.event_lbl = tk.Label(self, text="Выберите тип события партии: ")
		self.event_lbl.grid(row=1, column=0, sticky='W')
		self.event_cmbox = ttk.Combobox(self, state='readonly')
		self.event_cmbox['values'] = ('Обычная игра', 'Турнирная игра')
		self.event_cmbox.current(0)
		self.event_cmbox.grid(row=1, column=1, sticky='W')

		self.white_lbl = tk.Label(self, text="Белые: ")
		self.white_lbl.grid(row=2, column=0, sticky='W')
		self.white_entry = tk.Entry(self, width=20)
		self.white_entry.grid(row=2, column=1, sticky='W')
		self.white_entry.insert(0, 'shahmatpatblog')

		self.black_lbl = tk.Label(self, text="Чёрные: ")
		self.black_lbl.grid(row=3, column=0, sticky='W')
		self.black_entry = tk.Entry(self, width=20)
		self.black_entry.grid(row=3, column=1, sticky='W')
		self.black_entry.insert(0, 'somebody')

		self.date_lbl = tk.Label(self, text="Дата (год, месяц, день): ")
		self.date_lbl.grid(row=4, column=0, sticky='W')
		self.year_box = ttk.Spinbox(self, from_=2010, to=2030, width=11)
		self.year_box.grid(row=4, column=1, sticky='W')
		self.year_box.insert(0, '2021')
		self.month_box = ttk.Spinbox(self, from_=1, to=12, width=6)
		self.month_box.grid(row=4, column=1, sticky='E')
		self.month_box.insert(0, '01')
		self.day_box = ttk.Spinbox(self, from_=1, to=31, width=6)
		self.day_box.grid(row=4, column=2, sticky='W')
		self.day_box.insert(0, '01')

		self.time_lbl = tk.Label(self, text="Время (ЧЧ:ММ): ")
		self.time_lbl.grid(row=5, column=0, sticky='W')
		self.hour_box = ttk.Spinbox(self, from_=0, to=23, width=6)
		self.hour_box.grid(row=5, column=1, sticky='W')
		self.hour_box.insert(0, '12')
		self.minute_box = ttk.Spinbox(self, from_=0, to=59, width=6)
		self.minute_box.grid(row=5, column=1, sticky='E')
		self.minute_box.insert(0, '00')

		self.elo_white_lbl = tk.Label(self, text="Рейтинг белых: ")
		self.elo_white_lbl.grid(row=6, column=0, sticky='W')
		self.elo_white_entry = tk.Entry(self, width=10)
		self.elo_white_entry.grid(row=6, column=1, sticky='W')
		self.elo_white_entry.insert(0, '2200')

		self.elo_black_lbl = tk.Label(self, text="Рейтинг чёрных: ")
		self.elo_black_lbl.grid(row=7, column=0, sticky='W')
		self.elo_black_entry = tk.Entry(self, width=10)
		self.elo_black_entry.grid(row=7, column=1, sticky='W')
		self.elo_black_entry.insert(0, '2200')

		self.timecontrol_lbl = tk.Label(self, text="Выберите контроль времени: ")
		self.timecontrol_lbl.grid(row=8, column=0, sticky='W')
		self.timecontrol_cmbox = ttk.Combobox(self, state='readonly')
		self.timecontrol_cmbox['values'] = ('Блиц', 'Пуля')
		self.timecontrol_cmbox.current(0)
		self.timecontrol_cmbox.grid(row=8, column=1, sticky='W')

		self.predict_btn = tk.Button(self, text="Предсказать\nрезультат!", height=3, width="13", command=self.get_prediction)
		self.predict_btn.grid(row=9, column=0, sticky='W', pady=10)

		self.output_field = tk.Label(self, text='', font=("Calibri", 12, "bold"), justify='left')
		self.output_field.grid(row=10, column=0, sticky='W')


	def get_prediction(self):
		data_ = [self.event_cmbox.get(), self.white_entry.get(), self.black_entry.get(), self.year_box.get(),\
				self.month_box.get(), self.day_box.get(), self.hour_box.get(), self.minute_box.get(),\
				self.elo_white_entry.get(), self.elo_black_entry.get(), self.timecontrol_cmbox.get()]
		#data_ = ['Обычная игра', 'shahmatpatblog', 'shahmatpatblo1', '2021', '04', '18', '02', '14', '2239', '2366', 'Блиц']
		#self.output_field.configure(text=data_)
		prediction = ft.predict_result(data_)
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
	root = tk.Tk () #main screen
	my_window = Totalizer_window(root)
	#print(my_window.event_var)
	my_window.mainloop()
	#print(my_window.event_var)

if __name__ == '__main__':
	main()

