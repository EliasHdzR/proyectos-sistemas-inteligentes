from PyQt6.QtCore import QSize, Qt
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QLineEdit, QVBoxLayout, QHBoxLayout, QWidget
import sys
class MainWindow(QMainWindow):
	def __init__(self):
		super().__init__()
		self.setWindowTitle("My App")
		self.label = QLabel()
		self.input = QLineEdit()
		self.input.textChanged.connect(self.label.setText)
		#self.setFixedSize(QSize(1920,1080))

		self.button = QPushButton("Press Me!")                          
		self.button.clicked.connect(self.the_button_was_clicked)
		layout = QVBoxLayout()
		numPad = QHBoxLayout()
		
		layoutEnmedio = QVBoxLayout()
		layoutInterno = QHBoxLayout()

		#filas
		col1 = QVBoxLayout()
		col2 = QVBoxLayout()
		col3 = QVBoxLayout()
		col4 = QVBoxLayout()

		col1.addWidget(QPushButton("7"))
		col1.addWidget(QPushButton("4"))
		col1.addWidget(QPushButton("1"))
		col1.addWidget(QPushButton("0"))

		col2.addWidget(QPushButton("8"))
		col2.addWidget(QPushButton("5"))
		col2.addWidget(QPushButton("2"))

		col3.addWidget(QPushButton("9"))
		col3.addWidget(QPushButton("6"))
		col3.addWidget(QPushButton("3"))

		layoutInterno.addLayout(col2)
		layoutInterno.addLayout(col3)

		layoutEnmedio.addLayout(layoutInterno)
		layoutEnmedio.addWidget(QPushButton("="))

		col4.addWidget(QPushButton("/"))
		col4.addWidget(QPushButton("*"))
		col4.addWidget(QPushButton("-"))
		col4.addWidget(QPushButton("+"))

		numPad.addLayout(col1)
		numPad.addLayout(layoutEnmedio)
		numPad.addLayout(col4)

		layout.addWidget(self.input)
		layout.addLayout(numPad)

		container = QWidget()
		container.setLayout(layout)


		# Set the central widget of the Window.
		self.setCentralWidget(container)
	def the_button_was_clicked(self):
		self.button.setEnabled(False)
		print(self.input.text())

app = QApplication(sys.argv)

window = MainWindow()
window.show()

app.exec()
