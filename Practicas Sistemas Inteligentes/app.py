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

		self.button = QPushButton("Press Me!")                          
		self.button.clicked.connect(self.the_button_was_clicked)
		layout = QVBoxLayout()
		layout2 = QHBoxLayout()
		layout.addWidget(self.input)
		layout.addWidget(self.label)
		layout.addWidget(self.button)

		layout2.addLayout(layout)
		self.input2 = QLineEdit()
		layout2.addWidget(self.input2);

		container = QWidget()
		container.setLayout(layout2)

		# Set the central widget of the Window.
		self.setCentralWidget(container)
	def the_button_was_clicked(self):
		self.button.setEnabled(False)
		print(self.input.text())

app = QApplication(sys.argv)

window = MainWindow()
window.show()

app.exec()
