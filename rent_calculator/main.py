import sys, calculations
from PyQt6.QtWidgets import (QApplication, QMainWindow, QLabel, QLineEdit, QVBoxLayout, QComboBox, QHBoxLayout,
                             QWidget, QPushButton, QFrame)
from PyQt6.QtGui import QIcon, QPixmap

class MainWindow(QMainWindow):
    """
    Definición de los componentes y layouts de la interfaz de la calculadora de renta.
    """

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Rent Calculator")

        # columna izquierda pa los labels
        disposableLabel = QLabel("Your monthly debt payback")
        disposableButton = QPushButton("")
        disposableButton.setStyleSheet("border-radius: 50;")
        disposableButton.setIcon(QIcon("resources/question.png"))
        disposableButton.setToolTip(
            "The total of the minimum amounts you\npay each month to keep up with the\nongoing debts, such as students loans, car \nloans, credit cards, child support, alimony\npaid, and personal loans.")

        disposableLayout = QHBoxLayout()
        disposableLayout.addWidget(disposableLabel)
        disposableLayout.addWidget(disposableButton)

        leftColumn = QVBoxLayout()
        leftColumn.addWidget(QLabel("Your pre-tax income"))
        leftColumn.addLayout(disposableLayout)

        # columna central pa los inputs
        self.incomeInput = QLineEdit("$80,000")
        self.debtInput = QLineEdit("$0")

        # evento que se dispara cuando el input pierde el focus y hubo un cambio en el contenido del input
        self.incomeInput.editingFinished.connect(self.format_inputs)
        self.debtInput.editingFinished.connect(self.format_inputs)

        centerColumn = QVBoxLayout()
        centerColumn.addWidget(self.incomeInput)
        centerColumn.addWidget(self.debtInput)

        # columna derecha pa los restos
        self.timeCombo = QComboBox()
        self.timeCombo.addItem("per year")
        self.timeCombo.addItem("per month")
        self.timeCombo.setMaximumWidth(100)

        rightColumn = QVBoxLayout()
        rightColumn.addWidget(self.timeCombo)
        rightColumn.addWidget(QLabel("car/student loan, credit cards, etc"))

        # meterlo todow al sub layout
        inputsLayout = QHBoxLayout()
        inputsLayout.addLayout(leftColumn)
        inputsLayout.addLayout(centerColumn)
        inputsLayout.addLayout(rightColumn)

        # boton calcular
        self.calculateButton = QPushButton("Calculate")
        self.calculateButton.setStyleSheet("""
		    QPushButton {
		        background-color: #4c7b25;
		        color: white;
		    }
		    QPushButton:hover {
		        background-color: #444444;
		    }
		""")
        self.calculateButton.setIcon(QIcon("resources/play.png"))
        self.calculateButton.setMinimumWidth(150)
        self.calculateButton.setMinimumHeight(40)
        self.calculateButton.clicked.connect(self.calculate)

        # boton limpiar inputs
        self.clearButton = QPushButton("Clear")
        self.clearButton.setStyleSheet("""
		    QPushButton {
		        background-color: #ababab;
		        color: white;
		    }
		    QPushButton:hover {
		        background-color: #444444;
		    }
		""")
        self.clearButton.setMinimumWidth(40)
        self.clearButton.setMinimumHeight(40)
        self.clearButton.clicked.connect(self.clear)

        # botones layout
        buttonsLayout = QHBoxLayout()
        buttonsLayout.addStretch()  # pa que se centren los botones
        buttonsLayout.addWidget(self.calculateButton)
        buttonsLayout.addWidget(self.clearButton)
        buttonsLayout.addStretch()

        # label del warning
        self.warningLabel = QLabel("Please provide a positive income value.")
        self.warningLabel.setStyleSheet("color: red; font-size: 18px;")

        # CONTENEDOR DE LOS RESULTADOS
        self.resultLabel = QLabel("")
        self.thirdPartLabel = QLabel("")

        # estos son los tags de la imagen
        self.safeTag = QLabel("")
        self.safeTag.setContentsMargins(0, 0, 40, 0)
        self.aggresiveTag = QLabel("")
        tagsLayout = QHBoxLayout()
        tagsLayout.addStretch()
        tagsLayout.addWidget(self.safeTag)
        tagsLayout.addWidget(self.aggresiveTag)
        tagsLayout.addStretch()

        # esta es la imagen
        rent_map = QPixmap("resources/rent-map.png")
        rent_map_label = QLabel()
        rent_map_label.setPixmap(rent_map)

        # armamos todito
        resultsLayout = QVBoxLayout()
        resultsLayout.addWidget(self.resultLabel)
        resultsLayout.addLayout(tagsLayout)
        resultsLayout.addWidget(rent_map_label)
        resultsLayout.addWidget(self.thirdPartLabel)
        resultsLayout.setSpacing(10)

        # frame pa los resultados
        self.resultsFrame = QFrame()
        self.resultsFrame.setLayout(resultsLayout)

        # LABEL PARA MOSTRAR QUE ES POBRE
        self.poorLabel = QLabel("At that income and debt level, it will be hard to meet rent payments.")

        # frame pa los inputs y botones pq los frames pueden tener estilos y ser ocultados
        subLayout = QVBoxLayout()
        subLayout.addLayout(inputsLayout)
        subLayout.addLayout(buttonsLayout)

        inputsFrame = QFrame()
        inputsFrame.setLayout(subLayout)
        inputsFrame.setContentsMargins(0, 10, 0, 0)
        inputsFrame.setObjectName("inputsFrame")
        inputsFrame.setStyleSheet('''
			#inputsFrame {
				background-color: #eeeeee;
				border: 1px solid #000000;
			}
		''')

        # main layout
        mainLayout = QVBoxLayout()
        mainLayout.addWidget(self.warningLabel)
        mainLayout.addWidget(self.poorLabel)
        mainLayout.addWidget(self.resultsFrame)
        mainLayout.addWidget(inputsFrame)

        # CONTENEDOR MAIN
        container = QWidget()
        container.setLayout(mainLayout)

        # Set the central widget of the Window.
        self.setCentralWidget(container)
        self.setMaximumHeight(self.minimumHeight())  # se estiraba poquillo sin esto
        self.hide_all()


##################################
    """
    Funciones de la interfaz de la calculadora de renta.
    """
    def calculate(self):
        """
        Al hacer click en el botón de calcular, se obtienen los valores de los inputs y se pasan a la función
        can_afford_rent de calculations.py para obtener los resultados.
        :return: Renderiza los resultados en la interfaz.
        """

        self.hide_all()
        inputs = self.get_inputs()

        # como puede regresar un false si el input es incorrecto pues lo usamos para verificar
        if inputs:
            results = calculations.can_afford_rent(calculations, inputs)

            if results == "poor":
                self.render_poor()
                return

            self.resultsFrame.show()

            if len(results) == 3:
                self.render_results(results)
                self.render_a_third(results[2])

            if len(results) == 2:
                self.render_results(results)

    def clear(self):
        """
        Limpia los inputs de la interfaz.
        """

        self.incomeInput.setText("$")
        self.debtInput.setText("$")

    def format_inputs(self):
        """
        Formatea los inputs de los campos de ingresos y deudas.
        """

        if self.incomeInput.text() == "": self.incomeInput.setText("$")
        elif self.incomeInput.text() != "$": self.incomeInput.setText(format_as_number(self.incomeInput.text()))

        if self.debtInput.text() == "": self.debtInput.setText("$")
        elif self.debtInput.text() != "$": self.debtInput.setText(format_as_number(self.debtInput.text()))

    def get_inputs(self):
        """
        Obtiene los valores de los campos de ingresos, deudas y periodo.
        """

        income = self.incomeInput.text()

        if income == '' or income == "$":
            self.render_warning()
            return False

        income = income.replace("$", "")
        income = income.replace(",", "")
        income = float(income)

        if income <= 0:
            self.render_warning()
            return False

        debt = self.debtInput.text()

        if debt == '' or debt == "$":
            debt = '0'

        debt = debt.replace("$", "")
        debt = debt.replace(",", "")
        debt = float(debt)

        period = self.timeCombo.currentText()

        return income, debt, period

    def hide_all(self):
        """
        Oculta todos los elementos de la interfaz que no sean el frame de los inputs
        """

        self.warningLabel.hide()
        self.resultsFrame.hide()
        self.thirdPartLabel.hide()
        self.poorLabel.hide()

    def render_warning(self):
        """
        Muestra un mensaje de advertencia si el usuario no ha ingresado un valor de ingreso.
        """

        self.setMaximumHeight(self.minimumHeight())
        self.warningLabel.show()

    def render_results(self, results):
        """
        Muestra los resultados en la interfaz.
        :param results: Lista con los resultados de la función can_afford_rent.
        """

        self.resultLabel.setText(
            f"You can afford up to <font color='green'><b>${results[0]}</b></font> per month on a rental payment.<br>It is recommended to keep your rental payment below <font color='green'><b>${results[1]}</b></font> per month.")
        self.safeTag.setText(f"<font color='green'><b>${results[1]}</b></font>")
        self.safeTag.setMinimumWidth(len(self.safeTag.text()))
        self.aggresiveTag.setText(f"<b>${results[0]}</b>")
        self.aggresiveTag.setMinimumWidth(len(self.aggresiveTag.text()))

    def render_a_third(self, third):
        """
        Muestra un mensaje en la interfaz con el valor de un tercio del ingreso.
        :param third: Valor de un tercio del ingreso.
        """

        self.thirdPartLabel.setText(
            f"Some landlords may not accept applications with more than 1/3 of gross income on rent,\nwhich is ${third}.")
        self.thirdPartLabel.show()

    def render_poor(self):
        """
        Muestra un mensaje en la interfaz si el usuario es un muerto de hambre.
        """

        self.poorLabel.show()

def format_as_number(number):
    """
    Formatea un número en formato de moneda. Le de derecha a izquierda el input de los ingresos,
    cada 3 dígitos le pone una coma y al final le pone un signo de dólar.
    :param number: Número a formatear.
    :return: Número formateado.
    """

    number = number.replace("$", "")
    number = number.replace(",", "")
    counter = 0
    new_number = ""

    for i in range(len(number) - 1, -1, -1):
        if not number[i].isnumeric(): continue

        counter += 1
        new_number = number[i] + new_number

        if counter % 3 == 0:
            new_number = "," + new_number

    if new_number[0] == ",": new_number = new_number[1:]

    return "$" + new_number

app = QApplication(sys.argv)

window = MainWindow()
window.show()

app.exec()
