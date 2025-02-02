# estos son las constantes que definí en el documento de latex
aggresive_afforgade_percentage = 0.36
safe_affordage_percentage = 0.28
disposable_percentage = 0.0267

# resultados finales
income = None
debt = None
period = None

def can_afford_rent(self, inputs):
    """
    Es la función principal que se encarga de calcular si una persona puede pagar la renta, si la persona no es pobre
    entonces continúa los cálculos llamando a la función calculate_affordage, en caso contrario regresa "poor"
    :param inputs: Tupla que contiene los valores de ingreso, deuda y periodo
    :return: Tupla con los valores de aggresive_affordage, safe_affordage y a_third, ó "poor"
    """

    self.income = inputs[0]
    self.debt = inputs[1]
    self.period = inputs[2]

    if self.income == 0:
        return "poor"

    # si el periodo es por año, entonces se divide el ingreso entre la cantidad de meses
    if period == 'per year':
        self.income = self.income / 12

    # si el ingreso por mes es menor a la deuda, entonces la persona es pobre
    if self.income * aggresive_afforgade_percentage - self.debt < 0:
        return "poor"

    return self.calculate_affordage(self)

def calculate_affordage(self):
    """
    Calcula el aggresive_affordage, safe_affordage y a_third, si la deuda es menor al ingreso disponible, entonces
    se calcula el tercio del ingreso, en caso contrario no se calcula el tercio
    :return: Tupla con los valores de aggresive_affordage, safe_affordage y a_third
    """

    aggresive_affordage = round(self.income * aggresive_afforgade_percentage - self.debt)
    safe_affordage = round(self.income * safe_affordage_percentage - self.debt)

    disposable_income = self.income * disposable_percentage

    if self.debt < disposable_income:
        a_third = round(self.income / 3)
        return aggresive_affordage, safe_affordage, a_third

    if self.debt >= disposable_income:
        return aggresive_affordage, safe_affordage
