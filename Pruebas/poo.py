class Coordinate(object):
	def __init__(self, x, y):
		print("Constructor")
		self.x = x
		self.y = y

	def getX(self):
		return self.x

	def getY(self):
		return self.y

	def distance(self, other):
		x_diff_sq = (self.x-other.x)**2
		y_diff_sq = (self.y-other.y)**2
		return (x_diff_sq + y_diff_sq)**0.5

	def __str__(self):
		return ("X:" + str(self.x) + ", Y:" + str(self.y))

c = Coordinate(3,4)
origin = Coordinate(0,0)
print(c.getX())
print(origin.getY())

# se puede usar una instancia o el objeto para llamar a la funcion
print(c.distance(origin))
print(Coordinate.distance(c,origin))

print(c)

