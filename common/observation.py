class Observation:
	def __init__(self, position=(0.0, 0.0), gas_concentration=0.0, wind_vector=(0.0, 0.0), time=0, dimensions=2, contains='gas'):

		assert(type(dimensions) is int and dimensions >= 1)
		assert(len(position) == dimensions and len(wind_vector) == dimensions)
		assert(gas_concentration >= 0)
		assert(contains == 'gas' or contains == 'wind' or contains == 'gas+wind')
		
		self.position           = position
		self.gas_concentration  = gas_concentration
		self.wind_vector        = wind_vector
		self.time               = time
		self.dimensions         = dimensions
		self.contains           = contains
		
		return