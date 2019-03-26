class Observation:
	def __init__(self, position=(0.0, 0.0), gas_concentration=0.0, wind_vector=(0.0, 0.0), obstacle=0.0, time=0, dimensions=2, content='gas'):

		assert(type(dimensions) is int and dimensions >= 1)
		assert(len(position) == dimensions and len(wind_vector) == dimensions)
		assert(gas_concentration >= 0)
		assert(1.0 >= obstacle >= 0.0)
		assert(content == 'gas' or content == 'wind' or content == 'gas+wind')
		
		self.position			= position
		self.gas_concentration	= gas_concentration
		self.wind_vector			= wind_vector
		self.obstacle			= obstacle
		self.time				= time
		self.dimensions			= dimensions
		self.content 			= content
		
		return
	
	
	
