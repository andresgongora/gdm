from .observation import Observation
from .map import DiscreteMap


##============================================================================
class GasDistributionMapper:
	
	def __init__(self):
		"""Init class"""
	
		self._observations = []
		self._has_new_observations = False
		return

	
	def addObservation(self, observation):
		""" Add a single or multiple observations to the data pool """

		if type(observation) is list:
			assert(type(observation[0]) is Observation)
			self._observations += observation
		else:
			assert(type(observation) is Observation)
			self._observations += [observation]
		
		self._has_new_observations = True
		self._updateObservations()
		return self


	def predict(self):
		""" Update prediction if new observations become available """
		if (self._has_new_observations):
			self._predict()
			self._has_new_observations = False
		return self




##============================================================================
class DiscreteGasDistributionMapper(GasDistributionMapper, DiscreteMap):
	
	def __init__(self, size, resolution):
		GasDistributionMapper.__init__(self)
		DiscreteMap.__init__(self, size, resolution)
		return
	



"""
mb = GasDistributionMapper()
o = Observation()
mb.addObservation([o])
m = DiscreteGasDistributionMapper((10,10), 0.1)
"""

