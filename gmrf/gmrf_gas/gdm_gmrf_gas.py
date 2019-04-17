import numpy as np
import scipy.sparse.linalg

from common import DiscreteGasDistributionMapper, DiscreteObstacleMap, Lattice2DScalar, Observation
from gmrf.pool_inverse import getDiagonalOfInverse
#import billiard



##============================================================================
class GMRF(DiscreteGasDistributionMapper):
	
	def __init__(self, size, resolution):		
		DiscreteGasDistributionMapper.__init__(self, size, resolution)
		return


	def computeUncertainty(self):
		self._computeUncertainty()
		return self


	def _updateObservations(self):
		return self



##============================================================================
class GMRF_Gas(GMRF):

	## ------------------------------------------------------------------------
	def __init__(self, obstacle_map, sigma_z=0.075, sigma_r=0.22, sigma_b=500, tk=0, max_gas_concentration=1):
		assert(type(obstacle_map) is DiscreteObstacleMap)
		assert(sigma_z > 0)
		assert(sigma_r > 0)
		assert(sigma_b > 0)
		assert(tk >= 0)
		assert(max_gas_concentration > 0)		
		
		GMRF.__init__(self, obstacle_map.size, obstacle_map.resolution)
		
		self._om = obstacle_map		
		self._sigma_z = sigma_z * max_gas_concentration
		self._sigma_r = sigma_r * max_gas_concentration
		self._sigma_b = sigma_b * max_gas_concentration
		self._tk = tk
				
		self.gas = Lattice2DScalar(self.shape, self.resolution)
		self.gas_uncertainty = Lattice2DScalar(self.shape, self.resolution)	
		
		self._H = 0


		return
		
	
	## --------------------------------------------------------------------------
	def updateObstacleMap(self, obstacle_map):
		assert(obstacle_map.resolution is self.resolution)
		assert(obstacle_map.size is self.resolution)
		assert(obstacle_map.shape is self.resolution)
	
		self._om = obstacle_map
		#self._J_rb, self._L_rb, self._r_rb = _getStaticAxb(self, np.zeros(self._num_cells))
		
		return self
	
	
	"""
	## --------------------------------------------------------------------------
	def _getStaticAxb(self, dm):
		J_r, L_r, r_r = self._getAxb_r(dm)
		J_b, L_b, r_b = self._getAxb_b(dm)
		
		J_rb = scipy.sparse.bmat([[J_r],		[J_b]])
		L_rb = scipy.sparse.bmat([[L_r,None], 	[None,L_b]])
		r_rb = scipy.sparse.bmat([[r_r],		[r_b]])
		return J_rb, L_rb, r_rb
	"""
	
	
	
	## --------------------------------------------------------------------------
	def _getAxb_r(self, dm):
		row  = []
		col  = []
		data = []
		r    = []
		var  = []
		n    = 0
		var_r = self._sigma_r**2

		for i  in range(0, self.shape[0]-1):
			for j in range(0, self.shape[1]):
				cell    = (i,j)
				cell_d  = (i+1,j)				
				index   = self._convertCellToIndex(cell)
				index_d = self._convertCellToIndex(cell_d)
				
				obstacle = self._om.getObstacleProbabilityBetweenCells(cell, cell_d)
				r    += [dm[index] - dm[index_d]]
				row  += [n,n]
				col  += [index, index_d]
				data += [1, -1]
				var  += [(1-obstacle)/var_r]
				n    += 1
				
		
		for i  in range(0, self.shape[0]):
			for j in range(0, self.shape[1]-1):
				cell    = (i,j)
				cell_r  = (i,j+1)
				index   = self._convertCellToIndex(cell)
				index_r = self._convertCellToIndex(cell_r)
				
				obstacle = self._om.getObstacleProbabilityBetweenCells(cell, cell_r)
				r    += [dm[index] - dm[index_r]]
				row  += [n,n]
				col  += [index, index_r]
				data += [1, -1]
				var  += [(1-obstacle)/var_r]
				n    += 1	
		
				
		J_r = scipy.sparse.csc_matrix((data, (row, col)), shape=(n, self._num_cells))
		L_r = scipy.sparse.diags(var)
		r_r = scipy.sparse.csc_matrix(r).T
		return J_r, L_r, r_r
	

	
	## ------------------------------------------------------------------------
	def _getAxb_b(self, dm):
		var_b = self._sigma_b**2
		J_b = scipy.sparse.identity(self._num_cells)
		r_b = scipy.sparse.csc_matrix(dm).T
		L_b = scipy.sparse.diags(  (1/(var_b*(1-self._om.toMatrix().T.flatten()**2)+1e-10)) * np.ones(self._num_cells)  )		
		return J_b, L_b, r_b

		
	
	
	## ------------------------------------------------------------------------
	def _getAxb_z(self, dm):
		row  = []
		col  = []
		data = []
		r    = []
		var  = []
		n    = 0
		var_z = self._sigma_z**2
		
		for observation in self._observations:
			observation_cell = self._convertPositionToCell(observation.position)
			if(self._om.getObstacleProbability(observation_cell) < 0.5):
				index = self._convertCellToIndex(observation_cell)
			
				r    += [observation.gas_concentration - dm[index]]
				row  += [n]
				col  += [index]
				data += [-1]
				var  += [1/(var_z+observation.time*self._tk)]
				n    += 1

		J_z = scipy.sparse.csc_matrix((data, (row, col)), shape=(n, self._num_cells))
		L_z = scipy.sparse.diags(var)
		r_z = scipy.sparse.csc_matrix(r).T
		return J_z, L_z, r_z



	##-------------------------------------------------------------------------
	def _getAxb(self, dm):
		J_r, L_r, r_r = self._getAxb_r(dm)
		J_b, L_b, r_b = self._getAxb_b(dm)
		J_z, L_z, r_z = self._getAxb_z(dm)
		
		J = scipy.sparse.bmat([[J_r],           [J_b],           [J_z]])
		L = scipy.sparse.bmat([[L_r,None,None], [None,L_b,None], [None,None,L_z]])
		r = scipy.sparse.bmat([[r_r],           [r_b],           [r_z]])
		return J, L, r
	
	
	## ------------------------------------------------------------------------
	def _predict(self):
		if(self._has_new_observations):

			self.x = np.zeros(self._num_cells) # Choose starting conditions. Not really relevant as it has closed solution. But useful to derive to GMRF_FAS_WIND
			J, L, r = self._getAxb(self.x)
			
			Jt = J.T
			H = (Jt * L * J).tocsc()
			g = (Jt * L * (-r)).tocsc()
			
			dx = scipy.sparse.linalg.spsolve(H,g)
					
			self.x += dx 
			gas_matrix = scipy.sparse.csc_matrix(self.x).reshape(self.shape, order='F').toarray()		
			self.gas.loadMatrix(gas_matrix)
			self._H = H
			
		return self


	def _computeUncertainty(self):
		
		num_variables = self._num_cells
		solver = scipy.sparse.linalg.factorized(self._H)
		diagonal  = np.zeros(num_variables)
		
		for i in range(0, num_variables):
			e_c = np.zeros(num_variables)
			e_c[i] = 1
			diagonal[i] = solver(e_c)[i]

		uncertainty = diagonal.reshape(self.shape).T
		self.gas_uncertainty.loadMatrix(uncertainty)
		return self





##============================================================================
class GMRF_Gas_Parallel(GMRF_Gas):
	
	## ------------------------------------------------------------------------
	def __init__(self, obstacle_map, sigma_z=0.075, sigma_r=0.22, sigma_b=500, tk=0, max_gas_concentration=1):
		GMRF_Gas.__init__(self, obstacle_map, sigma_z, sigma_r, sigma_b, tk, max_gas_concentration)
		self._solver = 0
		return

	
	def _predict(self):
		if(self._has_new_observations):

			self.x = np.zeros(self._num_cells) # Choose starting conditions. Not really relevant as it has closed solution. But useful to derive to GMRF_FAS_WIND
			J, L, r = self._getAxb(self.x)
			
			Jt = J.T
			H = (Jt * L * J).tocsc()
			g = (Jt * L * (-r)).tocsc()
			
			dx = scipy.sparse.linalg.spsolve(H,g)
					
			self.x += dx 
			gas_matrix = scipy.sparse.csc_matrix(self.x).reshape(self.shape, order='F').toarray()		
			self.gas.loadMatrix(gas_matrix)
			self._H = H
			
		return self


	def _computeUncertainty(self):
		
		diagonal = getDiagonalOfInverse(self._H)
		uncertainty = diagonal.reshape(self.shape).T
		self.gas_uncertainty.loadMatrix(uncertainty)
		return self		
	
	
	def _getUncertaintyOfCellIndex(self, index):
		print(index)
		num_variables = self._num_cells
		e_c = np.zeros(num_variables)
		e_c[index] = 1
		return self._solver(e_c)[index]
	
	
		
		
##============================================================================
print("gmrf_gas:" + str(__name__))
if(__name__ is "__main__"):

	o = Lattice2DScalar.fromPGM("../../common/data/test_environments/small_office/3/obstacles.pgm", 0.1).normalize().invert()
	om = DiscreteObstacleMap(o)
	g = GMRF_Gas_Parallel(om)

	obs = [Observation((5,5), 1), Observation((5,6), 1.8), Observation((6,5), 0.2)]
	g.addObservation(obs)
	g.predict()
	g.gas.plot()
	g.computeUncertainty()
	g.gas_uncertainty.plot()
