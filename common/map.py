import numpy as np

#   Visual representation of coordinate systems (x,y) vs (i,j)
#
#       -----> j
#       |	+------------------+
#       |	|                  |
#       V	|                  |
#       i	|                  |
#           |     LATTICE      |
#           |                  |
#       y	|                  |
#       A	|                  |
#       |	|                  |
#       |	+------------------+
#       -----> x
#

	
##============================================================================	
class Map:
    
	def __init__(self, size=(0.0, 0.0)):
		size_x = size[0]
		size_y = size[1]
		assert(size_x >= 0)
		assert(size_y >= 0)
		self.size = size
		return


		
##============================================================================
class DiscreteMap(Map):
	
	def __init__(self, size=(0.0, 0.0), resolution=0.1):
		assert(resolution > 0.0)
		
		Map.__init__(self, size)
		
		size_x = size[0]
		size_y = size[1]		
		cells_i = (np.ceil(size_y / resolution)).astype(int)
		cells_j = (np.ceil(size_x / resolution)).astype(int)
		
		self.shape = (cells_i, cells_j)
		self.resolution = resolution
		self._num_cells = cells_i*cells_j
		return
	
	
	def _convertPositionToCell(self, position):		
		x = position[0]
		y = position[1]
		assert(0.0 <= x <= self.size[0])
		assert(0.0 <= y <= self.size[1])
		
		i = (np.floor(y / self.resolution)).astype(int)
		j = (np.floor(x / self.resolution)).astype(int)
		cell = (i,j)
		return cell
	
	
	def _convertCellToPosition(self, cell):		
		i = cell[0]
		j = cell[1]
		assert(0 <= i < self.shape[0])
		assert(0 <= j < self.shape[1])
		
		x = j*self.resolution + 0.5*self.resolution
		y = i*self.resolution + 0.5*self.resolution
		
		max_x = self.size[0]
		max_y = self.size[1]
		
		if(x > max_x):
			x = max_x
		if(y > max_y):
			y = max_y
	
		position = (x,y)
		return position
	
	
	def _convertCellToIndex(self, cell):
		i = cell[0]
		j = cell[1]
		
		cells_i = self.shape[0]
		cells_j = self.shape[1]
		
		assert(0 <= i < cells_i)
		assert(0 <= j < cells_j)
		
		index = i + j*cells_i
		assert(0 <= index <= self._num_cells)
		
		return index
	

##============================================================================
class DiscreteObstacleMap(DiscreteMap):
	
	def __init__(self, obstacle):
		
		cells_i = obstacle.shape[0]
		cells_j = obstacle.shape[1]
		resolution = obstacle.resolution
		
		assert(cells_i >= 0)
		assert(cells_j >= 0)
		assert(resolution >= 0)
		
		size_x = cells_j*resolution
		size_y = cells_i*resolution
		size = (size_x, size_y)
		DiscreteMap.__init__(self, size, resolution)

		self._obstacle = obstacle
	
	
	def getObstacleProbabilityBetweenCells(self, cell_1, cell_2):
		
		cell_1 = np.array(cell_1)
		cell_2 = np.array(cell_2)
		vector = cell_2 - cell_1
		unit_v = (vector / np.sqrt(vector[0]**2 + vector[1]**2))
		
		cells = [cell_1]
		length = 1
				
		while(not np.array_equal(cells[-1], cell_2)):
			current_cell = (cell_1 + length*unit_v).astype(int)
			length +=1
			if(not np.array_equal(current_cell, cells[-1])):
				cells += [current_cell]
			
		free_probability = 1
		for cell in cells:
			free_probability *= (1-self._obstacle.getCell(cell))
			
		occupied_probability = (1-free_probability)
		assert(0.0 <= occupied_probability <= 1.0)
		return occupied_probability
	
	
	def getObstacleProbability(self, cell):
		return self._obstacle.getCell(cell)
		
	
	def toMatrix(self):
		return self._obstacle.toMatrix()
	
	
"""	
o = Lattice2DScalar.fromPGM("/home/gongora/Documents/MAPIR/2018 - Gas mapping/Simulation/environments/office/gasmap1.pgm", 0.1).normalize().invert().plot()
om = DiscreteObstacleMap(o)
cell_1 = (40,10)
cell_2 = (1,1)
print(om.getObstacleBetweenCells(cell_1,cell_2))
"""