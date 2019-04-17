import numpy as np
from .map import DiscreteMap


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
class Map():
    
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
		cells_i = np.ceil(size_y / resolution)
		cells_j = np.ceil(size_x / resolution)
		
		self.shape = (cells_i, cells_j)
		self.resolution = resolution
		return


	def _convertCoordinatesToCell(self, coordinates):
		
		x = coordinates[0]
		y = coordinates[1]
		assert(0.0 <= x <= self.size[0])
		assert(0.0 <= y <= self.size[1])
		
		i = np.floor(y / self.resolution)
		j = np.floor(x / self.resolution)
		cell = (i,j)
		return cell
	
	
	def _convertCellToCoordinates(self, cell):
		
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
	
		coordinates = (x,y)
		return coordinates
	
	
	def _convertCellToIndex(self, cell):
		i = cell[0]
		j = cell[1]
		
		cells_i = self.shape[0]
		cells_j = self.shape[j]
		
		assert(0 <= i < cells_i)
		assert(0 <= j < cells_j)
		
		return i + j*self.cells_i
	

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

		self.obstacle = obstacle
	
	
	def getObstacleProbabilityBetweenCells(self, cell_1, cell_2):
		
		cell_1 = np.array(cell_1)
		cell_2 = np.array(cell_2)
		vector = cell_2 - cell_1
		unit_v = (vector / np.sqrt(vector[0]**2 + vector[1]**2))
		
		print(unit_v)
		
		cells = [cell_1]
		length = 1
				
		while(not np.array_equal(cells[-1], cell_2)):
			current_cell = (cell_1 + length*unit_v).astype(int)
			length +=1
			if(not np.array_equal(current_cell, cells[-1])):
				cells += [current_cell]
			
		free_probability = 1
		for cell in cells:
			free_probability *= (1-self.obstacle.getCell(cell))
			
		occupied_probability = (1-free_probability)
		return occupied_probability
	
	
	def getObstacleProbability(self, cell):
		return self.obstacle.getCell(cell)
		
	
	
	
"""	
o = Lattice2DScalar.fromPGM("/home/gongora/Documents/MAPIR/2018 - Gas mapping/Simulation/environments/office/gasmap1.pgm", 0.1).normalize().invert().plot()
t = DiscreteObstacleMap(o)
cell_1 = (40,10)
cell_2 = (1,1)
print(t.getObstacleBetweenCells(cell_1,cell_2))
"""