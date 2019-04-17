import numpy as np
import re
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


##============================================================================
FIG_SIZE = (8,8)


def plotScalar(matrix, vmin=0, vmax=0, mask=0, save="", scale="lin"):

	fig = plt.figure(num=None, figsize=FIG_SIZE, dpi=100, facecolor='w', edgecolor='w')
	ax = fig.add_subplot(111)
	if vmax == 0:
		vmax = matrix.max()
		
	vmin = matrix.min()
	if vmin > 0.0:
		vmin = 0.0
		
	if scale=="lin":
		cax = ax.matshow(matrix, interpolation='nearest', vmin=vmin, vmax=vmax)#, norm=LogNorm(0.0001,self._m.max()))
	elif scale=="log":
		cax = ax.matshow(matrix, interpolation='nearest', norm=LogNorm(0.0001,vmax))
		
	if(mask==1):
		"""
		ax.spines['left'].set_position('center')
		ax.spines['right'].set_color('outward')
		ax.spines['bottom'].set_position('outward')
		ax.spines['top'].set_position('outward')
		ax.spines['right'].set_position('outward')
		ax.spines['top'].set_color('outward')
		"""
		plt.axis('off')
	else:
		fig.colorbar(cax)
	
	if(save == ""):
		plt.show()
	else:
		fig.savefig(save)
	return


def plotVector(m_i, m_j, interpol = 5, scale = 20, vmax=0, mask=0, save=""):
	magnitude = np.sqrt(m_i**2 + m_j**2)
	cells_i, cells_j = m_i.shape
	vector_i = np.zeros((cells_i, cells_j))
	vector_j = np.zeros((cells_i, cells_j))
	
	if interpol == 5:
		vector_i[::interpol, ::interpol]  = (m_i[0::interpol, 0::interpol] + m_i[1::interpol, 1::interpol] + m_i[2::interpol, 2::interpol] )/3
		vector_j[::interpol, ::interpol]  = (m_j[0::interpol, 0::interpol] + m_j[1::interpol, 1::interpol] + m_j[2::interpol, 2::interpol] )/3
	else:
		vector_i[::interpol, ::interpol]  = m_i[::interpol, ::interpol]
		vector_j[::interpol, ::interpol]  = m_j[::interpol, ::interpol]
	
	if vmax == 0:
		vmax = magnitude.max()
	
	
	fig=plt.figure(num=None, figsize=FIG_SIZE, dpi=100, facecolor='w', edgecolor='k')
	plt.imshow(magnitude, extent=[0, cells_j-1, cells_i-1, 0], cmap='Blues', interpolation='nearest', vmax=vmax)
	plt.quiver(vector_j, -vector_i, scale=scale, minlength=0.001)
	
	if(mask==1):
		"""
		plt.axes().spines['left'].set_position('none')
		plt.axes().spines['right'].set_color('none')
		plt.axes().spines['bottom'].set_position('none')
		plt.axes().spines['top'].set_position('none')
		plt.axes().spines['right'].set_position('none')
		plt.axes().spines['top'].set_color('none')
		"""
		plt.axis('off')

	
	
	
	if(save == ""):
		plt.show()
	else:
		fig.savefig(save)
	return


def plotScalarVector(m_a, m_i, m_j, interpol=3, scale=20, vmax=0.0, save=""):
	
	magnitude = m_a
	cells_i, cells_j = m_i.shape
	vector_i = np.zeros((cells_i, cells_j))
	vector_j = np.zeros((cells_i, cells_j))
	
	if interpol == 3:
		vector_i[::interpol, ::interpol]  = (m_i[0::interpol, 0::interpol] + m_i[1::interpol, 1::interpol] + m_i[2::interpol, 2::interpol] )/3
		vector_j[::interpol, ::interpol]  = (m_j[0::interpol, 0::interpol] + m_j[1::interpol, 1::interpol] + m_j[2::interpol, 2::interpol] )/3
	else:
		vector_i[::interpol, ::interpol]  = m_i[::interpol, ::interpol]
		vector_j[::interpol, ::interpol]  = m_j[::interpol, ::interpol]
	
	if vmax == 0.0:
		vmax = magnitude.max()
	
	fig=plt.figure(num=None, figsize=FIG_SIZE, dpi=100, facecolor='w', edgecolor='k')
	plt.imshow(magnitude, extent=[0, cells_j-1, cells_i-1, 0], cmap='Blues', interpolation='nearest', vmax=vmax)
	plt.quiver(vector_j, -vector_i, scale=scale, minlength=0.001)
	
	
	
	if(save == ""):
		plt.show()
	else:
		fig.savefig(save)
	return




##============================================================================
class Lattice2D:
	
	def __init__(self, shape, resolution=0.0):

		assert (type(shape) == tuple)
		assert (shape[0] >= 0)
		assert (shape[1] >= 0)
		assert (resolution >= 0.0)
		
		self.shape = shape
		self.resolution = resolution
		
		
	def getCell(self, cell):
		if(type(cell) is np.ndarray):
			cell = (cell[0], cell[1])
		
		self._checkCell(cell)
		return self._getCell(cell)
	
	
	def setCell(self, cell, value):
		self._checkCell(cell)
		self._setCell(self, cell)
		return self

	
	def getPosition(self, position):
		return self.getCell(self._positionToCoordinates(position))

	
	def setPosition(self, position, value):
		self.setCell(self._positionToCoordinates(position), value)
		return self
	
	
	def plot(self):
		return self
	
	
	def normalize(self):
		return self._normalize()

	## --------------------------------------------------------------------------

	def _checkCell(self, cell):
		assert(type(cell) is tuple)
		
		i = cell[0]
		j = cell[1]
		total_cells_i = self.shape[0]
		total_cells_j = self.shape[1]
		
		assert i >= 0
		assert j >= 0
		assert i < total_cells_i
		assert j < total_cells_j
		
		
	def _position2Coordinates(self, position, ):
		x = position[0]
		y = position[1]
		total_cells_i = self.shape[0]
		total_cells_j = self.shape[1]
		
		i = total_cells_i - int(y/self._resolution) - 1
		j = int(x/self._resolution)	
		return (i,j)		


	def _positionToCoordinates(self, position):
		coordinates = self._position2Coordinates(position)
		self._checkCoordinates((coordinates))
		return coordinates
	


	
	
	
##============================================================================
class Lattice2DScalar(Lattice2D):
	
	def __init__(self, shape=(0,0), resolution=0, init_value=0):
		Lattice2D.__init__(self, shape, resolution)
		self._data = np.zeros(self.shape)
		self._data[:,:] = init_value
		return
	
	
	@classmethod
	def fromMatrix(self, matrix, resolution):	
		shape = matrix.shape
		instance = Lattice2DScalar(shape, resolution)
		instance.loadMatrix(matrix)
		return instance
	
	
	@classmethod
	def fromPGM(self, pgm_file, resolution, byteorder='>'):
		with open(pgm_file, 'rb') as f:
			buffer = f.read()
			try:
				header, width, height, maxval = re.search(
					b"(^P5\s(?:\s*#.*[\r\n])*"
					b"(\d+)\s(?:\s*#.*[\r\n])*"
					b"(\d+)\s(?:\s*#.*[\r\n])*"
					b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer
				).groups()
			except AttributeError:
				raise ValueError("Not a raw PGM file: '%s'" % pgm_file)
			
			matrix = np.frombuffer(
				buffer,
				dtype='u1' if int(maxval) < 256 else byteorder+'u2',
				count=int(width)*int(height),
				offset=len(header)
			).reshape((int(height), int(width)))
			
			return Lattice2DScalar.fromMatrix(matrix, resolution)
		

	def toMatrix(self):
		return self._data


	def plot(self, vmin=0, vmax=0, mask=0, save=""):
		plotScalar(self._data, vmin, vmax, mask, save=save)
		return self
	
	
	def invert(self):
		vmax = self._data.max()
		vmin = self._data.min()
		self._data = vmax - self._data + vmin
		return self
		
	def loadMatrix(self, matrix):
		assert (self.shape == matrix.shape)
		self._data = matrix
		return self
	
	## --------------------------------------------------------------------------		
	
	def _getCell(self, coordinates):
		return self._data[coordinates]
	
	
	def _setCell(self, cell, value):
		self._data[cell] = value
		return self
	
	
	def _normalize(self):
		vmax = self._data.max()
		vmin = self._data.min()
		self._data = (self._data-vmin)/(vmax-vmin)
		return self
	
	
	
##============================================================================
class Lattice2DVector(Lattice2D):
	
	def __init__(self, shape=(0,0), resolution=0, init_value=(0,0)):		
		Lattice2D.__init__(self, shape, resolution)
		self._data_i = np.zeros(self.shape)
		self._data_i[:,:] = init_value[0]
		self._data_j = np.zeros(self.shape)
		self._data_j[:,:] = init_value[1]
		return
		
	
	@classmethod
	def fromMatrix(self, matrix_i, matrix_j, resolution=0):
		assert(matrix_i.shape == matrix_j.shape)
		shape = matrix_i.shape
		instance = Lattice2DVector(shape, resolution)
		instance._loadMatrix(matrix_i, matrix_j)
		return instance


	def toMatrix(self, index):
		return (self._data_i, self._data_j)
	
	
	def rotateVectors(self,angle):
		self._data_i = self._data_i*np.cos(angle) - self._data_j*np.sin(angle)
		self._data_j = self._data_i*np.sin(angle) + self._data_j*np.cos(angle)
		return self	
	
	
	def plot(self, interpol=3, scale=15, vmax=0, mask=1, save=""):
		plotVector(self._i, self._j, interpol=interpol, vmax=vmax, scale=scale, mask=mask,save=save)
		return self
	
	## --------------------------------------------------------------------------		
	
	def _loadMatrix(self, matrix_i, matrix_j):
		assert(matrix_i.shape == matrix_j.shape)
		self.data_i = matrix_i
		self.data_j = matrix_j
		return self
	
	
	def _getCell(self, coordinates):
		return (self.data_i[coordinates], self.data_j[coordinates])
	
	
	def _setCell(self, coordinates, value):
		self.data_i[coordinates] = value[0]
		self.data_j[coordinates] = value[1]
		return self
	

	def _plot(self, interpol=3, scale=15, vmax=0, mask=1, save=""):
		#plotVector(self._i, self._j, interpol=interpol, vmax=vmax, scale=scale, mask=mask,save=save)
		print("Not implemented yet!!")
		return self
	

	def _normalize(self):
		length = np.sqrt(self._i**2 + self._j**2)
		self._data_i /= length
		self._data_j /= length
		return self
	

	
	
	
	
"""	
Lattice2D((10,10),0.1)
print(Lattice2DScalar.fromMatrix(np.ones((5,5)), 0.2)._data)
print(Lattice2DVector.fromMatrix(np.ones((5,5)), np.ones((5,5)), 0.2)._data_i)
Lattice2DScalar.fromPGM("/home/gongora/Documents/MAPIR/2018 - Gas mapping/Simulation/environments/office/gasmap1.pgm").plot()
"""

