"""
    +----------------------------------++----------------------------------+
    |                                                                      |
    |                              GDM - COMMON                            |
    |                                                                      |
    | Copyright (c) 2019, Andres Gongora - www.andresgongora.com           |
    | Machine Perception and Intelligent Robotics - http://mapir.uma.es    |
    |                                                                      |
    | This program is free software: you can redistribute it and/or modify |
    | it under the terms of the GNU General Public License as published by |
    | the Free Software Foundation, either version 3 of the License, or    |
    | (at your option) any later version.                                  |
    |                                                                      |
    | This program is distributed in the hope that it will be useful,      |
    | but WITHOUT ANY WARRANTY; without even the implied warranty of       |
    | MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the        |
    | GNU General Public License for more details.                         |
    |                                                                      |
    | You should have received a copy of the GNU General Public License    |
    | along with this program. If not, see <http://www.gnu.org/licenses/>. |
    |                                                                      |
    +----------------------------------------------------------------------+
"""

__author__		= "Andres Gongora"
__copyright__	= "2019, Andres Gongora"
__license__ 		= "GPLv3"
__credits__ 		= []


from aux import *

###############################################################################


class MapperBase:
	
	def __init__(self):
		
		self.observations = []
		
		return
	
	
	
	def update(self):
		self._update()
		return self
	
	
	
	def addObservation(self, observation):
	
		if type(observation) is list:
			assert(type(observation[0]) is Observation)
			self.observations += observation
		else:
			assert(type(observation) is Observation)
			self.observations += [observation]
		
		return self

			
mb = MapperBase()
o = Observation()
mb.addObservation([o])