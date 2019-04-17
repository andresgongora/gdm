from gmrf import gmrf_gas
import common

print("gmrf_gas:" + str(__name__))
if(__name__ is "__main__"):

	o = common.Lattice2DScalar.fromPGM("../../../common/data/test_environments/small_office/3/obstacles.pgm", 0.1).normalize().invert()
	om = common.DiscreteObstacleMap(o)
	g = gmrf_gas.GMRF_Gas_Parallel(om)

	obs = [common.Observation((5,5), 1), common.Observation((5,6), 1.8), common.Observation((6,5), 0.2)]
	g.addObservation(obs)
	g.predict()
	g.gas.plot()
	g.computeUncertainty()
	g.gas_uncertainty.plot()
