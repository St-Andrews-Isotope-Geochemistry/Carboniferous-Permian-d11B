from geochemistry_helpers import Sampling,GaussianProcess
from matplotlib import pyplot
import preprocessing
import numpy

def calculateCalciumMagnesium():
    calcium_magnesium = preprocessing.importData("calcium_magnesium")

    calcium_magnesium["calcium"] = calcium_magnesium["calcium"]/1e3
    calcium_magnesium["magnesium"] = calcium_magnesium["magnesium"]/1e3

    ion_x = numpy.arange(0,100.1,0.1)/1e3

    calcium_constraints =  [Sampling.Sampler(ion_x,"Gaussian",(calcium,5e-3),"Monte_Carlo",location=age).normalise() for age,calcium in zip(calcium_magnesium["age"].to_numpy(),calcium_magnesium["calcium"].to_numpy(),strict=True) if not numpy.isnan(calcium)]
    magnesium_constraints =  [Sampling.Sampler(ion_x,"Gaussian",(magnesium,10e-3),"Monte_Carlo",location=age).normalise() for age,magnesium in zip(calcium_magnesium["age"].to_numpy(),calcium_magnesium["magnesium"].to_numpy(),strict=True) if not numpy.isnan(magnesium)]

    calcium_gp = GaussianProcess().constrain(calcium_constraints).setKernel("rbf",(1,50)).query(preprocessing.interpolation_ages)
    magnesium_gp = GaussianProcess().constrain(magnesium_constraints).setKernel("rbf",(1,50)).query(preprocessing.interpolation_ages)

    return (calcium_gp,magnesium_gp)
