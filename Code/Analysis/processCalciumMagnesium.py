from geochemistry_helpers import Sampling,GaussianProcess
from matplotlib import pyplot
import preprocessing
import numpy

def calculateCalciumMagnesium():
    calcium_magnesium = preprocessing.importData("calcium_magnesium")

    calcium_magnesium["calcium"] = calcium_magnesium["calcium"]/1e3
    calcium_magnesium["magnesium"] = calcium_magnesium["magnesium"]/1e3

    calcium_magnesium["calcium_low"] = calcium_magnesium["calcium_low"]/1e3
    calcium_magnesium["calcium_high"] = calcium_magnesium["calcium_high"]/1e3
    calcium_magnesium["calcium_uncertainty"] = (calcium_magnesium["calcium_high"]-calcium_magnesium["calcium_low"])/2

    calcium_magnesium["magnesium_low"] = calcium_magnesium["magnesium_low"]/1e3
    calcium_magnesium["magnesium_high"] = calcium_magnesium["magnesium_high"]/1e3
    calcium_magnesium["magnesium_uncertainty"] = (calcium_magnesium["magnesium_high"]-calcium_magnesium["magnesium_low"])/2
    calcium_magnesium["magnesium_uncertainty"] = calcium_magnesium["magnesium_uncertainty"].fillna(calcium_magnesium["magnesium_uncertainty"].max())

    calcium_magnesium["calcium_uncertainty"] = calcium_magnesium["calcium_uncertainty"].fillna(calcium_magnesium["calcium_uncertainty"].max())

    ion_x = numpy.arange(0,100.1,0.1)/1e3

    calcium_constraints =  [Sampling.Sampler(ion_x,"Gaussian",(calcium,calcium_uncertainty),"Monte_Carlo",location=age).normalise() for age,calcium,calcium_uncertainty in zip(calcium_magnesium["age"].values,calcium_magnesium["calcium"].values,calcium_magnesium["calcium_uncertainty"].values,strict=True) if not numpy.isnan(calcium)]
    magnesium_constraints =  [Sampling.Sampler(ion_x,"Gaussian",(magnesium,magnesium_uncertainty),"Monte_Carlo",location=age).normalise() for age,magnesium,magnesium_uncertainty in zip(calcium_magnesium["age"].values,calcium_magnesium["magnesium"].values,calcium_magnesium["magnesium_uncertainty"].values,strict=True) if not numpy.isnan(magnesium)]

    calcium_gp = GaussianProcess().constrain(calcium_constraints).setKernel("rbf",(1,50)).query(preprocessing.interpolation_ages)
    magnesium_gp = GaussianProcess().constrain(magnesium_constraints).setKernel("rbf",(1,50)).query(preprocessing.interpolation_ages)

    return (calcium_gp,magnesium_gp)
