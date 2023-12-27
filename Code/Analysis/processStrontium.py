import preprocessing

def makeStrontiumGP(plot=False):
    import numpy,pandas
    import preprocessing
    from geochemistry_helpers import GaussianProcess,Sampling

    data = preprocessing.getData("strontium")

    interpolation_ages = preprocessing.interpolation_ages

    strontium_x = preprocessing.strontium_x
    strontium_constraints = [Sampling.Distribution(strontium_x,"Gaussian",(row["strontium"],row["strontium_uncertainty"]),location=row["age"]).normalise() for index,row in data.iterrows()]

    strontium_gp,strontium_mean_gp = GaussianProcess().constrain(strontium_constraints).removeLocalMean(fraction=(5,5))
    strontium_no_mean_gp = strontium_gp.setKernel("rbf",(0.00005,10),specified_mean=0).query(interpolation_ages)
    strontium_combined_gp = strontium_no_mean_gp.addLocalMean(strontium_mean_gp).getSamples(1000)

    strontium_combined_gp.toJSON("./Data/Output/strontium_GP.json")

    if plot:        
        from matplotlib import pyplot
        
        strontium_combined_gp.plotSamples()
        strontium_combined_gp.plotMean(color="red")
        strontium_combined_gp.plotConstraints(color="black")

        pyplot.ylim((0.706,0.710))
        pyplot.show()
    return strontium_combined_gp

def generateNormalisedStrontium(plot=False):
    import json,numpy
    import preprocessing
    from geochemistry_helpers import Sampling,GaussianProcess

    data = preprocessing.getData("boron")

    with open("./Data/Output/strontium_GP.json") as file:
        strontium_data = json.loads(file.read())
    strontium_means = strontium_data["means"]

    strontium_minimum = min(strontium_means[1])
    strontium_maximum = max(strontium_means[1])
    strontium_range = strontium_maximum-strontium_minimum

    normalised_strontium_means = [((numpy.array(strontium_mean)-strontium_minimum)/strontium_range)-1 for strontium_mean in strontium_means]
    
    normalised_x = numpy.arange(-1,1,1e-3)
    normalised_constraints = []
    normalised_constraints += [Sampling.Distribution(normalised_x,"Gaussian",(normalised_strontium_means[1][index],0.01),location=location).normalise() for index,location in enumerate(strontium_data["locations"][1])] 

    normalised_strontium_gp = GaussianProcess().constrain(normalised_constraints)
    normalised_strontium_gp.means = normalised_strontium_means

    if plot:
        from matplotlib import pyplot
        pyplot.plot(strontium_data["locations"],normalised_strontium_means)
        pyplot.show()
    
    return normalised_strontium_gp

