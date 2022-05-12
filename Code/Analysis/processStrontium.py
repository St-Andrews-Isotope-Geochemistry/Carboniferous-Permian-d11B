import preprocessing

def makeStrontiumGP(plot=False):
    import numpy,pandas
    import preprocessing
    from geochemistry_helpers import GaussianProcess,Sampling

    data = preprocessing.getData("strontium")

    interpolation_ages = [preprocessing.equally_spaced_ages]

    strontium_x = preprocessing.strontium_x
    strontium_constraints = [Sampling.Distribution(strontium_x,"Gaussian",(row["Sr87"],row["Sr87_SD"]),location=row["Age"]).normalise() for index,row in data.iterrows()]

    strontium_gp = GaussianProcess().constrain(strontium_constraints).setKernel("rbf",(0.0002,10),specified_mean=False).query(interpolation_ages).getSamples(1000)

    strontium_gp.toJSON("./Data/Strontium_GP.json")

    if plot:        
        from matplotlib import pyplot
        
        strontium_gp.plotSamples()
        strontium_gp.plotMean(color="red")
        strontium_gp.plotConstraints(color="black")

        pyplot.ylim((0.706,0.710))
        pyplot.show()
    return strontium_gp

def generateNormalisedStrontium(plot=False):
    import json,numpy
    import preprocessing
    from geochemistry_helpers import Sampling,GaussianProcess

    data = preprocessing.getData("boron")

    interpolation_ages = [data["Age"].to_numpy(),preprocessing.equally_spaced_ages]

    with open("./Data/Strontium_GP.json") as file:
        strontium_data = json.loads(file.read())
    strontium_means = numpy.array(strontium_data["means"][0])

    normalised_strontium_means = (strontium_means-min(strontium_means))/(max(strontium_means)-min(strontium_means))
    
    normalised_x = numpy.arange(0,1,1e-3)
    normalised_constraints = []
    normalised_constraints += [Sampling.Distribution(normalised_x,"Gaussian",(normalised_strontium_means[index],0.01),location=location).normalise() for index,location in enumerate(strontium_data["locations"][0])] 

    normalised_strontium_gp = GaussianProcess().constrain(normalised_constraints).setKernel("rbf",(0.01,1),specified_mean=False).query(interpolation_ages).getSamples(1000)

    if plot:
        from matplotlib import pyplot
        pyplot.plot(strontium_data["locations"],normalised_strontium_means)
        pyplot.show()
    
    return normalised_strontium_gp

