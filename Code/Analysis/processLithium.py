import preprocessing

def makeLithiumGP(plot=False):
    import numpy,pandas
    import preprocessing
    from geochemistry_helpers import GaussianProcess,Sampling

    data = preprocessing.getData("lithium")

    interpolation_ages = [preprocessing.equally_spaced_ages]

    lithium_x = preprocessing.lithium_x
    lithium_constraints = [Sampling.Distribution(lithium_x,"Gaussian",(row["lithium"],0.1),location=row["age"]).normalise() for index,row in data.iterrows()]

    lithium_gp,lithium_mean_gp = GaussianProcess().constrain(lithium_constraints).removeLocalMean(fraction=(2,2))
    lithium_no_mean_gp = lithium_gp.setKernel("rbf",(0.1,6),specified_mean=0).query(interpolation_ages)
    lithium_combined_gp = lithium_no_mean_gp.addLocalMean(lithium_mean_gp).getSamples(1000)

    lithium_combined_gp.toJSON("./Data/Output/lithium_GP.json")

    if plot:        
        from matplotlib import pyplot
        
        lithium_combined_gp.plotSamples()
        lithium_combined_gp.plotMean(color="red")
        lithium_combined_gp.plotConstraints(color="black")

        pyplot.ylim((0.706,0.710))
        pyplot.show()
    return lithium_combined_gp

def generateNormalisedLithium(plot=False):
    import json,numpy
    import preprocessing
    from geochemistry_helpers import Sampling,GaussianProcess

    data = preprocessing.getData("boron")

    interpolation_ages = [data["age"].to_numpy(),preprocessing.equally_spaced_ages]

    with open("./Data/Output/lithium_GP.json") as file:
        lithium_data = json.loads(file.read())
    lithium_means = numpy.array(lithium_data["means"][0])

    normalised_lithium_means = (lithium_means-min(lithium_means))/((max(lithium_means)-min(lithium_means))/2)-1
    
    normalised_x = numpy.arange(-1,1,1e-3)
    normalised_constraints = []
    normalised_constraints += [Sampling.Distribution(normalised_x,"Gaussian",(normalised_lithium_means[index],0.01),location=location).normalise() for index,location in enumerate(lithium_data["locations"][0])] 

    normalised_lithium_gp = GaussianProcess().constrain(normalised_constraints).setKernel("rbf",(0.01,1),specified_mean=False).query(interpolation_ages).getSamples(1000)

    if plot:
        from matplotlib import pyplot
        pyplot.plot(lithium_data["locations"],normalised_lithium_means)
        pyplot.show()
    
    return normalised_lithium_gp

