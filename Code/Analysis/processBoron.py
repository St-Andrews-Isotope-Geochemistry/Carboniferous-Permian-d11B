def calculatepH(plot=False):
    import numpy
    from matplotlib import pyplot

    from geochemistry_helpers import Sampling,GaussianProcess
    
    from cbsyst import boron_isotopes
    from cbsyst.helpers import Bunch
    import kgen

    from processStrontium import makeStrontiumGP,generateNormalisedStrontium
    import preprocessing


    # Load the data
    data = preprocessing.getData("boron")
    boron_age = data["Age"].to_numpy()
    d11B4 = data["d11B_Borate"].to_numpy()
    d11B4_uncertainty = data["d11B_Uncertainty"].to_numpy()

    d11B4_uncertainty[d11B4_uncertainty==0] = 0.3

    interpolation_ages = [boron_age,preprocessing.equally_spaced_ages]

    strontium_gp = makeStrontiumGP()
    normalised_strontium_gp = generateNormalisedStrontium()
    d11Bsw_range = getd11BswRange(d11B4)
    d11B_minimum = numpy.argmin(d11B4)
    d11B_maximum = numpy.argmax(d11B4)

    # Set up some priors
    # Can reuse the same one if the prior is the same across the time series
    
    # Boron
    d11B4_priors = [Sampling.Sampler(preprocessing.d11Bsw_x,"Gaussian",(d11B4,uncertainty),"Monte_Carlo",location=age).normalise() for age,d11B4,uncertainty in zip(boron_age,d11B4,d11B4_uncertainty,strict=True)]
    pH_prior = [Sampling.Distribution(preprocessing.pH_x,"Gaussian",(8,1)).normalise()]
    d11Bsw_priors = [Sampling.Sampler(preprocessing.d11Bsw_x,"Flat",(range[0],range[1]),"Monte_Carlo",location=age).normalise() for age,range in zip(boron_age,d11Bsw_range,strict=True)]
    
    d11Bsw_starting_prior = d11Bsw_priors[d11B_minimum]
    d11Bsw_scaling_prior = Sampling.Sampler(numpy.arange(-1000,1000,0.1),"Flat",(0,10),"Monte_Carlo").normalise()
    
    d11B4_gp = GaussianProcess().constrain(d11B4_priors).setKernel("rbf",(1,5)).query(interpolation_ages)
    d11Bsw_gp = GaussianProcess().constrain(d11Bsw_priors).setKernel("rbf",(1,10)).query(interpolation_ages)
    pH_gp = GaussianProcess().constrain(pH_prior).setQueryLocations(interpolation_ages)

    total_count = 0
    accepts = 0
    rejects = 0
    count = 0
    
    d11Bsw_scaling_jitter_sampler = Sampling.Sampler(numpy.arange(-100,100,1e-3),"Gaussian",(0,0.5),"Monte_Carlo").normalise()
    d11Bsw_scaling_prior.getSamples(1)
    d11Bsw_scaling_jitter_sampler.getSamples(1)
    d11Bsw_scaling = (d11Bsw_scaling_prior.samples[0],)
    
    d11Bsw_start_jitter_sampler = Sampling.Sampler(numpy.arange(-100,100,1e-3),"Gaussian",(0,1),"Monte_Carlo").normalise()
    d11Bsw_starting_prior.getSamples(1)
    d11Bsw_start = (d11Bsw_starting_prior.samples[0],)

    d11Bsws = d11Bsw_start[0]+d11Bsw_scaling[0]*numpy.squeeze(normalised_strontium_gp.means[0])
    d11Bsw_to_store = [d11Bsw_start[0]+d11Bsw_scaling[0]*numpy.squeeze(norm_mean) for norm_mean in normalised_strontium_gp.means]
            
    probability_store = (-1e6,)
    d11Bsw_sample_store = (d11Bsw_to_store,)

    epsilon = 27.2

    Kb = Bunch({"KB":kgen.calc_K("KB")})
    pH = boron_isotopes.calculate_pH(Kb,d11Bsws,d11B4,epsilon)

    pH_to_store = [boron_isotopes.calculate_pH(Kb,d11Bsws,d11B4,epsilon) for d11B4,d11Bsws in zip(d11B4_gp.means,d11Bsw_to_store,strict=True)]

    pH_sample_store = (pH_to_store,)
    
    #pyplot.ion()
    #pyplot.show()


    while count<1000:
        total_count+=1

        print(count)
        d11Bsw_scaling_jitter_sampler.getSamples(1)
        d11Bsw_scaling_jittered = d11Bsw_scaling[-1]+d11Bsw_scaling_jitter_sampler.samples[0]
        d11Bsw_start_jitter_sampler.getSamples(1)
        d11Bsw_start_jittered = d11Bsw_start[-1] + d11Bsw_start_jitter_sampler.samples[0]

        d11Bsws = d11Bsw_start_jittered+d11Bsw_scaling_jittered*numpy.squeeze(normalised_strontium_gp.means[0])
        epsilon = 27.2

        Kb = Bunch({"KB":kgen.calc_K("KB")})
        pH = boron_isotopes.calculate_pH(Kb,d11Bsws,d11B4,epsilon)

        d11Bsw_probability = numpy.squeeze(numpy.array([d11Bsw_prior.getProbability(d11Bsw) for d11Bsw_prior,d11Bsw in zip(d11Bsw_priors,d11Bsws,strict=True)]))
        d11Bsw_scaling_probability = d11Bsw_scaling_prior.getProbability(d11Bsw_scaling_jittered)
        d11Bsw_start_probability = d11Bsw_starting_prior.getProbability(d11Bsw_start_jittered)
        if any(d11Bsw_probability==0) or d11Bsw_scaling_probability==0 or d11Bsw_start_probability==0:
            impossible = True
        else:
            impossible = False
            d11Bsw_cumulative_probability = sum(numpy.log(d11Bsw_probability))
            pH_probability = sum(pH_prior[0].getLogProbability(pH))
        
        keep = False
        if not impossible:
            cumulative_probability = pH_probability+d11Bsw_cumulative_probability
            if cumulative_probability>probability_store[-1]:
                keep = True
            else:
                probability_difference = 10**(cumulative_probability-probability_store[-1])
                r = numpy.random.random(1)[0]
                if r<=probability_difference:
                    keep = True
            

        if keep:
            d11Bsw_to_store = [d11Bsw_start_jittered+d11Bsw_scaling_jittered*numpy.squeeze(norm_mean) for norm_mean in normalised_strontium_gp.means]
            pH_to_store = [boron_isotopes.calculate_pH(Kb,d11Bsws,d11B4,epsilon) for d11B4,d11Bsws in zip(d11B4_gp.means,d11Bsw_to_store,strict=True)]

            d11Bsw_start = d11Bsw_start+(d11Bsw_start_jittered,)
            d11Bsw_scaling = d11Bsw_scaling+(d11Bsw_scaling_jittered,)
            probability_store = probability_store+(cumulative_probability,)
            d11Bsw_sample_store += (d11Bsw_to_store,)
            pH_sample_store += (pH_to_store,)

            accepts+=1
            count += 1
        else:
            rejects+=1

    print(str(round(rejects/total_count,4)*100)+"% rejected")
    print(str(round(accepts/total_count,4)*100)+"% accepted")
    
    d11Bsw_gp.fromMCMCSamples(d11Bsw_sample_store)
    pH_gp.fromMCMCSamples(pH_sample_store)
    

    figure_1,axes_1 = pyplot.subplots(nrows=3,sharex=True)

    d11Bsw_gp.plotPcolor(axis=axes_1[0],invert_x=True,colourbar=False,map="Blues",mask=True,vmin=0.01,vmax=0.04)
    d11Bsw_gp.plotConstraints(axis=axes_1[0],color="black",fmt="None",zorder=0)
    d11B4_gp.plotMean(axis=axes_1[0],group=1,color="red")
    axes_1[0].set_ylim((00,50))
    axes_1[0].set_ylabel("$\delta^{11}B_{sw}$")
    #axes_1[0].invert_xaxis()

    #pH_gp.plotArea(axis=axes_1[1],group=1)
    pH_gp.plotPcolor(axis=axes_1[1],invert_x=True,colourbar=False,map="Blues",mask=True,vmin=0.01,vmax=0.04)
    axes_1[1].set_ylabel("pH")
    axes_1[1].set_ylim((7,9))

    strontium_gp.plotMean(color="red",axis=axes_1[2],zorder=2)
    strontium_gp.plotConstraints(color="black",axis=axes_1[2],fmt="o",zorder=1)

    axes_1[-1].set_xlabel("Age (Ma)")
    axes_1[-1].set_xlim((330,230))

    ##
    figure_2,axes_2 = pyplot.subplots(nrows=1)

    pyplot.set_cmap('nipy_spectral')
    axes_2.scatter(d11Bsw_scaling,d11Bsw_start,c=range(count+1),marker="x",zorder=3)
    #axes_2.hist2d(d11Bsw_scaling,d11Bsw_start,bins=(30,30),cmap=pyplot.cm.Blues,zorder=1)

    d11Bsw_absolute_minimum = min([value[0] for value in d11Bsw_range])
    d11Bsw_flat_minimum = max([value[0] for value in d11Bsw_range])
    d11Bsw_absolute_maximum = max([value[1] for value in d11Bsw_range])
    d11Bsw_flat_maximum = min([value[1] for value in d11Bsw_range])

    minimum_d11Bsw_linspace = numpy.linspace(d11Bsw_absolute_minimum,d11Bsw_absolute_maximum,10)
    delta_d11Bsw_linspace = numpy.linspace(0,d11Bsw_absolute_maximum-d11Bsw_absolute_minimum,50)

    pH_queries = numpy.linspace(7,9,11)
    minimum_d11Bsws = boron_isotopes.calculate_d11BT(pH_queries,Kb,d11B4[d11B_minimum],epsilon)
    maximum_d11Bsws = boron_isotopes.calculate_d11BT(pH_queries,Kb,d11B4[d11B_maximum],epsilon)

    d11B_borate_maximum = []
    for delta_d11Bsw_lin in delta_d11Bsw_linspace:
        d11B_borate_maximum += [d11Bsw_flat_maximum if d11Bsw_flat_maximum+delta_d11Bsw_lin<d11Bsw_absolute_maximum else d11Bsw_absolute_maximum-delta_d11Bsw_lin]

    delta_d11Bsws = []
    for maximum_d11Bsw in maximum_d11Bsws:
        delta_d11Bsws += [maximum_d11Bsw-minimum_d11Bsw_linspace]

    axes_2.plot([0,0],[d11Bsw_flat_minimum,d11Bsw_flat_maximum],color="black",zorder=4)
    axes_2.plot([0,d11Bsw_absolute_maximum-d11Bsw_absolute_minimum],[d11Bsw_flat_minimum,d11Bsw_absolute_minimum],color="black",zorder=4)
    axes_2.plot(delta_d11Bsw_linspace,d11B_borate_maximum,color="black",zorder=4)

    import matplotlib.patches as patches
    patch = patches.Polygon([[0,0],[0,d11Bsw_flat_minimum],[d11Bsw_absolute_maximum-d11Bsw_absolute_minimum,d11Bsw_absolute_minimum]],color="white",zorder=3)
    axes_2.add_patch(patch)
    #axes_2.patch([0,d11Bsw_absolute_maximum-d11Bsw_absolute_minimum],[d11Bsw_flat_minimum,d11Bsw_absolute_minimum],color="black")
    patch2 = patches.Polygon([[0,d11Bsw_flat_maximum],[40,d11Bsw_flat_maximum],[40,50],[0,50]],color="white",zorder=3)
    axes_2.add_patch(patch2)

    for minimum_d11Bsw,pH_query in zip(minimum_d11Bsws,pH_queries):
        axes_2.plot([0,40],[minimum_d11Bsw,minimum_d11Bsw],color="gray",zorder=2)
        axes_2.text(35,minimum_d11Bsw+1,round(pH_query,1),rotation=0,color="gray")
    for delta_d11Bsw,pH_query in zip(delta_d11Bsws,pH_queries):
        axes_2.plot(delta_d11Bsw,minimum_d11Bsw_linspace,color="OliveDrab",zorder=2)
        axes_2.text(delta_d11Bsw[0]-1,minimum_d11Bsw_linspace[0],round(pH_query,1),rotation=-45,color="OliveDrab")
    
    axes_2.text(3,5.5,"pH at maximum $\delta^{11}B_{4}$",color="OliveDrab")
    axes_2.text(25,35,"pH at minimum $\delta^{11}B_{4}$",color="gray")

    axes_2.set_xlabel("$\Delta\delta^{11}B_{sw}$")
    axes_2.set_ylabel("Minimum $\delta^{11}B_{sw}$")

    axes_2.set_xlim((0,40))
    axes_2.set_ylim((d11Bsw_absolute_minimum,d11Bsw_flat_maximum+5))

    figure_3,axes_3 = pyplot.subplots(nrows=1)
    d11Bsw_gp.plotPcolor(axis=axes_3,invert_x=True,colourbar=False,map="Blues",mask=True,vmin=0.01,vmax=0.04)

    axes_3.set_xlim((330,230))
    axes_3.set_ylim((25,45))

    axes_3.set_xlabel("Age (Ma)")
    axes_3.set_ylabel("$delta^{11}B_{sw}$")
    
    ##
    #figure_3,axes_3 = pyplot.subplots(nrows=1)
    #sr = (normalised_strontium_gp.means[len(d11B4):-1]*(0.708-0.707))+0.707
    #for d11Bsws_sample in d11Bsw_gp.samples:
    #    axes_3.plot(sr,d11Bsws_sample)
    #axes_3.set_xlabel("Sr")
    #axes_3.set_ylabel("$\delta^{11}B_{sw}$")

    a = 5
    pyplot.show()
    print("End")
def getd11BswRange(d11B4,plot=False):
    from cbsyst import boron_isotopes

    d11Bsw_range = boron_isotopes.calculate_d11BT_range(d11B4)
    return d11Bsw_range

calculatepH()