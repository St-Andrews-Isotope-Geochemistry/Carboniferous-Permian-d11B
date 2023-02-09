from matplotlib import pyplot
from geochemistry_helpers import Sampling,GaussianProcess
from processStrontium import makeStrontiumGP
from processLithium import makeLithiumGP
import numpy,pandas
import preprocessing,processCalciumMagnesium

from cbsyst import boron_isotopes,Csys
from cbsyst.helpers import Bunch
import kgen

markov_chain = Sampling.MarkovChain().fromJSON("./Data/Output/markov_chain.json")

# d11B4_gp = GaussianProcess().setQueryLocations([preprocessing.interpolation_ages[0]],preprocessing.d11B_x)
d11B4_25_gp = GaussianProcess().setQueryLocations(preprocessing.interpolation_ages,preprocessing.d11B_x)
dic_gp = GaussianProcess().setQueryLocations(preprocessing.interpolation_ages,preprocessing.carbon_x)
d11Bsw_gp = GaussianProcess().setQueryLocations(preprocessing.interpolation_ages,preprocessing.d11B_x)
pH_gp = GaussianProcess().constrain(preprocessing.pH_prior).setQueryLocations(preprocessing.interpolation_ages)    
co2_gp = GaussianProcess().constrain(preprocessing.co2_prior).setQueryLocations(preprocessing.interpolation_ages)


# d11B4_gp.fromMCMCSamples(d11B4_markov_chain.accumulate("d11B4"))
d11B4_25_gp.fromMCMCSamples(markov_chain.accumulate("d11B4"))
d11Bsw_gp.fromMCMCSamples(markov_chain.accumulate("d11Bsw"))
pH_gp.fromMCMCSamples(markov_chain.accumulate("pH"))
co2_gp.fromMCMCSamples(markov_chain.accumulate("co2"))
dic_gp.fromMCMCSamples(markov_chain.accumulate("dic"))

d18O = preprocessing.data["d18O"].to_numpy()
temperature = 15.7 - 4.36*((d18O-(-1))) + 0.12*((d18O-(-1))**2)
temperature_constraints = [Sampling.Distribution(preprocessing.temperature_x,"Gaussian",(temperature,0.1),location=location).normalise() for temperature,location in zip(temperature,preprocessing.data["age"].to_numpy(),strict=True)]
temperature_gp,temperature_mean_gp = GaussianProcess().constrain(temperature_constraints).removeLocalMean(fraction=(10,2))
temperature_gp = temperature_gp.setKernel("rbf",(1,5),specified_mean=0).query(preprocessing.interpolation_ages).addLocalMean(temperature_mean_gp)

calcium_gp,magnesium_gp = processCalciumMagnesium.calculateCalciumMagnesium()
strontium_gp = makeStrontiumGP()
lithium_gp = makeLithiumGP()

Kb = Bunch({"KB":kgen.calc_K("KB",TempC=temperature,Ca=calcium_gp.means[0][0],Mg=magnesium_gp.means[0][0])})
pH_values = boron_isotopes.calculate_pH(Kb,30,preprocessing.data["d11B4"].to_numpy(),preprocessing.epsilon)

Kb_25 = Bunch({"KB":kgen.calc_K("KB",TempC=25,Ca=calcium_gp.means[0][0],Mg=magnesium_gp.means[0][0])})
d11B4_at_25 = boron_isotopes.calculate_d11B4(pH_values,Kb_25,30,preprocessing.epsilon)


figure_1,axes_1 = pyplot.subplots(nrows=7,sharex=True,figsize=(6,8))

axes_1[0].scatter(preprocessing.interpolation_ages[0],d11B4_at_25,color="orange",alpha=0.5,edgecolors="black",marker="D",zorder=3)
axes_1[0].scatter(preprocessing.interpolation_ages[0],preprocessing.data["d11B4"].to_numpy(),color="green",alpha=0.3,edgecolors="black",marker="D",zorder=0)
d11B4_25_gp.plotMean(axis=axes_1[0],group=1,color="grey",zorder=2)
d11B4_25_gp.plotArea(axis=axes_1[0],group=1,color="blue",alpha=0.5,zorder=1,set_axes=False)

axes_1[0].set_ylim((5,20))
axes_1[0].set_ylabel("$\delta^{11}B_{4}$")

from matplotlib.patches import Polygon,Rectangle

strontium_gp.plotMean(color="black",axis=axes_1[1],zorder=2)
strontium_gp.plotConstraints(color="#e98e42",axis=axes_1[1],fmt="o",zorder=1)
axes_1[1].set_ylabel("$^{87}/_{86}$Sr")

lithium_gp.plotMean(color="black",axis=axes_1[2],zorder=2)
lithium_gp.plotConstraints(color="#7bc660",axis=axes_1[2],fmt="o",zorder=1)
axes_1[2].set_ylabel("$\delta^{7}$Li")

d11Bsw_gp.plotPcolor(axis=axes_1[3],invert_x=True,colourbar=False,map="Purples",mask=True,vmin=0.001,vmax=0.04)
axes_1[3].set_ylabel("$\delta^{11}B_{sw}$")
axes_1[3].set_ylim((20,40))

x = pH_gp.queries[-1][0].bin_midpoints
y = numpy.array([query.location for query in pH_gp.queries[-1]])
probabilities = numpy.transpose(numpy.array([query.probabilities for query in pH_gp.queries[-1]]))
probabilities = 5*(probabilities/numpy.max(probabilities))
probabilities[probabilities>1] = 1
probabilities[probabilities<0.01] = 0
cmap = pyplot.get_cmap('rainbow')

for x_index in range(len(x)-1):
    for y_index in range(len(y)-1):
        if probabilities[x_index,y_index]!=0:
            rect = Rectangle((y[y_index],x[x_index]), y[y_index+1]-y[y_index], x[x_index+1]-x[x_index], 
                            facecolor=cmap(255-int((x[x_index]-7.5)*(255/0.75))), alpha=probabilities[x_index,y_index], edgecolor='none')
            axes_1[4].add_patch(rect)


axes_1[4].set_ylabel("pH")
axes_1[4].set_ylim((7.0,9.0))

minmax = (max(preprocessing.equally_spaced_ages),min(preprocessing.equally_spaced_ages))
dic_gp.plotArea(axis=axes_1[5],group=1,color="red",alpha=0.3)
dic_gp.plotMedian(axis=axes_1[5],group=1,color="red",zorder=2)


axes_1[5].set_ylabel("DIC")
axes_1[5].set_ylim((0,10000))

co2_data = pandas.read_excel("./Data/Input/CO2.xlsx",usecols="A:B",names=["age","co2"],header=0,sheet_name="Matlab")
axes_1[6].scatter(co2_data["age"],co2_data["co2"],color="black",marker=".",zorder=3)

co2_gp.plotArea(axis=axes_1[6],group=1,color="blue",alpha=0.3)
co2_gp.plotMedian(axis=axes_1[6],group=1,color="blue",zorder=2)

axes_1[6].set_ylabel("CO$_2$")
axes_1[6].set_ylim((0,6000))

axes_1[-1].set_xlabel("Age (Ma)")
axes_1[-1].set_xlim((max(preprocessing.equally_spaced_ages),min(preprocessing.equally_spaced_ages)))

figure_1.savefig("./Figures/stacked_plot.png",dpi=600)

a = 5