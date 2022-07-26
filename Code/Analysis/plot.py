from matplotlib import pyplot
import numpy


figure_1,axes_1 = pyplot.subplots(nrows=6,sharex=True,figsize=(6,8))

for age_local,d11B4_local in zip(preprocessing.data["Age"].to_numpy(),preprocessing.data["d11B4"].to_numpy(),strict=True):
    axes_1[0].scatter(age_local,d11B4_local,color="black",alpha=0.5,edgecolors="black",marker="D")
d11B4_gp.plotMean(axis=axes_1[0],group=1,color="grey")
#d11B4_gp.plotConstraints(axis=axes_1[0],color="#dfa7a6",fmt="None",zorder=0,marker="D")
axes_1[0].set_ylabel("$\delta^{11}B_{4}$")
#axes_1[0].invert_xaxis()

from matplotlib.patches import Polygon,Rectangle
# d11B4_patch_1 = Polygon([[466,0],[466,15],[462,15],[462,0]],color="white",zorder=3,alpha=0.7)
# d11B4_patch_2 = Polygon([[442,0],[442,15],[433,15],[433,0]],color="white",zorder=3,alpha=0.7)

# axes_1[0].add_patch(d11B4_patch_1)
# axes_1[0].add_patch(d11B4_patch_2)

strontium_gp.plotMean(color="black",axis=axes_1[1],zorder=2)
strontium_gp.plotConstraints(color="#e98e42",axis=axes_1[1],fmt="o",zorder=1)
axes_1[1].set_ylabel("$^{87}/_{86}$Sr")

d11Bsw_gp.plotPcolor(axis=axes_1[2],invert_x=True,colourbar=False,map="Purples",mask=True,vmin=0.01,vmax=0.04)
axes_1[2].set_ylabel("$\delta^{11}B_{sw}$")
axes_1[2].set_ylim((20,40))

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
            axes_1[3].add_patch(rect)

#pH_gp.plotPcolor(axis=axes_1[3],invert_x=True,colourbar=False,map="Blues",mask=True,vmin=0.01,vmax=0.04)

# pH_patch_1 = Polygon([[466,7.0],[466,8.5],[462,8.5],[462,7.0]],color="white",zorder=3,alpha=0.7)
# pH_patch_2 = Polygon([[442,7.0],[442,8.5],[433,8.5],[433,7.0]],color="white",zorder=3,alpha=0.7)

# axes_1[3].add_patch(pH_patch_1)
# axes_1[3].add_patch(pH_patch_2)


axes_1[3].set_ylabel("pH")
axes_1[3].set_ylim((7.0,8.5))

minmax = (max(preprocessing.equally_spaced_ages),min(preprocessing.equally_spaced_ages))
for dic_sample in dic:
    axes_1[4].plot(minmax,[dic_sample,dic_sample])

axes_1[4].set_ylabel("DIC")

#co2_log_gp.plotArea(axis=axes_1[4],group=1,color="blue",alpha=0.3)
co2_prior[0].bin_edges = numpy.log2(co2_prior[0].bin_edges)
co2_gp.log2Samples().plotArea(axis=axes_1[5],group=1,color="blue",alpha=0.3)
#co2_gp_2.log2Samples().plotArea(axis=axes_1[4],group=1,color="red",alpha=0.3)

axes_1[5].set_yticks(numpy.log2(250*(2**numpy.array([0,1,2,3,4,5,6]))))
axes_1[5].set_yticklabels(250*(2**numpy.array([0,1,2,3,4,5,6])))

# co2_patch_1 = Polygon([[466,numpy.log2(10)],[466,numpy.log2(20000)],[462,numpy.log2(20000)],[462,numpy.log2(10)]],color="white",zorder=3,alpha=0.7)
# co2_patch_2 = Polygon([[442,numpy.log2(10)],[442,numpy.log2(20000)],[433,numpy.log2(20000)],[433,numpy.log2(10)]],color="white",zorder=3,alpha=0.7)

# axes_1[4].add_patch(co2_patch_1)
# axes_1[4].add_patch(co2_patch_2)

axes_1[5].set_ylabel("CO$_2$")
axes_1[5].set_ylim(numpy.log2([100,20000]))

axes_1[-1].set_xlabel("Age (Ma)")
axes_1[-1].set_xlim((max(preprocessing.equally_spaced_ages),min(preprocessing.equally_spaced_ages)))



# figure_1,axes_1 = pyplot.subplots(nrows=3,sharex=True)

# d11Bsw_gp.plotPcolor(axis=axes_1[0],invert_x=True,colourbar=False,map="Blues",mask=True,vmin=0.01,vmax=0.04)
# d11Bsw_gp.plotConstraints(axis=axes_1[0],color="black",fmt="None",zorder=0)
# d11B4_gp.plotMean(axis=axes_1[0],group=1,color="red")
# axes_1[0].set_ylim((00,50))
# axes_1[0].set_ylabel("$\delta^{11}B_{sw}$")
# #axes_1[0].invert_xaxis()

# #pH_gp.plotArea(axis=axes_1[1],group=1)
# pH_gp.plotPcolor(axis=axes_1[1],invert_x=True,colourbar=False,map="Blues",mask=True,vmin=0.01,vmax=0.04)
# axes_1[1].set_ylabel("pH")
# axes_1[1].set_ylim((7,9))

# strontium_gp.plotMean(color="red",axis=axes_1[2],zorder=2)
# strontium_gp.plotConstraints(color="black",axis=axes_1[2],fmt="o",zorder=1)

# axes_1[-1].set_xlabel("Age (Ma)")
# axes_1[-1].set_xlim((330,230))

##
# figure_2,axes_2 = pyplot.subplots(nrows=1)

# pyplot.set_cmap('nipy_spectral')
# axes_2.scatter(d11Bsw_scaling,d11Bsw_minimum,c=range(count+1),marker="x",zorder=5)
# #axes_2.hist2d(d11Bsw_scaling,d11Bsw_minimum,bins=(30,30),cmap=pyplot.cm.Blues,zorder=1)

# d11Bsw_absolute_minimum = min([value[0] for value in d11Bsw_range])
# d11Bsw_flat_minimum = max([value[0] for value in d11Bsw_range])
# d11Bsw_absolute_maximum = max([value[1] for value in d11Bsw_range])
# d11Bsw_flat_maximum = min([value[1] for value in d11Bsw_range])

# minimum_d11Bsw_linspace = numpy.linspace(d11Bsw_absolute_minimum,d11Bsw_absolute_maximum,100)
# delta_d11Bsw_linspace = numpy.linspace(-(d11Bsw_absolute_maximum-d11Bsw_absolute_minimum),d11Bsw_absolute_maximum-d11Bsw_absolute_minimum,500)

# pH_queries = numpy.linspace(7,9,11)
# minimum_d11Bsws = boron_isotopes.calculate_d11BT(pH_queries,Kb,d11B4[d11B_minimum],epsilon)
# maximum_d11Bsws = boron_isotopes.calculate_d11BT(pH_queries,Kb,d11B4[d11B_maximum],epsilon)

# d11B_borate_maximum = []
# for delta_d11Bsw_lin in delta_d11Bsw_linspace:
#     d11B_borate_maximum += [d11Bsw_flat_maximum if d11Bsw_flat_maximum+abs(delta_d11Bsw_lin)<d11Bsw_absolute_maximum else d11Bsw_absolute_maximum-abs(delta_d11Bsw_lin)]

# delta_d11Bsws = []
# negative_delta_d11Bsws = []
# for maximum_d11Bsw in maximum_d11Bsws:
#     delta_d11Bsws += [maximum_d11Bsw-minimum_d11Bsw_linspace]
#     negative_delta_d11Bsws += [-(maximum_d11Bsw-minimum_d11Bsw_linspace)]

# # At what d11Bsw is the pH equal to 9 for both the minimum and maximum d11B4?
# # Already have the minimum d11B4 as that's a flat line
# # So for maximum d11B4 we ask,
# crossover = boron_isotopes.calculate_d11BT(pH_queries,Kb,d11B4[d11B_maximum],epsilon)-minimum_d11Bsws
# #axes_2.plot(crossover,minimum_d11Bsws,color="#a4d9a1")

# #axes_2.plot([0,0],[d11Bsw_flat_minimum,d11Bsw_flat_maximum],color="black",zorder=4)
# axes_2.plot([0,d11Bsw_absolute_maximum-d11Bsw_absolute_minimum],[d11Bsw_flat_minimum,d11Bsw_absolute_minimum],color="black",zorder=4)
# axes_2.plot([0,-(d11Bsw_absolute_maximum-d11Bsw_absolute_minimum)],[d11Bsw_flat_minimum,d11Bsw_absolute_minimum],color="black",zorder=4)
# above_zero = delta_d11Bsw_linspace>=0
# below_zero = delta_d11Bsw_linspace<=0
# axes_2.plot(delta_d11Bsw_linspace[above_zero],numpy.array(d11B_borate_maximum)[above_zero],color="black",zorder=4)
# axes_2.plot(delta_d11Bsw_linspace[below_zero],numpy.array(d11B_borate_maximum)[below_zero],color="black",zorder=4)

# import matplotlib.patches as patches
# patch = patches.Polygon([[-max(delta_d11Bsw_linspace),d11Bsw_absolute_minimum],[0,d11Bsw_flat_minimum],[max(delta_d11Bsw_linspace),d11Bsw_absolute_minimum]],color="white",zorder=3)
# axes_2.add_patch(patch)

# patch2 = patches.Polygon([[-40,d11Bsw_flat_maximum],[40,d11Bsw_flat_maximum],[40,50],[-40,50]],color="white",zorder=3)
# axes_2.add_patch(patch2)

# patch3 = patches.Polygon([[15,32.6],[40,7.9],[40,32.6]],color="white",zorder=3)
# axes_2.add_patch(patch3)

# patch4 = patches.Polygon([[-15,32.6],[-40,7.9],[-40,32.6]],color="white",zorder=3)
# axes_2.add_patch(patch4)

# patch5 = patches.Polygon([[15,14.6],[15,32.5],[40,7.7],[40,6]],color="Grey",zorder=1,alpha=0.25)
# axes_2.add_patch(patch5)

# patch6 = patches.Polygon([[-15,14.6],[-15,32.5],[-40,7.7],[-40,6]],color="Grey",zorder=1,alpha=0.25)
# axes_2.add_patch(patch6)

# for minimum_d11Bsw,pH_query in zip(minimum_d11Bsws,pH_queries):
#     axes_2.plot([-40,40],[minimum_d11Bsw,minimum_d11Bsw],color="#a1d9d4",zorder=2)
# for delta_d11Bsw,negative_delta_d11Bsw,pH_query in zip(delta_d11Bsws,negative_delta_d11Bsws,pH_queries,strict=True):
#     positive_indices = delta_d11Bsw>=0
#     axes_2.plot(delta_d11Bsw[positive_indices],minimum_d11Bsw_linspace[positive_indices],color="#a4d9a1",zorder=2)
#     negative_indices = negative_delta_d11Bsw<=0
#     axes_2.plot(negative_delta_d11Bsw[negative_indices],minimum_d11Bsw_linspace[negative_indices],color="#a4d9a1",zorder=2)
            
# axes_2.text(16.5,31.5,"7.0",rotation=0,color="#a1d9d4",zorder=4)
# axes_2.text(21.5,26.3,"8.0",rotation=0,color="#a1d9d4",zorder=4)
# axes_2.text(35.5,12.5,"9.0",rotation=0,color="#a1d9d4",zorder=4)

# axes_2.text(40,5,"7.0",rotation=-80,color="#a4d9a1",zorder=4)
# axes_2.text(33,6,"8.0",rotation=-80,color="#a4d9a1",zorder=4)
# axes_2.text(12,13,"9.0",rotation=-80,color="#a4d9a1",zorder=4)


# axes_2.text(3,5.5,"pH at maximum $\delta^{11}B_{4}$",color="#a4d9a1")
# axes_2.text(10,35,"pH at minimum $\delta^{11}B_{4}$",color="#a1d9d4")

# axes_2.set_xlabel("$\Delta\delta^{11}B_{sw}$")
# axes_2.set_ylabel("Minimum $\delta^{11}B_{sw}$")

# axes_2.set_xlim((-40,40))
# axes_2.set_ylim((d11Bsw_absolute_minimum,d11Bsw_flat_maximum+5))

# #axes_2.scatter(d11Bsw_scaling_reject,d11Bsw_minimum_reject,marker="x",color="Gray",zorder=0,alpha=0.5)

# figure_3,axes_3 = pyplot.subplots(nrows=1)
# d11Bsw_gp.plotPcolor(axis=axes_3,invert_x=True,colourbar=False,map="Blues",mask=True,vmin=0.01,vmax=0.04)

# axes_3.set_xlim((330,230))
# axes_3.set_ylim((25,45))

# axes_3.set_xlabel("Age (Ma)")
# axes_3.set_ylabel("$\delta^{11}B_{sw}$")

# a = 5
pyplot.savefig("CarboniferousPermian_d11B4_Sr_d11Bsw_pH_DIC_CO2.png",dpi=600)
pyplot.show()