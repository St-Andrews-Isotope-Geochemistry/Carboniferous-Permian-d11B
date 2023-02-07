import pandas,cbsyst,numpy
from matplotlib import pyplot

calibration_data = pandas.read_excel("./Data/Input/Species_Calibration.xlsx",sheet_name="Matlab",header=1,usecols="G,I,J,L,M,N,O,Q",names=["depth","d11B","d11B_uncertainty","pH","pH_uncertainty","temperature","salinity","d11B4"])

calibration_data["d11B_uncertainty"] = calibration_data["d11B_uncertainty"].fillna(calibration_data["d11B_uncertainty"].max())
calibration_data["pH_uncertainty"] = calibration_data["pH_uncertainty"].fillna(calibration_data["pH_uncertainty"].max())

bsys = cbsyst.Bsys(pHtot=calibration_data["pH"],BT=400,dBT=39.61,T_in=calibration_data["temperature"],S_in=calibration_data["salinity"],P_in=calibration_data["depth"]/10)

calibration_used = lambda measured: (measured-5.0936)/0.7735
calibration_estimated_coefficients = numpy.polynomial.polynomial.polyfit(bsys.dBO4,calibration_data["d11B"],1,full=True)[0]
calibration_estimated = lambda measured: (measured-calibration_estimated_coefficients[0])/calibration_estimated_coefficients[1]

d11B4 = bsys.dBO4

figure,axes = pyplot.subplots(nrows=1)
axes = [axes]

axes[0].scatter(calibration_data["d11B"],calibration_data["d11B4"],color="#32982d",edgecolor="black")
axes[0].scatter(calibration_data["d11B"],bsys.dBO4,color="#d65656",edgecolor="black")
axes[0].plot([12,22],[12,22],color="black",linestyle="dashed")
axes[0].plot([12,22],calibration_used(numpy.array([12,22])),color="#32982d")
axes[0].plot([12,22],calibration_estimated(numpy.array([12,22])),color="#d65656")

axes[0].set_xlabel("Measured $\delta^{11}B$ (‰)")
axes[0].set_ylabel("Calculated $\delta^{11}B_{4}$ (‰)")

axes[0].set_xlim((12,22))
axes[0].set_ylim((12,22))
axes[0].set_box_aspect(1)


a = 5