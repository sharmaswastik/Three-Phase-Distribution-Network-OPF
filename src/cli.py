import os
import sys
import click
import pandas as pd
import random
import numpy as np
from pathlib import Path
from glob import glob
import datetime
import pytz
from tqdm import tqdm
import re
import threading
import pickle
import time
from model import build_model
from model.utils import (create_matrix_dict_pandas, correct_names, data_indexing, phases_from_config, descending_sort_dict, get_duals_in_numpy_vector)

np.seterr(all='raise')
1
SOLVER = os.getenv('PSST_SOLVER')


def OPF(solver, data=None):
	t = datetime.datetime.now(pytz.timezone('Asia/Kolkata')).strftime("%Y-%m-%d-%H-%M-%S")
	output = os.path.join(os.getcwd(), "../OutputFiles/Results_{}.dat").format(t)
	#Doing this helps in creating results in a sequential manner for each run at a time.

	branch_df = pd.read_csv("OPF_data/branch.csv")
	DER_df = pd.read_csv("OPF_data/der.csv")
	bus_df = pd.read_csv("OPF_data/bus.csv")
	capacitor_df = pd.read_csv("OPF_data/CapData.csv")
	substation_df = pd.read_csv("OPF_data/Substation.csv")
	SpotLoadsP = data_indexing("OPF_data/SpotLoadsP.csv", 'Node', len(bus_df), 'Bus')
	SpotLoadsQ = data_indexing("OPF_data/SpotLoadsQ.csv", 'Node', len(bus_df), 'Bus')
	
	if os.path.isfile("OPF_data/thermal_constraint_data.xlsx"):	
		Inp_thermal = input("\nDo you want to use Thermal Limit Constraints, '1':'Yes', 'Any Other Number':'No\n")
		if int(Inp_thermal) == 1:
			print("\nEmploying Thermal limit Constraints")
			thermal_const_df = pd.read_excel("OPF_data/thermal_constraint_data.xlsx")
		else:
			print("\nIgnoring Thermal limit Constraints")
	
	# if os.path.isfile("OPF_data/thermal_constraint_data.xlsx"): 
	# 	thermal_const_df = pd.read_excel("OPF_data/thermal_constraint_data.xlsx")
	# 	print("\n\nThermal constraints related data found.")
	# 	Inp_thermal = '1'

	# else:
	# 	thermal_const_df = None
	# 	Inp_thermal = '0'
	# 	print("Thermal constraint related data file not found, the problem will ignore thermal constraints.")

	branch_df = correct_names(branch_df, 'F_BUS', 'Bus')
	branch_df = correct_names(branch_df, 'T_BUS', 'Bus')
	branch_df = phases_from_config(branch_df, need_phases=True)
	bus_df = correct_names(bus_df, 'bus_i', 'Bus')
	bus_df = phases_from_config(bus_df, need_phases=False)
	DER_df = correct_names(DER_df, 'DER_i', 'DER')
	DER_df = correct_names(DER_df, 'DER_BUS', 'Bus')
	capacitor_df = correct_names(capacitor_df, 'Cap_i', 'Cap')
	capacitor_df = correct_names(capacitor_df, 'Node', 'Bus')
	substation_df = correct_names(substation_df, 'HeadBus', 'Bus')

	cwd = os.getcwd()
	path = r"OPF_data\load" 
	path = os.path.join(cwd, path)
	Pload_df = []
	Qload_df = []

	csv_files = glob(os.path.join(path, "*.csv"))
	global model
	global SolverOutcomes
	for f in csv_files:
		df = pd.read_csv(f)
		p_df = os.path.basename(f)
		p = p_df.split('-')[-1].split('.')[0]
		p = p.capitalize()
		df['Phase'] = p
		if p_df[0] == 'P':
			Pload_df.append(df)
		else:
			Qload_df.append(df)
	Pload_df = pd.concat(Pload_df, ignore_index=False)
	Qload_df = pd.concat(Qload_df, ignore_index=False)


	# CapData    = data_indexing("OPF_data/CapData.csv", 'Node', len(bus_df), 'Bus')
	# HeadBus    = bus_df.loc[bus_df['HeadBus'] == 1,'bus_i'].values[0]
	HeadBus = substation_df['HeadBus'].values[0]
	config_dict, config_numpy = create_matrix_dict_pandas("OPF_data/config.csv")
	
	def display_loading_bar():
		while not done:
			print("Working on problem...", end="\r")
			time.sleep(0.5)
	
	done = False
	loading_thread = threading.Thread(target=display_loading_bar)
	loading_thread.start()

	try:
		model = build_model(substation_df = substation_df, branch_df=branch_df, DER_df=DER_df, bus_df=bus_df, Pload_df=Pload_df, Qload_df=Qload_df, config = (config_dict, config_numpy), SpotLoads = (SpotLoadsP,SpotLoadsQ), CapData=capacitor_df, thermal_df=thermal_const_df, Inp_thermal=Inp_thermal)

		SolverOutcomes = model.solve(solver=solver)
		Status= str(SolverOutcomes[1])

	finally:
		done=True
		loading_thread.join()
		datetime_india = datetime.datetime.now(pytz.timezone('Asia/Kolkata'))

		if (Status == 'optimal'):
			model.sort_buses()
			instance = model._model

			with open(output.strip("'"), 'w') as f:

				f.write("THE OPF WAS RUN AT : ") 
				f.write(datetime_india.strftime('%Y:%m:%d %H:%M:%S %Z %z'))
				f.write("\n\nSOLUTION_STATUS\n")
				f.write("optimal \t")
				f.write("\nEND_SOLUTION_STATUS\n\n")
				
				f.write("DGResultsforActivePower\n\n")
				
				for g in instance.DERs.data():
					f.write("%s\n" % str(g).ljust(8))
					for t in instance.TimePeriods:
						f.write("Interval: {}\n".format(str(t)))
						for p in instance.Phases:
							if instance.ActivePowerGenerated[p, g, t].value != None:
								f.write("\tActivePowerGenerated: {} kW at Phase {}\n".format(round(instance.ActivePowerGenerated[p, g, t].value * instance.S_Base.value/3,5), p))
							else:
								f.write("\tActivePowerGenerated: {} kW at Phase {}\n".format(0, p))
				f.write("\nEND_DGResultsforActivePower\n\n")

				f.write("DGResultsforReactivePower\n\n")
				for g in instance.DERs.data():
					f.write("%s\n" % str(g).ljust(8))
					for t in instance.TimePeriods:
						f.write("Interval: {}\n".format(str(t)))
						for p in instance.Phases:
							if instance.ReactivePowerGenerated[p, g, t].value != None:
								f.write("\tReactivePowerGenerated: {} kVaR at Phase {}\n".format(round(instance.ReactivePowerGenerated[p, g, t].value * instance.S_Base.value/3,5), p))
							else:
								f.write("\tReactivePowerGenerated: {} kVaR at Phase {}\n".format(0, p))
				f.write("\nEND_DGResultsforReactivePower\n\n")

				f.write("VOLTAGE MAGNITUDES\n\n")
				for bus in instance.Buses:
					for t in instance.TimePeriods:
						for p in instance.Phases: 
							if p in instance.BusPhase[bus]:
								f.write('Phase: {} Bus: {} Interval: {} : {} p.u.\n'.format(str(p), str(bus), str(t), str(round(np.sqrt(instance.V[p, bus, t].value), 5))))
							else:
								f.write('Phase: {} Bus: {} Interval: {} : {} \n'.format(str(p), str(bus), str(t), 'NaN', 5))
				f.write("\nEND VOLTAGE MAGNITUDES\n\n")

				f.write("Active Power at Each Node [Excludes Source Bus Injection]\n\n")
				for bus in instance.Buses:
					for t in instance.TimePeriods:
						for p in instance.Phases:
							f.write('Phase: {} Bus: {} Interval: {} : {} kW\n'.format(str(p), str(bus), str(t), str(round(instance.P[p, bus, t].value * instance.S_Base.value/3, 5))))
				f.write("\nEND_ACTIVE_POWER_AT_EACH_NODE\n\n")

				f.write("LINE_ACTIVE_POWER_FLOWS\n\n")
				for l in sorted(instance.TransmissionLines):
					for t in instance.TimePeriods:
						for p in instance.Phases:
							f.write('Phase: {} Line Connecting: {} to {} Interval: {} : {} kW\n'.format(str(p), str(instance.BusFrom[l]), str(instance.BusTo[l]), str(t), str(round(instance.P_L[p, l, t].value  * instance.S_Base.value/3, 5))))
				f.write("\nEND_LINE_ACTIVE_POWER_FLOWS\n\n")

				f.write("Reactive Power at Each Node [Excludes Source Bus Injection]\n\n")
				for bus in instance.Buses:
					for t in instance.TimePeriods:
						for p in instance.Phases:
							f.write('Phase: {} Bus: {} Interval: {} : {} kVaR\n'.format(str(p), str(bus), str(t), str(round(instance.Q[p, bus, t].value * instance.S_Base.value/3, 5))))
				f.write("\nEND_REACTIVE_POWER_AT_EACH_NODE\n\n")

				f.write("LINE_REACTIVE_POWER_FLOWS\n\n")
				for l in sorted(instance.TransmissionLines):
					for t in instance.TimePeriods:
						for p in instance.Phases:
							f.write('Phase: {} Line Connecting: {} to {} Interval: {} : {} kVaR\n'.format(str(p), str(instance.BusFrom[l]), str(instance.BusTo[l]), str(t), str(round(instance.Q_L[p, l, t].value * instance.S_Base.value/3, 5))))
				f.write("\nEND_LINE_REACTIVE_POWER_FLOWS\n\n")

				f.write("ACTIVE_POWER_AT_SOURCE_BUS\n\n")
				for bus in instance.Buses:
					for t in instance.TimePeriods:
						for p in instance.Phases:
							if bus == HeadBus:
								if instance.ActivePowerAtSourceBus[p,bus, t].value != None:
									f.write('Phase: {} Bus: {} Interval: {} : {} kW\n'.format(str(p), str(bus), str(t), str(round(instance.ActivePowerAtSourceBus[p, bus, t].value * instance.S_Base.value/3, 5))))
								else:
									f.write('Phase: {} Bus: {} Interval: {} : {} kW\n'.format(str(p), str(bus), str(t), str(0)))
				f.write("\nEND_ACTIVE_POWER_AT_SOURCE_BUS\n\n")

				f.write("REACTIVE_POWER_AT_SOURCE_BUS\n\n")
				for bus in instance.Buses:
					for t in instance.TimePeriods:
						for p in instance.Phases:
							if bus == HeadBus:
								if instance.ReactivePowerAtSourceBus[p,bus, t].value != None:
									f.write('Phase: {} Bus: {} Interval: {} : {} kVaR\n'.format(str(p), str(bus), str(t), str(round(instance.ReactivePowerAtSourceBus[p, bus, t].value * instance.S_Base.value/3, 5))))
								else:
									f.write('Phase: {} Bus: {} Interval: {} : {} kVaR\n'.format(str(p), str(bus), str(t), str(0)))
				f.write("\nEND_REACTIVE_POWER_AT_SOURCE_BUS\n\n")

		elif (Status == 'infeasible'):
			with open(output.strip("'"), 'w') as f:
				f.write("THE OPF WAS RUN AT : ") 
				f.write(datetime_india.strftime('%Y:%m:%d %H:%M:%S %Z %z'))
				f.write("\nSOLUTION_STATUS\n")
				f.write("infeasible \t")
				f.write("\nEND_SOLUTION_STATUS\n")

	return

if __name__ == "__main__":

	OPF(SOLVER)
	