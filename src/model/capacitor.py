from pyomo.environ import *

def initialize_cap(model, 
                  cap_names =None,
                  cap_at_bus =None):
    model.Capacitors = Set(initialize = cap_names)
    model.CapacitorsAtBus = Set(model.Buses, initialize = cap_at_bus)

def maximum_output_capacitors(model,
                        maximum_output = None,
                        Switching = None):
    model.CapacitorsMaximumOutput = Param(model.Capacitors, model.Phases, initialize=maximum_output, within=NonNegativeReals, default=0.0)
    
def switching_cap_func(model,
                        Switching = None,
                        SwitchingCost=None):
    model.CapacitorSwitching = Param(model.Capacitors, initialize=Switching, within=NonNegativeReals)
    model.CapacitorSwitchingCost = Param(model.Capacitors, initialize=SwitchingCost, within=NonNegativeReals)
 