from pyomo.environ import *
import click

def initialize_DERs(model,
                        DER_names=None,
                        DER_at_bus=None):

    model.DERs = Set(initialize=DER_names)
    model.DERsAtBus = Set(model.Buses, initialize=DER_at_bus)
    
def maximum_minimum_activepower_output_DERs(model, minimum_power_output=None, maximum_power_output=None):

    model.MinimumActivePowerOutput = Param(model.DERs, initialize=minimum_power_output, within=NonNegativeReals, default=0.0)
    model.MaximumActivePowerOutput = Param(model.DERs, initialize=maximum_power_output, within=NonNegativeReals, default=0.0)

def maximum_minimum_reactivepower_output_DERs(model, minimum_power_output=None, maximum_power_output=None):

    model.MinimumReactivePowerOutput = Param(model.DERs, initialize=minimum_power_output, within=Reals, default=0.0)
    model.MaximumReactivePowerOutput = Param(model.DERs, initialize=maximum_power_output, within=Reals, default=0.0)
