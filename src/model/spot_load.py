from pyomo.environ import *
from .utils import filter_by_phase_to_dict

def initialize_spotload(model, SpotLoads=None):
    
    
    model.SpotLoadP_A = Param(model.Buses, initialize=SpotLoads[0]['A'].to_dict(), default=0.0, mutable=True, within=NonNegativeReals)
    model.SpotLoadP_B = Param(model.Buses, initialize=SpotLoads[0]['B'].to_dict(), default=0.0, mutable=True, within=NonNegativeReals)
    model.SpotLoadP_C = Param(model.Buses, initialize=SpotLoads[0]['C'].to_dict(), default=0.0, mutable=True, within=NonNegativeReals)
    
    model.SpotLoadQ_A = Param(model.Buses, initialize=SpotLoads[1]['A'].to_dict(), default=0.0, mutable=True, within=Reals)
    model.SpotLoadQ_B = Param(model.Buses, initialize=SpotLoads[1]['B'].to_dict(), default=0.0, mutable=True, within=Reals)
    model.SpotLoadQ_C = Param(model.Buses, initialize=SpotLoads[1]['C'].to_dict(), default=0.0, mutable=True, within=Reals)
