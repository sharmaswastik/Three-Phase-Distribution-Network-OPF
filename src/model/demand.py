from pyomo.environ import *
from .utils import filter_by_phase_to_dict

def initialize_load(model, NetLoadP=None, NetLoadQ=None):
    

    NetLoadP_A = filter_by_phase_to_dict(NetLoadP, 'A')
    NetLoadP_B = filter_by_phase_to_dict(NetLoadP, 'B')
    NetLoadP_C = filter_by_phase_to_dict(NetLoadP, 'C')
    
    NetLoadQ_A = filter_by_phase_to_dict(NetLoadQ, 'A')
    NetLoadQ_B = filter_by_phase_to_dict(NetLoadQ, 'B')
    NetLoadQ_C = filter_by_phase_to_dict(NetLoadQ, 'C')

    model.NetLoadP_A = Param(model.Buses, model.TimePeriods, initialize=NetLoadP_A, default=0.0, mutable=True, within=NonNegativeReals)
    model.NetLoadP_B = Param(model.Buses, model.TimePeriods, initialize=NetLoadP_B, default=0.0, mutable=True, within=NonNegativeReals)
    model.NetLoadP_C = Param(model.Buses, model.TimePeriods, initialize=NetLoadP_C, default=0.0, mutable=True, within=NonNegativeReals)
    
    model.NetLoadQ_A = Param(model.Buses, model.TimePeriods, initialize=NetLoadQ_A, default=0.0, mutable=True, within=NonNegativeReals)
    model.NetLoadQ_B = Param(model.Buses, model.TimePeriods, initialize=NetLoadQ_B, default=0.0, mutable=True, within=NonNegativeReals)
    model.NetLoadQ_C = Param(model.Buses, model.TimePeriods, initialize=NetLoadQ_C, default=0.0, mutable=True, within=NonNegativeReals)
