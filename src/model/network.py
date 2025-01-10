from pyomo.environ import *
import sys
import cmath
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
from .utils import create_res_rec_from_dicts, convert_matrix_to_dict

def initialize_network(model,
                    transmission_lines=None,
                    leng=None,
                    bus_from=None,
                    bus_to=None,
                    config=None,
                    Z_base=None,
                    ThermalCap=None,
                    branch_phases=None,
                    HeadBus=['Bus0']):
                    
    Phase = ['A','B','C']
    model.Phases = Set(initialize = Phase)
    model.TransmissionLines = Set(initialize=transmission_lines)
    model.Length = Param(model.TransmissionLines, initialize = leng)
    model.BusFrom = Param(model.TransmissionLines, initialize=bus_from, within=Any)
    model.BusTo = Param(model.TransmissionLines, initialize=bus_to, within=Any)
    model.Config = Param(model.TransmissionLines, initialize=config, within=NonNegativeIntegers)
    model.Z_Base = Param(initialize=Z_base)
    model.HeadBus = Set(initialize=HeadBus)
    model.ThermalCap = Param(model.TransmissionLines, initialize=ThermalCap, within=Any)
    model.BranchPhase = Param(model.TransmissionLines, initialize=branch_phases, within=Any)
    
# Alternative to lines_to
def _derive_connections_to(m, b):
   return (l for l in m.TransmissionLines if m.BusTo[l]==b)

# Alternative to lines_from
def _derive_connections_from(m, b):
   return (l for l in m.TransmissionLines if m.BusFrom[l]==b)

def derive_network(model,
                lines_from=_derive_connections_from,
                lines_to=_derive_connections_to):

   model.LinesTo = Set(model.Buses, initialize=lines_to)
   model.LinesFrom = Set(model.Buses, initialize=lines_from)

def create_incidence_matrix(model):

   # incidence_matrix = np.zeros((len(model.Buses), len(model.TransmissionLines)), dtype=int)
   incidence_matrix = np.zeros((len(model.TransmissionLines), len(model.Buses)), dtype=int)
   unique_buses = model.Buses
   bus_index = {bus: idx for idx, bus in enumerate(unique_buses)}
  
   for l in range(1, len(model.TransmissionLines)+1):
      incidence_matrix[l-1, bus_index[model.BusFrom[l]]] = 1
      incidence_matrix[l-1, bus_index[model.BusTo[l]]] = -1
   
   A_0 = incidence_matrix[:, bus_index[model.HeadBus.at(1)],].reshape(-1,1)
   A   = np.delete(incidence_matrix, bus_index[model.HeadBus.at(1)], axis=1)
   
   A_tilde = np.kron(incidence_matrix, np.eye(3, dtype=int)) 
   A_0 = np.kron(A_0, np.eye(3, dtype=int))
   A = np.kron(A, np.eye(3, dtype=int))
   
   # print(A_tilde.shape, A_0.shape, A.shape)
   return A_tilde, A_0, A
   
def line_resistance_init(model, line, r):
   return r[model.Config[line]] * model.Length[line] /model.Z_Base 

def line_reactance_init(model, line, x):
   return x[model.Config[line]] * model.Length[line] /model.Z_Base 

def calculate_network_parameters(model, config=None):
   
   model.c = Set(initialize = np.arange(1,13))
   model.ImpedanceConfig = Param(model.c, model.Phases, model.Phases, initialize=config[0])
   
   r, x = create_res_rec_from_dicts(config[1])
 
   Res = {}
   Rec = {}
 
   for l in model.TransmissionLines:
      Res[l] = line_resistance_init(model, l, r)
      Rec[l] = line_reactance_init(model, l, x)
   

   Diag_matrix_res = np.zeros((3*len(model.TransmissionLines), 3*len(model.TransmissionLines)))
   Diag_matrix_rec = np.zeros((3*len(model.TransmissionLines), 3*len(model.TransmissionLines)))
   
   i,j = 0,0

   for l in model.TransmissionLines:
      Diag_matrix_res[i:i+3, j:j+3] = Res[l]
      Diag_matrix_rec[i:i+3, j:j+3] = Rec[l]
      i = i+3
      j = j+3

   return Diag_matrix_res, Diag_matrix_rec

