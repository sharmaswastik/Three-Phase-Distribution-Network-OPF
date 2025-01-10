import numpy as np
from pyomo.environ import *
import logging
from functools import partial
from . import global_params
import itertools

logger = logging.getLogger(__file__)
eps = 1e-3


def fix_first_voltage_rule(m, p ,t, V_Slack, slack_bus=1):
    return m.V[p, m.HeadBus.at(slack_bus),  t] == V_Slack

def source_bus_active_power_rule(m, t, b, p):
    if b == m.HeadBus.at(1):
        return m.ActivePowerAtSourceBus[p,b,t] == m.P_L[p, m.TransmissionLines.at(1), t]
    else:
       return Constraint.Skip

def source_bus_reactive_power_rule(m, t, b, p):
    if b == m.HeadBus.at(1):    
        return m.ReactivePowerAtSourceBus[p,b,t] == m.Q_L[p, m.TransmissionLines.at(1), t]
    else:
       return Constraint.Skip

def active_power_source_bus_constraint(model):
    model.ActivePowerSourceBusConstraint = Constraint(model.TimePeriods, model.Buses, model.Phases, rule=source_bus_active_power_rule)
def reactive_power_source_bus_constraint(model):
    model.ReactivePowerSourceBusConstraint = Constraint(model.TimePeriods, model.Buses, model.Phases, rule=source_bus_reactive_power_rule)

def constraint_slack_bus(model, V_Slack, slack_bus=1):
    partial_fix_first_voltage_rule = partial(fix_first_voltage_rule, V_Slack=V_Slack, slack_bus=slack_bus)
    model.FixFirstVoltage = Constraint(model.Phases, model.TimePeriods, rule=partial_fix_first_voltage_rule)
    
def voltage_constraint(model, A_dash, R_d, X_d):
    
    def voltage_constraint_rule(m, t, b, p):
    
        if b == m.HeadBus.at(1):
            return Constraint.Skip
            
        else:
            Constraint_1 = (m.V[p, b, t] == sum(-A_dash[global_params.count][j-1] * m.V[f, m.HeadBus.at(1), t] for j, f in zip(m.Dim_a, m.Phases)) + 2 * sum(R_d[global_params.count][k-1]\
            * m.P_L[f, m.TransmissionLines.at(int((k-1)/(len(m.Phases))+1)), t]  for k, f in zip(m.Dim_c, itertools.cycle(m.Phases))) + 2 * sum(X_d[global_params.count][k-1] * \
            m.Q_L[f, m.TransmissionLines.at(int((k-1)/(len(m.Phases))+1)), t] for k, f in zip(m.Dim_c, itertools.cycle(m.Phases))))
            
            global_params.count += 1
            
            if global_params.count == len(m.Dim_r):
                global_params.initialize_param()
        
            return Constraint_1

    model.VoltageConstraint = Constraint(model.TimePeriods, model.Buses, model.Phases, rule=voltage_constraint_rule)

def activepower_balance_constraint(model,A, A_0):

    def activepower_balance_constraint_rule(m,t,b,p):

        if b == m.HeadBus.at(1):
            Constraint = sum(A_0[global_params.count][k-1] * m.P_L[f,m.TransmissionLines.at(int((k-1)/(len(m.Phases))+1)),t] for k, f in zip(m.Dim_c, itertools.cycle(m.Phases))) == m.P[p,b,t] + m.ActivePowerAtSourceBus[p,b,t]
            global_params.count+=1 
            if global_params.count == 3:
                global_params.initialize_param()   
            
            return Constraint 

        else:
            constraint_1 = sum(A[global_params.count][k-1] * m.P_L[f,m.TransmissionLines.at(int((k-1)/(len(m.Phases))+1)),t] for k, f in zip(m.Dim_c, itertools.cycle(m.Phases))) == m.P[p,b,t]
            global_params.count+=1
                
            if global_params.count == len(m.Dim_r):
                global_params.initialize_param()
                
            return constraint_1

    model.ActivePowerBalanceConstraint = Constraint(model.TimePeriods, model.Buses, model.Phases, rule=activepower_balance_constraint_rule)
    
    

def reactivepower_balance_constraint(model, A, A_0):

    def reactivepower_balance_constraint_rule(m,t,b,p):
        
        if b == m.HeadBus.at(1):
            Constraint = sum(A_0[global_params.count][k-1] * m.Q_L[f,m.TransmissionLines.at(int((k-1)/(len(m.Phases))+1)),t] for k, f in zip(m.Dim_c, itertools.cycle(m.Phases))) == m.Q[p,b,t] + m.ReactivePowerAtSourceBus[p,b,t]
            global_params.count+=1
                    
            if global_params.count == 3:
                global_params.initialize_param()   
            return Constraint 

        else:
            constraint_1 = sum(A[global_params.count][k-1] * m.Q_L[f,m.TransmissionLines.at(int((k-1)/(len(m.Phases))+1)),t] for k, f in zip(m.Dim_c, itertools.cycle(m.Phases))) == m.Q[p,b,t]
            global_params.count+=1
                    
            if global_params.count == len(m.Dim_r):
                global_params.initialize_param()
                    
            return constraint_1

    model.ReactivePowerBalanceConstraint = Constraint(model.TimePeriods, model.Buses, model.Phases, rule=reactivepower_balance_constraint_rule)
    

def net_activepower_at_bus_rule(m, t, b, p):
    
    if p == 'A':
        constraint = - (m.SpotLoadP_A[b])/(m.S_Base/3)
    if p == 'B':
        constraint = - (m.SpotLoadP_B[b])/(m.S_Base/3)
    if p == 'C':
        constraint = - (m.SpotLoadP_C[b])/(m.S_Base/3)
    if b in m.DERsAtBus:
        constraint = constraint + sum(m.ActivePowerGenerated[p, g, t] for g in m.DERsAtBus[b])

    constraint = m.P[p, b, t] == constraint
    
    return constraint

def constraint_net_activepower(model):
    model.CalculateNetActivePowerAtBus = Constraint(model.TimePeriods, model.Buses, model.Phases, rule=net_activepower_at_bus_rule)
    
def net_reactivepower_at_bus_rule(m, t, b, p):
    
    if p == 'A':
        constraint = - (m.SpotLoadQ_A[b])/(m.S_Base/3) 
    if p == 'B':
        constraint = - (m.SpotLoadQ_B[b])/(m.S_Base/3)
    if p == 'C':
        constraint = - (m.SpotLoadQ_C[b])/(m.S_Base/3)

    if b in m.DERsAtBus:
        constraint = constraint + sum(m.ReactivePowerGenerated[p, g, t] for g in m.DERsAtBus[b])

    constraint = m.Q[p, b, t] == constraint
    
    return constraint

def constraint_net_reactivepower(model):
    model.CalculateNetReactivePowerAtBus = Constraint(model.TimePeriods, model.Buses, model.Phases, rule=net_reactivepower_at_bus_rule)

def thermal_limit_rule(m, p, l, e, t):

    constraint = m.Alpha_c[e] * m.P_L[p, l, t] + m.Beta_c[e] * m.Q_L[p,l,t] + (m.Delta_c[e] * m.ThermalCap[l]/m.I_Base)
    
    return constraint <= 0

def thermal_limit_constraint(model):

    model.ThermalLimitConstraint = Constraint(model.Phases, model.TransmissionLines, model.Edges, model.TimePeriods, rule=thermal_limit_rule)

def substation_power_limit_rule(m, p, l, e, t):
    if l == 1:
        constraint = m.Alpha_c[e] * m.P_L[p, l, t] + m.Beta_c[e] * m.Q_L[p,l,t] + (m.Delta_c[e] * m.SubSMax/(m.S_Base/3))
        return constraint <= 0
    
    else:
        return Constraint.Skip

def substation_power_limit_constraint(model):

    model.SubstationPowerLimitConstraint = Constraint(model.Phases, model.TransmissionLines, model.Edges, model.TimePeriods, rule=substation_power_limit_rule)

def model_objective_function(m):
    
    return sum(m.ActivePowerGenerated[p,g,t] for p in m.Phases for g in m.DERs for t in m.TimePeriods)

def objective_function(model):
    model.GenerationObjective = Objective(rule=model_objective_function, sense=minimize)



