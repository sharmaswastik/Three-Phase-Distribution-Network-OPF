#This PSST file, originally due to Dheepak Krishnamurthy, has been modified by Swathi Battula to return the sorted list of LMPs.

import pandas as pd
import click
import re

class PSSTResults(object):

    def __init__(self, model):

        self._model = model
        self._maximum_hours = 24

    
    @property
    def voltage_dual(self):
        return self._get('VoltageConstraint', self._model, dual=True)
    
    @property
    def activepowerbalance_dual(self):
        return self._get('ActivePowerBalanceConstraint', self._model, dual=True)

    @property
    def reactivepowerbalance_dual(self):
        return self._get('ReactivePowerBalanceConstraint', self._model, dual=True)

    @property
    def thermallimit_dual(self):
        return self._get_edge_duals('ThermalLimitConstraint', self._model, dual=True)

    @property
    def substationlimit_dual(self):
        return self._get_edge_duals('SubstationPowerLimitConstraint', self._model, dual=True)

    @staticmethod
    def _get_edge_duals(attribute, model, dual=False):
        index = getattr(model, attribute).keys()
        lines = set()
        phases = set()
        duals = {}
        line_violation_edge = {}
        for idx in index:
            if isinstance(idx, tuple):
                parts = idx
        
            if len(parts) >= 3:
                phase = parts[0]
                line = parts[1]
                edge = parts[2]
                
                lines.add(line)
                phases.add(phase)
                
                constraint = getattr(model, attribute)[idx]
                val = model.dual.get(constraint, None)

                if val != None and abs(val) > 1e-6:
                    duals[(line, phase)] = val#/(model.S_Base/3)
                   
                    if line not in line_violation_edge:
                        line_violation_edge[line] = edge

        data = []
        for line in sorted(lines):
            violating_edge = line_violation_edge.get(line)
            
            if violating_edge == None:
                violating_edge = '1'  
            for phase in sorted(phases):
                key = (line, phase)
                dual = duals.get(key, 0.0)  
                data.append({'Line': line, 'Phase': phase, 'Dual': dual, 'Edge': violating_edge})
        
        df = pd.DataFrame(data)
        df.set_index(['Line', 'Phase'], inplace=True)
        df.sort_index(inplace=True)
        return df
    
    @staticmethod
    def _get(attribute, model, dual=False):
        index = getattr(model, attribute).keys()
        Buses = set()
        phases = set()
        duals = {}
        data = []

        for idx in index:
            if isinstance(idx, tuple):
                parts = idx
            
            if len(parts) >= 3:
                bus = parts[1]
                phase = parts[2]

                if bus != 'Bus0':
                    Buses.add(bus)
                phases.add(phase)

                constraint = getattr(model, attribute)[idx]
                val = model.dual.get(constraint, None)

                if val != None and bus != 'Bus0':
                    duals[(bus, phase)] = val#/(model.S_Base/3)

        for bus in sorted(Buses, key=lambda x: int(x[3:])):
            for phase in sorted(phases):
                key = (bus,phase)
                dual = duals.get(key, None)
                data.append({'Bus':bus, 'Phase':phase, 'Dual':dual})

        df = pd.DataFrame(data)
        df.set_index(['Bus', 'Phase'], inplace=True)
        return df

    @staticmethod
    def _get_Bus0(attribute, model, dual=False):
        index = getattr(model, attribute).keys()
        Buses = set()
        phases = set()
        duals = {}
        data = []

        for idx in index:
            if isinstance(idx, tuple):
                parts = idx
            
            if len(parts) >= 3:
                bus = parts[1]
                phase = parts[2]
                if bus == 'Bus0':
                    Buses.add(bus)
                    phases.add(phase)
                    constraint = getattr(model, attribute)[idx]
                    val = model.dual.get(constraint, None)

                    if val != None:
                        duals[(bus, phase)] = val#/(model.S_Base/3)

        for bus in sorted(Buses):
            for phase in sorted(phases):
                key = (bus,phase)
                dual = duals.get(key, None)
                data.append({'Bus':bus, 'Phase':phase, 'Dual':dual})

        df = pd.DataFrame(data)
        df.set_index(['Bus', 'Phase'], inplace=True)
        df.sort_index(inplace=True)

        return df


   