from pyomo.environ import SolverFactory
import warnings
import os
import click
from .results import PSSTResults
from pyomo.common.tempfiles import TempfileManager

PSST_WARNING = os.getenv('PSST_WARNING', 'ignore')


def solve_model(model, solver='glpk', solver_io=None, keepfiles=True, verbose=True, symbolic_solver_labels=True, is_mip=True, mipgap=1e-3):
    if solver == 'xpress':
        solver = SolverFactory(solver, solver_io=solver_io, is_mip=is_mip)
    else:
        solver = SolverFactory(solver, solver_io=solver_io)
    
    if is_mip:
        solver.options['mipgap'] = mipgap

    with warnings.catch_warnings():
        
        warnings.simplefilter(PSST_WARNING)
        TempfileManager.tempdir = os.path.join(os.getcwd(),"../PyomoTempFiles") 
        
        resultsPSST = solver.solve(model, suffixes=['dual'], tee=verbose, keepfiles=True, symbolic_solver_labels=symbolic_solver_labels)
        
        TC = str(resultsPSST.solver.termination_condition)
        click.echo("Termination Condition : ")
        click.echo(TC)
    

    return model, TC
