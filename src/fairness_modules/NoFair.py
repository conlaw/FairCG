import pandas as pd
import numpy as np
import gurobipy as gp
from .FairnessModule import FairnessModule

class NoFair(FairnessModule):
    
    def __init__(self, args = {}):
        super().__init__()
        return
        
    def defineObjective(self, delta, z, Y, args):
        '''
        Returns gurobi objective for pricing problem
        '''
        if 'lam' not in args or 'mu' not in args:
            raise Exception('Required arguments not supplied for NoFair Objective Definition.')
        objective = gp.LinExpr(np.ones(sum(~Y)), np.array(delta)[~Y]) #Y = False misclass term
        objective.add(gp.LinExpr(np.array(args['coeff']), np.array(delta)[Y])) #Y = True misclass term
        objective.add(gp.LinExpr(args['lam']*np.ones(len(z)), z)) #Complexity term
        return objective
  
    
    def computeObjective(self, X, Y, features, args):
        '''
        Returns reduced cost for pricing problem for given inputs
        '''
        if 'lam' not in args or 'mu' not in args:
            raise Exception('Required arguments not supplied for NoFair Objective Computation.')

        classPos = np.all(X[:,features],axis=1)
        return args['lam']*(1+len(features)) + np.dot(classPos[Y],np.array(args['coeff'])) \
                                                      + sum(classPos[~Y])

    
    def extractDualVariables(self):
        '''
        Returns dict with dual variables related to fairness constraint
        '''
        return

    def createFairnessConstraint(self, model, x, Y):
        '''
        Returns constraint for fairness
        '''
        return
