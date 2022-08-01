import pandas as pd
import numpy as np
import gurobipy as gp
from .FairnessModule import FairnessModule

class EqualityOfOpportunity(FairnessModule):
    
    def __init__(self, args = {}):
        super().__init__()
        
        if 'group' not in args:
            raise Exception('No group assignments!')
        
        self.group = args['group']
        self.fairConstNames = ['ubFair','lbFair']
        self.eps = args['epsilon'] if 'epsilon' in args else 0.05
        
        return
        
    def defineObjective(self, delta, z, Y, args):
        '''
        Returns gurobi objective for pricing problem
        '''
        if 'lam' not in args or 'mu' not in args or 'fairDuals' not in args or 'row_samples' not in args:
            raise Exception('Required arguments not supplied for NoFair Objective Definition.')
        
        #Retrieve group membership for samples
        g = self.group[args['row_samples']]
        
        #Construct Objective
        objective = gp.LinExpr(np.ones(sum(~Y)), np.array(delta)[~Y]) #Y = False misclass term
        objective.add(gp.LinExpr(np.array(args['coeff']), np.array(delta)[Y])) #Y = True misclass term
        objective.add(gp.LinExpr(args['lam']*np.ones(len(z)), z)) #Complexity term
        
        return objective
  
    
    def computeObjective(self, X, Y, features, args):
        '''
        Returns reduced cost for pricing problem for given inputs
        '''
        if 'lam' not in args or 'mu' not in args or 'fairDuals' not in args or 'row_samples' not in args:
            raise Exception('Required arguments not supplied for NoFair Objective Computation.')
        
        classPos = np.all(X[:,features],axis=1)
        g = self.group[args['row_samples']]

        return args['lam']*(1+len(features)) + np.dot(classPos[Y],np.array(args['coeff'])) + sum(classPos[~Y])

    

    def createFairnessConstraint(self, model, x, Y):
        '''
        Returns constraint for fairness
        '''
        constraints = []
        x = np.array(x)
        g = self.group[Y]

        constraints.append(model.addConstr( 1/sum(g)*sum(x[g]) - \
                                                1/sum(~g)*sum(x[~g])<= self.eps, 
                                                name = 'ubFair'))
        constraints.append(model.addConstr( -1/sum(g)*sum(x[g]) + \
                                                1/sum(~g)*sum(x[~g])<= self.eps, 
                                                name = 'lbFair'))

        return constraints
