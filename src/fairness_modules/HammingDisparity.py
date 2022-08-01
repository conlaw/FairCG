import pandas as pd
import numpy as np
import gurobipy as gp
from .FairnessModule import FairnessModule

class HammingDisparity(FairnessModule):
    
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
        
        if 'ubFair' not in args['fairDuals'] or 'lbFair' not in args['fairDuals']:
            raise Exception('Required fairness dual variables not supplied for NoFair Objective Definition.')
        
        g = self.group[args['row_samples']]
        coeff_1 = 1+ (args['fairDuals']['ubFair'] - args['fairDuals']['lbFair'])/sum(g)
        coeff_2 = 1+ (args['fairDuals']['lbFair'] - args['fairDuals']['ubFair'])/sum(~g)

        objective = gp.LinExpr(np.ones(sum(~Y)), np.array(delta)[~Y]) #Y = False misclass term
        objective.add(gp.LinExpr(np.array(args['mu'])*-1, np.array(delta)[Y])) #Y = True misclass term
        objective.add(gp.LinExpr(args['lam']*np.ones(len(z)), z)) #Complexity term
        objective.add(coeff_1*sum(np.array(delta)[~Y & g]) + coeff_2*sum(np.array(delta)[~Y & ~g]))

        return objective
  
    
    def computeObjective(self, X, Y, features, args):
        '''
        Returns reduced cost for pricing problem for given inputs
        '''
        if 'lam' not in args or 'mu' not in args or 'fairDuals' not in args or 'row_samples' not in args:
            raise Exception('Required arguments not supplied for NoFair Objective Computation.')
        
        if 'ubFair' not in args['fairDuals'] or 'lbFair' not in args['fairDuals']:
            raise Exception('Required fairness dual variables not supplied for NoFair Objective Computation.')
        
        classPos = np.all(X[:,features],axis=1)
        g = self.group[args['row_samples']]
        coeff_1 = 1+ (args['fairDuals']['ubFair'] - args['fairDuals']['lbFair'])/sum(g)
        coeff_2 = 1+ (args['fairDuals']['lbFair'] - args['fairDuals']['ubFair'])/sum(~g)

        return args['lam']*(1+len(features)) - np.dot(classPos[Y],args['mu']) + \
               coeff_1*sum(classPos[~Y & g]) + coeff_2*sum(classPos[~Y & ~g])

    
    
    def updateFairnessConstraint(self, column, constraints, args):
        column.addTerms(1/sum(self.group)*sum(args['K_z'][self.group[~args['Y']]]) - \
                        1/sum(~self.group)*sum(args['K_z'][~self.group[~args['Y']]]), 
                        constraints['ub'])
        column.addTerms(-1/sum(self.group)*sum(args['K_z'][self.group[~args['Y']]]) + \
                        1/sum(~self.group)*sum(args['K_z'][~self.group[~args['Y']]]), 
                        constraints['lb'])
        return

    
    def createFairnessConstraint(self, model, x, Y):
        '''
        Returns constraint for fairness
        '''
        constraints = {}
        x = np.array(x)
        g = self.group

        constraints['ub'] = model.addConstr( 1/sum(g)*sum(x[g & Y]) - 1/sum(~g)*sum(x[~g & Y]) <= self.eps, 
                                                name = 'ubFair')
        constraints['lb'] = model.addConstr( -1/sum(g)*sum(x[g & Y]) + 1/sum(~g)*sum(x[~g & Y])<= self.eps, 
                                                name = 'lbFair')

        return constraints
