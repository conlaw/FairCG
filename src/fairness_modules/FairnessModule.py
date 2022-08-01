import numpy as np

class FairnessModule(object):
    '''
    Abstract object type for different implementing fairness constraints within CG framework
    '''
    
    def __init__(self, args = {}):
        
        #Fair Const Names stores names of constraints in model (used to retrieve duals)
        self.fairConstNames = []
        self.fairDuals = {}
        pass
        
    def defineObjective(self, rules, reduced_costs, col_samples):
        '''
        Returns gurobi objective for pricing problem
        '''
        pass  
    
    def computeObjective(self, X, Y, features, args):
        '''
        Returns reduced cost for pricing problem for given inputs
        '''
        pass  
    
    def computeReducedCosts(self, X, Y, rules, args):
        '''
        Returns reduced costs for all rules
        '''
        reduced_costs = []
        
        for rule in rules:
            reduced_costs.append(self.computeObjective(X, Y, np.nonzero(rule)[0], args))
        
        return np.array(reduced_costs)
    
    def extractDualVariables(self, constraint):
        '''
        Returns dict with dual variables related to fairness constraint
        '''
        self.fairDuals[constraint.ConstrName] = constraint.Pi
        
        return

    def createFairnessConstraint(self, model, x, Y):
        '''
        Returns constraint for fairness
        '''
        pass
    
    def updateFairnessConstraint(self, column, constraints, args):
        return
    
    def bulkComputeGroupKz(self,K_z, Y):
        return {}
        
            

