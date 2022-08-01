import pandas as pd
import numpy as np

class RuleModel(object):
    '''
    Parent class to translate between data, rules and the coefficients needed for the restricted model.
    To add a new type of RuleModel:
        - Create a new child class
        - Specify how to check a data point meets a rule (computeK)
        - Specify how to compute the complexity of a rule (computeRuleC)
    '''
    
    def __init__(self, X, Y):
        #Save data
        self.X = X
        self.Y = Y
        
        #Initialize constants
        self.reset()
        
    def computeK(self, rules):
        '''
        Takes a set of rules and returns K_p, and K_z coefficient
        - Needs to be specified in the child class
        '''
        pass
    
    def computeRuleC(self, rules):
        '''
        Takes a set of rules and returns vector of complexities
        - Needs to be specified in the child class
        '''
        pass
    
    def getNewRules(self, rules):
        '''
        Takes a set of rules and checks them against current rules to get new rules
        - Needs to be specified in the child class
        '''
        pass
    
    def predict(self, X, rules):
        '''
        Makes class label predictions given data samples and a set of rules
        - Needs to be specified in the child class
        '''
        pass
    
    def reset(self):
        self.rules = None
        self.K_p = None 
        self.K_z = None
        self.C = None 

    
    def addRule(self, rules):
        '''
        General function for taking new rules and computing coefficients
        '''
        #Confirm rules are new
        new_rules = self.getNewRules(rules)
        
        #If there are no new rules, return empty arrays
        if len(new_rules) == 0:
            return [], [], [], []
        
        K_p, K_z_coeff, K_z = self.computeK(new_rules)
        C = self.computeRuleC(new_rules)
        
        if self.rules is not None:
            self.rules = np.concatenate([self.rules, new_rules], axis = 0)
        else:
            self.rules = new_rules
        
        return K_p, K_z_coeff, C, K_z
    
    
