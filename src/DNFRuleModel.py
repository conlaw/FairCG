import pandas as pd
import numpy as np
from RuleModel import RuleModel

class DNFRuleModel(RuleModel):
    '''
    Implementation of Rule Model for Binary DNF Rules
    '''
        
    def computeK(self, rules):
        '''
        Function to determine whether a data point meets a set of DNF rules
        '''
            
        #Data points meet a rule if all features in the rule have value True
        K = []
        for rule in rules:
            K.append(np.all(self.X[:,rule.astype(np.bool_)], axis=1))
        
        #Break down K matrix by sets P and Z
        K_p = np.transpose(np.array(K))[self.Y,:]
        K_z = np.transpose(np.array(K))[~self.Y,:]            
        
        if self.K_p is None:
            self.K_p = K_p
            self.K_z = K_z
        else:
            self.K_p = np.concatenate([self.K_p, K_p], axis = 1)
            self.K_z = np.concatenate([self.K_z, K_z], axis = 1)
        
        #Return the Kp matrix and how many data points are incorrectly classified by each rule
        return K_p, np.sum(K_z, axis = 0), K_z

    def computeRuleC(self, rules):
        '''
        Function to determine the complexity of a set of DNF rules
        '''
        
        #The complexity of a rule is just the number of features it includes
        c = []
        for rule in rules:
            c.append(sum(rule)+1)
        c = np.array(c)
        
        if self.C is None:
            self.C = c
        else:
            self.C = np.concatenate([self.C, c], axis = 0)
        
        return c
    
    def predict(self, X, rules, binary = True):
        
        if len(rules) == 0 or len(X) == 0:
            raise Exception('Need at least one rule and one data sample!')
        
        K = []
        for rule in rules:
            K.append(np.all(X[:,rule.astype(np.bool_)], axis=1))
        
        return np.sum(K, axis = 0) > 0 if binary else np.sum(K, axis = 0)
    
    def getNewRules(self, rules):
        '''
        Function to unique DNF rules
        '''
        
        if len(rules) == 0:
            return rules
        else:
            rules = np.unique(rules, axis = 0)
            
        #If there are no rules, every rule is new
        if self.rules is None:
            return rules
        
        #Iterate through new rules to check for novel rules
        new_rules = []
        for rule in rules:
            new = True
            for old_rule in self.rules:
                if np.array_equal(old_rule, rule):
                    new = False
                    break
            if new:
                new_rules.append(rule)
            
        return new_rules
