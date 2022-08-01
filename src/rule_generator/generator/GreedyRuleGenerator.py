import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from .RuleGenerator import RuleGenerator
import time

class GreedyRuleGenerator(RuleGenerator):
    '''
    Implementation of IP Pricing Problem Solver
    '''
    
    def __init__(self, fairnessModule, args = {}):        
        #Set rule complexity if supplied in arguments
        self.fairnessModule = fairnessModule
        self.ruleComplex = 5
        self.numRulesToKeep = args['numRulesToKeep'] if 'numRulesToKeep' in args else 20
        
        
    def generateRule(self, X, Y, args):
        '''
        Solve the IP Pricing problem to generate new rule(s)
        '''
                           
        verbose = args['verbose'] if 'verbose' in args else False
        
        timed = False
        if 'timeLimit' in args:
            timeLimit = args['timeLimit']
            start_time = time.time()
            timed = True

        feature_set = [[]]
        good_rules = []
        good_rule_obj = []

        for i in range(self.ruleComplex):
            newFeatures = []
            res = []
            for f in feature_set:
                for i in range(X.shape[1]):
                    #If the feature is already in the feature set move on
                    if i in f:
                        continue
                    
                    #Create new feature set
                    newF = np.concatenate([f, [i]]).astype(np.int64)
                    newFeatures.append(newF)
                    
                    #Compute objective
                    obj = self.fairnessModule.computeObjective(X, Y, newF, args)
                    res.append(obj)
                    
                    #If reduced cost is negative add to rules
                    if obj < 0:
                        rule = np.zeros(X.shape[1])
                        rule[newF] = 1
                        good_rules.append(rule)
                        good_rule_obj.append(obj)
                    
                    if timed:
                        if time.time() - start_time > timeLimit:
                            break
                            
                if timed:
                    if time.time() - start_time > timeLimit:
                        break
            if timed:
                if time.time() - start_time > timeLimit:
                    break
            #Adjust feature set to features of size i, best numKeep number
            feature_set = np.array(newFeatures)[np.argsort(res)][:self.numRulesToKeep] 
            
        #Only return rules with negative reduced costs
        return good_rules, good_rule_obj

        
                    
        
            