import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from .RuleGenerator import RuleGenerator
from .DNF_IP_RuleGeneratorOpt import DNF_IP_RuleGeneratorOpt
from .GreedyRuleGenerator import GreedyRuleGenerator

class HybridGenerator(RuleGenerator):
    '''
    Hybrid rule generator (use both greedy and ip)
    '''
    
    def __init__(self, fairnessModule, args = {}):
        
        #Set rule complexity if supplied in arguments
        self.fairnessModule = fairnessModule
        self.ruleComplex = args['ruleComplexity'] if 'ruleComplexity' in args else 100
        
        self.greedy = GreedyRuleGenerator(fairnessModule, args)
        self.ip = DNF_IP_RuleGeneratorOpt(fairnessModule, args)
        self.rule_count = 0
        self.doGreedy = True
        
    def generateRule(self, X, Y, args):
        '''
        Solve the IP Pricing problem to generate new rule(s)
        '''
        if self.rule_count < 4 and self.doGreedy:
            #print('Hybrid using greedy')
            rules, objs = self.greedy.generateRule(X,Y,args)
            
            if len(rules) == 0:
                self.doGreedy = False
        
        if self.rule_count >= 4 or not self.doGreedy:
            #print('Hybrid using IP')
            rules, objs = self.ip.generateRule(X,Y,args) 
        
        self.rule_count += 1
        return rules, objs
    
    def isFirstStage(self, args):
        if 'timeLeft' in args and 'timeLimit' in args:
            return float(args['timeLeft']) > 2 * float(args['timeLimit'])
        else:
            return self.rule_count  < 3
        



        
                    
        
            