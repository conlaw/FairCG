import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from .data_sampler import *
from .generator import *
from .rule_sampler import *
import time

class GeneralRuleGenerator(object):
    '''
    General framework for generating promising rules (i.e. solving the pricing problem)
    '''
    
    def __init__(self, ruleMod, fairnessModule,
                 args = {},
                 sampler = 'random',
                 ruleGenerator = 'Greedy',
                 ruleSelect = 'topX'):
        
        #Set global variables
        self.args = args
        self.ruleMod = ruleMod
        self.fairnessModule = fairnessModule
        
        #Extract Rule Generator 
        ruleGenerator = args['ruleGenerator'] if 'ruleGenerator' in args else ruleGenerator
        
        #Define class variables
        self.initSampler(sampler)
        self.initRuleGenerator(ruleGenerator)
        self.initRuleSelect(ruleSelect)
   
   
    def generateRule(self, args = {}):
        '''
        Main function to generate a new rule, takes three steps:
            1. Subsamples rows + columns (dicted by sampler object)
            2. Generate rules (dictated by ruleGen object)
            3. Select rules to return to master problem (dictated by ruleSelect object)
        '''
        
        #Check that dual values are included
        if 'lam' not in args or 'coeff' not in args:
            raise Exception('Required arguments not supplied for DNF IP Rule Generator.')
        
        #Init variables
        final_rules = []
        sampling = True
        
        #Set-up timing if needed
        if 'timeLimit' in args:
            timeLimited = True
            timeLimit = args['timeLimit']
            start_time = time.perf_counter()
        else: 
            timeLimited = False
        
        #Repeat processs until we have rules to return (if we're subsampling)
        while (len(final_rules) == 0) and (sampling):
        
            #Sample Datasets
            X, Y, args['coeff'], args['row_samples'], col_samples = self.sampler.getSample(self.ruleMod.X, 
                                                                                                       self.ruleMod.Y, 
                                                                                                       args['coeff'])
            #If we return everything, we're not subsampling
            sampling = not (len(Y) == len(self.ruleMod.Y))
            
            #Generate Rules
            rules, rcs = self.ruleGen.generateRule(X, Y, args)

            #Subsample rules to return
            final_rules, final_rcs = self.ruleSelect.getRules(rules, rcs, col_samples)
            
            #Break-out if we run out of time
            if timeLimited:
                if time.perf_counter() - start_time >= timeLimit:
                    break
        
        #Return our rules and a flag indiciating if we found anything
        return len(final_rules) > 0 , final_rules
        
        
                
    def initSampler(self, sampler):
        '''
        Function that maps string rule models to objects
           - To add a new rule sampler simply add the object to the if control flow
        '''
        self.sampler_type = sampler
                
        if sampler == 'full':
            self.sampler = notsosubSampler.NotSoSubSampler()
        elif sampler == 'random':
            self.sampler = RandomSampler.RandomSampler(self.args)
        else:
            raise Exception('No associated rule model found.')
    
    def initRuleGenerator(self, ruleGenerator):
        '''
        Function that maps string rule generators to objects
           - To add a new rule generator simply add the object to the if control flow
        '''

        if ruleGenerator == 'DNF_IP':
            self.ruleGen = DNF_IP_RuleGenerator.DNF_IP_RuleGenerator(self.fairnessModule, self.args)
        elif ruleGenerator == 'DNF_IP_OPT':
            self.ruleGen = DNF_IP_RuleGeneratorOpt.DNF_IP_RuleGeneratorOpt(self.fairnessModule, self.args)
        elif ruleGenerator == 'Greedy':
            self.ruleGen = GreedyRuleGenerator.GreedyRuleGenerator(self.fairnessModule, self.args)
        elif ruleGenerator == 'Hybrid':
            self.ruleGen = HybridGenerator.HybridGenerator(self.fairnessModule, self.args)
        else:
            raise Exception('No associated rule generator found.')
            
    def initRuleSelect(self, ruleSelect):
        '''
        Function that maps string rule selection rules to objects
           - To add a new rule selection rule simply add the object to the if control flow
        '''
        if ruleSelect == 'full':
            self.ruleSelect = FullRuleSampler.FullRuleSampler(self.args)
        elif ruleSelect == 'topX':
            self.ruleSelect = TopXRuleSampler.TopXRuleSampler(self.args)
        elif ruleSelect == 'random':
            self.ruleSelect = NaifRandomRuleSampler.NaifRandomRuleSampler(self.args)
        elif ruleSelect == 'softmax':
            self.ruleSelect = SoftmaxRandomRuleSampler.SoftmaxRandomRuleSampler(self.args)
        else:
            raise Exception('No associated rule selector found.')

            
        
