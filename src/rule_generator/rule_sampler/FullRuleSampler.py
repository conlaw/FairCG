from scipy.stats import bernoulli
import numpy as np
from .RuleSampler import RuleSampler

class FullRuleSampler(RuleSampler):
    '''
    Return all the rules we found
    '''
    
    def __init__(self, args = {}):
        return
    
    def getRules(self, rules, reduced_costs, col_samples):
        
        #Get rules back into non-subsampled format
        final_rules = []
        for rule in rules:
            fin_rule = np.zeros(len(col_samples))
            fin_rule[col_samples] = rules[rule]
            final_rules.append(fin_rule)
        
        #Return all rules
        return final_rules, reduced_costs