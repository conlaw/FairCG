from scipy.stats import bernoulli
import numpy as np
from .RuleSampler import RuleSampler

class NaifRandomRuleSampler(RuleSampler):
    '''
    Randomnly select rules to return
    '''
    
    def __init__(self, args = {}):
        self.numRulesToReturn = args['numRulesToReturn'] if 'numRulesToReturn' in args else 20
    
    def getRules(self, rules, reduced_costs, col_samples):
        #Set number of rules to return
        returnNum = min(self.numRulesToReturn, len(rules))
        
        #Select rules
        rules_to_return = np.random.choice(range(len(rules)), returnNum, replace = False)
        
        #Convert to non-subsampled form (for cols)
        final_rules = []
        rules = np.array(rules)
        for rule in rules_to_return:
            fin_rule = np.zeros(len(col_samples))
            fin_rule[col_samples] = rules[rule]
            final_rules.append(fin_rule)
            

        return final_rules, np.array(reduced_costs)[rules_to_return]