from scipy.stats import bernoulli
import numpy as np
from .RuleSampler import RuleSampler

class TopXRuleSampler(RuleSampler):
    '''
    Returns the best X rules (based on reduced cost)
    '''
    
    def __init__(self, args = {}):
        self.numRulesToReturn = args['numRulesToReturn'] if 'numRulesToReturn' in args else 100
    
    def getRules(self, rules, reduced_costs, col_samples):
        #Sort rules by reduced cost
        sorted_rc = np.argsort(reduced_costs)
        
        #Convert to non-subsampled form
        final_rules = []
        rules = np.array(rules)
        for rule in sorted_rc[:self.numRulesToReturn]:
            fin_rule = np.zeros(len(col_samples))
            fin_rule[col_samples] = rules[rule]
            final_rules.append(fin_rule)

        return final_rules, np.array(reduced_costs)[sorted_rc[:self.numRulesToReturn]]