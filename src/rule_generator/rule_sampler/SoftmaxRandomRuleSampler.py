from scipy.stats import bernoulli
import numpy as np
from .RuleSampler import RuleSampler

class SoftmaxRandomRuleSampler(RuleSampler):
    '''
    Randomnly select which rules to return by building a softmax dist. based on reduced costs
    '''
    
    def __init__(self, args = {}):
        self.numRulesToReturn = args['numRulesToReturn'] if 'numRulesToReturn' in args else 20
    
    def getRules(self, rules, reduced_costs, col_samples):
        if len(rules) == 0:
            return [], []
        
        returnNum = min(self.numRulesToReturn, len(rules))
        
        #Construct softmax distrubtion
        probs = np.exp(-1*np.array(reduced_costs))
        probs = probs/sum(probs)
                
        #Randomnly select rules based on softmax dist
        rules_to_return = list(np.random.choice(len(rules), returnNum, p=probs, replace = False))
        
        #Convert rules to non-subsampled form
        final_rules = []
        rules = np.array(rules)
        for rule in rules_to_return:
            fin_rule = np.zeros(len(col_samples))
            fin_rule[col_samples] = rules[rule]
            final_rules.append(fin_rule)
            
        return final_rules, np.array(reduced_costs)[rules_to_return]