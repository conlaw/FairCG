from .SubSampler import SubSampler
import numpy as np

class NotSoSubSampler(SubSampler):
    '''
    Sampler that returnss all th data
    '''
    
    def getSample(self, X, Y, coeff, args = {}):
        '''
        Returns all the data (note last two returned variables indicate which rows/cols are included)
        '''
        return X, Y, coeff, np.ones(X.shape[0]).astype(np.bool),  np.ones(X.shape[1]).astype(np.bool)      
