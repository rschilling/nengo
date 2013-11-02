from .. import objects
from . import Network

import copy
import numpy as np


class Array(Network):
    def make(self, length, neurons, dimensions, **ens_args):
        self.input = self.add(
                objects.PassthroughNode('input', dimensions=dimensions*length))
        
        self.ensembles = []
        transform = np.eye(dimensions*length)    
        for i in range(length):
            e = self.add(
                   objects.Ensemble('%d'%i, copy.deepcopy(neurons), dimensions))
            trans = transform[i*dimensions:(i+1)*dimensions,:]       
            self.input.connect_to(e, transform=trans, filter=None) 
            self.ensembles.append(e)
         
        self.add_output('output', dimensions, function=None)    
            
            
    def add_output(self, name, dimensions, function):
        length = len(self.ensembles)
    
        output = self.add(objects.PassthroughNode(name, dimensions=dimensions*length))
        setattr(self, name, output)
        
        transform = np.eye(dimensions*length)    
        for i,e in enumerate(self.ensembles):
            trans = transform[:,i*dimensions:(i+1)*dimensions]
            e.connect_to(output, transform=trans, filter=None, function=function)

