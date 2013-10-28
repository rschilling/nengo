import nengo
from .. import objects
from ..templates import EnsembleArray
from .base import Module

import nengo.networks as networks

class Memory(Module):
    def make(self, dimensions, subdimensions=16, neurons_per_dimension=50, 
                        filter=0.01, vocab=None):
    
        if vocab is None:
            vocab = dimensions 
            
        self.state = self.add(networks.Array('state', dimensions/subdimensions,
                                nengo.LIF(neurons_per_dimension*subdimensions),
                                subdimensions))
        #self.state = self.add(EnsembleArray('state', 
        #                       nengo.LIF(neurons_per_dimension*dimensions), 
        #                       dimensions, dimension_per_ensemble=16))
                               
        self.state.output.connect_to(self.state.input, filter=filter)
                                       
        
        self.inputs = dict(default=(self.state.input, vocab))
        self.outputs = dict(default=(self.state.output, vocab))
        
        

