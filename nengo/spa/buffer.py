import nengo
from .. import objects
from ..templates import EnsembleArray
from .base import Module

import nengo.networks as networks

class Buffer(Module):
    def make(self, dimensions, subdimensions=16, neurons_per_dimension=50, 
                        vocab=None):
    
        if vocab is None:
            vocab = dimensions 
            
        self.state = self.add(networks.Array('state', dimensions/subdimensions,
                                nengo.LIF(neurons_per_dimension*subdimensions),
                                subdimensions))
                               
        self.inputs = dict(default=(self.state.input, vocab))
        self.outputs = dict(default=(self.state.output, vocab))
        
        

