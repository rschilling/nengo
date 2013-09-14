import nengo
from .. import objects
from ..templates import EnsembleArray
from .base import Module

class Memory(Module):
    def make(self, dimensions, neurons_per_dimension=50, filter=0.1, vocab=None):
    
        if vocab is None:
            vocab = dimensions 
            
        self.state = self.add(EnsembleArray('state', 
                               nengo.LIF(neurons_per_dimension*dimensions), 
                               dimensions, dimension_per_ensemble=16))
                               
        self.state.connect_to(self.state, filter=filter)
                                       
        
        self.inputs = dict(default=(self.state, vocab))
        self.outputs = dict(default=(self.state, vocab))
        
        

