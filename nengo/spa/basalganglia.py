import nengo
from .. import objects
from ..templates import EnsembleArray
import nengo.networks as networks

from .rules import Rules
from .base import Module


class BasalGanglia(networks.BasalGanglia, Module):
    def make(self, rules, input_filter=0.002):
        self.rules = Rules(rules)
        self.input_filter = input_filter
        
        networks.BasalGanglia.make(self, dimensions=self.rules.count)
        
    
    def on_add(self, spa):
        self.rules.process(spa)
        
        for input, transform in self.rules.get_inputs().iteritems():
            
            input.connect_to(self.input, transform=transform, filter=self.input_filter)
    
            
        
        
        

