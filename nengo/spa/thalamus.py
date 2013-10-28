import nengo
from .. import objects
import nengo.networks as networks

from .rules import Rules
from .base import Module

import numpy as np


class Thalamus(Module):
    def make(self, bg, neurons_per_rule=50, inhibit=1, inhibit_filter=0.008, 
                        output_filter=0.01, rule_threshold=0.2):
        self.bg = bg
        self.neurons_per_rule = neurons_per_rule
        self.inhibit = inhibit
        self.inhibit_filter = inhibit_filter
        self.output_filter = output_filter
        self.rule_threshold = rule_threshold
    
    def on_add(self, spa):
        if isinstance(self.bg, str):
            self.bg = spa.modules[self.bg]
            
        N = self.bg.rules.count                
            
            
        rules = self.add(networks.Array('rules', 
                                        N, nengo.LIF(self.neurons_per_rule), 1,
                                        threshold=nengo.objects.Uniform(self.rule_threshold, 1)))
        for ens in rules.ensembles:
            ens.encoders=[[1.0]]*self.neurons_per_rule                                        
        
        bias = self.add(objects.ConstantNode('bias', 1))
        
        rules.output.connect_to(rules.input, transform=(np.eye(N)-1)*self.inhibit,
                                 filter=self.inhibit_filter)
                                 
        bias.connect_to(rules.input, transform=np.ones((N,1)))                                         
    
        self.bg.output.connect_to(rules.input)
            
    
        for output, transform in self.bg.rules.get_outputs().iteritems():
            rules.output.connect_to(output, transform=transform, filter=self.output_filter)
            
        
        
        

