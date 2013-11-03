import nengo
from .. import objects
from .base import Module
from .rules import Rules

import nengo.networks as networks
import numpy as np

class Cortical(Module):
    def make(self, rules, pstc=0.01):
        self.rules = Rules(rules)
        self.pstc = pstc
    
    def on_add(self, spa):
        Module.on_add(self, spa)
        
        self.rules.process(spa)

        
        for output, transform in self.rules.get_outputs_direct().iteritems():
            print output, transform
            raise Exception('Cortical module cannot yet handle direct connections')
            
        for index, route in self.rules.get_outputs_route():
            target, source = route

            if hasattr(source, 'convolve'):
                raise Exception('Cortical convolution not implemented yet')
            else:
                if source.invert:
                    raise Exception('Inverting on a communication channel not supported yet')
            
                if target.vocab is source.vocab:
                    transform = 1
                else:
                    transform = source.vocab.transform_to(target.vocab)
                    
                if hasattr(source, 'transform'):
                    t2 = source.vocab.parse(source.transform).get_convolution_matrix()
                    transform = np.dot(transform, t2)
                    
                source.obj.connect_to(target.obj, transform=transform, filter=self.pstc)
                
            
        

