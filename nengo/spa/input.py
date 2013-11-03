import nengo
from .. import objects
from .base import Module

import nengo.networks as networks

class Input(Module):
    def __init__(self, target_name, value):
        kwargs = dict(target_name=target_name, value=value)
        Module.__init__(self, 'input_%s'%target_name, **kwargs)
        
    def make(self, target_name, value):
        self.target_name = target_name
        self.value = value
    
    def on_add(self, spa):
        Module.on_add(self, spa)
        
        target, vocab = spa.get_module_input(self.target_name)
        if callable(self.value):
            val = lambda t: vocab.parse(self.value(t)).v
            self.input = self.add(objects.Node('input', val))
        else:
            val = vocab.parse(self.value).v
            self.input = self.add(objects.ConstantNode('input', val))
    
        self.input.connect_to(target, filter=None)        
        

