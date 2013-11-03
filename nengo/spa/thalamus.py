import nengo
from .. import objects
import nengo.networks as networks

from .rules import Rules
from .base import Module

import numpy as np


class Thalamus(Module):
    def make(self, bg, neurons_per_rule=50, inhibit=1, pstc_inhibit=0.008, 
                        output_filter=0.01, rule_threshold=0.2, 
                        neurons_per_channel_dim=50, channel_subdim=16,
                        channel_pstc=0.01, neurons_cconv=200, 
                        neurons_gate=40, gate_threshold=0.3, pstc_to_gate=0.002):
        self.bg = bg
        self.neurons_per_rule = neurons_per_rule
        self.inhibit = inhibit
        self.pstc_inhibit = pstc_inhibit
        self.output_filter = output_filter
        self.rule_threshold = rule_threshold
        self.neurons_per_channel_dim = neurons_per_channel_dim
        self.channel_subdim = channel_subdim
        self.channel_pstc = channel_pstc
        self.neurons_gate = neurons_gate
        self.neurons_cconv = neurons_cconv
        self.gate_threshold = gate_threshold
        self.pstc_to_gate = pstc_to_gate
        
    
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
                                 filter=self.pstc_inhibit)
                                 
        bias.connect_to(rules.input, transform=np.ones((N,1)), filter=None)                                         
    
        self.bg.output.connect_to(rules.input)
            
    
        for output, transform in self.bg.rules.get_outputs_direct().iteritems():
            rules.output.connect_to(output, transform=transform, filter=self.output_filter)
            
        for index, route in self.bg.rules.get_outputs_route():
            target, source = route

            dim = target.vocab.dimensions

            gate = self.add(objects.Ensemble('gate_%d_%s'%(index, target.name),
                                    nengo.LIF(self.neurons_gate), dimensions=1,
                                    intercept=(self.gate_threshold, 1)))
            gate.encoders = [[1]]*self.neurons_gate
            rules.ensembles[index].connect_to(gate, pstc=self.pstc_to_gate, transform=-1)
            bias.connect_to(gate, filter=None)

            
            if hasattr(source, 'convolve'):
                # TODO: this is an insanely bizarre computation to have to do
                #   whenever you want to use a CircConv network.  The parameter
                #   should be changed to specify neurons per ensemble
                n_neurons_d = self.neurons_cconv * (
                    2*dim - (2 if dim % 2 == 0 else 1))            
                channel = self.add(networks.CircularConvolution('cconv_%d_%s'%(index, target.name), 
                                nengo.LIF(n_neurons_d), dim,
                                invert_a = source.invert, invert_b = source.convolve.invert))
                
                channel.output.connect_to(target.obj, filter=self.channel_pstc)                            

                transform = [[-1]]*(self.neurons_cconv)
                for e in channel.ensemble.ensembles:
                    gate.connect_to(e.neurons, transform=transform, filter=self.pstc_inhibit)

                # connect first input        
                if target.vocab is source.vocab:
                    transform = 1
                else:
                    transform = source.vocab.transform_to(target.vocab)
                    
                if hasattr(source, 'transform'):
                    t2 = source.vocab.parse(source.transform).get_convolution_matrix()
                    transform = np.dot(transform, t2)
                    
                source.obj.connect_to(channel.A, transform=transform, filter=self.channel_pstc)

                # connect second input        
                if target.vocab is source.convolve.vocab:
                    transform = 1
                else:
                    transform = source.convolve.vocab.transform_to(target.vocab)
                    
                if hasattr(source.convolve, 'transform'):
                    t2 = source.convolve.vocab.parse(source.convolve.transform).get_convolution_matrix()
                    transform = np.dot(transform, t2)
                    
                source.convolve.obj.connect_to(channel.B, transform=transform, filter=self.channel_pstc)

                
                    
                    
                
            else:
            
                if source.invert:
                    raise Exception('Inverting on a communication channel not supported yet')
            
                subdim = self.channel_subdim
                channel = self.add(networks.Array('channel_%d_%s'%(index, target.name), 
                                dim/subdim, nengo.LIF(self.neurons_per_channel_dim*subdim), subdim))
                
                channel.output.connect_to(target.obj, filter=self.channel_pstc)                            

                transform = [[-1]]*(self.neurons_per_channel_dim*subdim)
                for e in channel.ensembles:
                    gate.connect_to(e.neurons, transform=transform, filter=self.pstc_inhibit)

                
                if target.vocab is source.vocab:
                    transform = 1
                else:
                    transform = source.vocab.transform_to(target.vocab)
                    
                if hasattr(source, 'transform'):
                    t2 = source.vocab.parse(source.transform).get_convolution_matrix()
                    transform = np.dot(transform, t2)
                    
                source.obj.connect_to(channel.input, transform=transform, filter=self.channel_pstc)
            
            
            
                                
        

