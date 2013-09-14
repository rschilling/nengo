import numpy as np

import nengo

from nengo.networks.basalganglia import BasalGanglia

from nengo.tests.helpers import SimulatorTestCase, unittest

import nengo.spa as spa
import nengo.networks


class TestSPA(SimulatorTestCase):
    def _test_connect(self):
        model = nengo.Model('test_connect')
        
        A = model.add(nengo.objects.PassthroughNode('A', 10))
        
        B = model.add(nengo.objects.PassthroughNode('B', 3))
        
        A.connect_to(B, transform=np.zeros((10,3)))
        
        sim = model.simulator(sim_class=self.Simulator)        
        sim.run(0.2)
        
        


    def test_basic(self):
    
        model = nengo.Model('test_spa')
        
        class SpaSequence(spa.SPA):
            class Rules:
                def rule_1():
                    match(state='A')
                    effect(state='B')
                def rule_2():
                    match(state='B')
                    effect(state='C')
                def rule_3():
                    match(state='C')
                    effect(state='D')
                def rule_4():
                    match(state='D')
                    effect(state='E')
                def rule_5():
                    match(state='E')
                    effect(state='A')
        
            def make(self):
                self.add(spa.Memory('state', dimensions=16))
                self.add(spa.BasalGanglia('bg', rules=self.Rules))
                self.add(spa.Thalamus('thal', 'bg'))
                
                
                
        
        s = model.add(SpaSequence('SPA'))
        
        
        def input_func(t):
            if t<0.1: return s.modules['state'].inputs['default'][1].parse('A').v
            else: return [0]*16
        
        model.make_node('input', input_func)
        model.connect('input', 'SPA.state.state')
        
        
        model.probe('SPA.state.state', filter=0.03)
        model.probe('SPA.bg.input', filter=0.03)
        model.probe('SPA.bg.output', filter=0.03)
        model.probe('SPA.thal.rules', filter=0.03)
        
        sim = model.simulator(sim_class=self.Simulator)
        
        sim.run(1)
        
        
        import pylab
        pylab.subplot(4,1,1)
        pylab.plot(sim.data(model.t), sim.data('SPA.state.state'))
        pylab.ylabel('state')
        pylab.subplot(4,1,2)
        pylab.plot(sim.data(model.t), sim.data('SPA.bg.input'))
        pylab.ylabel('BG input')
        pylab.subplot(4,1,3)
        pylab.plot(sim.data(model.t), sim.data('SPA.bg.output'))
        pylab.ylabel('BG output')        
        pylab.subplot(4,1,4)
        pylab.plot(sim.data(model.t), sim.data('SPA.thal.rules'))
        pylab.ylabel('thal rules')        
        pylab.show()
        
        


if __name__ == "__main__":
    nengo.log_to_file('log.txt', debug=True)
    unittest.main()
