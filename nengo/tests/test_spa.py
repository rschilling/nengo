import numpy as np

import nengo

from nengo.networks.basalganglia import BasalGanglia

from nengo.tests.helpers import SimulatorTestCase, unittest

import nengo.spa as spa
import nengo.networks


class TestSPA(SimulatorTestCase):
    def test_basic(self):
    
        model = nengo.Model('test_spa')
        D = 16
        
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
                self.add(spa.Memory('state', dimensions=D))
                self.add(spa.BasalGanglia('bg', rules=self.Rules))
                self.add(spa.Thalamus('thal', 'bg'))
                
                
                
        
        s = model.add(SpaSequence('SPA'))
        
        
        def input_func(t):
            if t<0.1: return s.modules['state'].inputs['default'][1].parse('A').v
            else: return [0]*D
        
        model.make_node('input', input_func)
        model.connect('input', 'SPA.state.state.input')
        
        
        model.probe('SPA.state.state.output', filter=0.03)
        model.probe('SPA.bg.input', filter=0.03)
        model.probe('SPA.bg.output', filter=0.03)
        model.probe('SPA.thal.rules.output', filter=0.01)
        
        sim = model.simulator(sim_class=self.Simulator)
        
        sim.run(1)
        
        
        import pylab
        pylab.subplot(4,1,1)
        pylab.plot(sim.data(model.t), sim.data('SPA.state.state.output'))
        pylab.ylabel('state')
        pylab.subplot(4,1,2)
        pylab.plot(sim.data(model.t), sim.data('SPA.bg.input'))
        pylab.ylabel('BG input')
        pylab.subplot(4,1,3)
        pylab.plot(sim.data(model.t), sim.data('SPA.bg.output'))
        pylab.ylabel('BG output')        
        pylab.subplot(4,1,4)
        pylab.plot(sim.data(model.t), sim.data('SPA.thal.rules.output'))
        pylab.ylabel('thal rules')        
        pylab.show()
        
        


if __name__ == "__main__":
    nengo.log_to_file('log.txt', debug=True)
    unittest.main()
