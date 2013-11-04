import numpy as np

import nengo
import nengo.simulator as simulator
from nengo.nonlinearities import Direct, LIF, LIFRate
from nengo.builder import Builder
from nengo.builder import Signal
from nengo.builder import ProdUpdate, Reset, DotInc, Copy
from nengo.tests.helpers import unittest, rms

import logging
logger = logging.getLogger(__name__)


def testbuilder(model, dt):
    model.dt = dt
    model.seed = 0
    if not hasattr(model, 'probes'):
        model.probes = []
    return model


class TestSimulator(unittest.TestCase):
    Simulator = simulator.Simulator

    def test_signal_init_values(self):
        """Tests that initial values are not overwritten."""
        m = nengo.Model("test_signal_init_values")
        zero = Signal(value=[0])
        one = Signal(value=[1])
        five = Signal(value=[5.0])
        zeroarray = Signal(value=np.array([[0,0,0]]))
        array = Signal(value=np.array([1,2,3]))
        m.signals = [zero, one, five, array]
        m.operators = [ProdUpdate(zero, zero, one, five),
                       ProdUpdate(one, zeroarray, one, array)]

        sim = m.simulator(sim_class=self.Simulator, builder=testbuilder)
        self.assertEqual(1, sim.signals[sim.get(one)])
        self.assertEqual(5.0, sim.signals[sim.get(five)])
        self.assertTrue(np.all(
            np.array([1,2,3]) == sim.signals[sim.get(array)]))
        sim.step()
        self.assertEqual(1, sim.signals[sim.get(one)])
        self.assertEqual(5.0, sim.signals[sim.get(five)])
        self.assertTrue(np.all(
            np.array([1,2,3]) == sim.signals[sim.get(array)]))

    def test_steps(self):
        m = nengo.Model("test_signal_indexing_1")
        sim = m.simulator(sim_class=self.Simulator)
        self.assertEqual(0, sim.signals[sim.model.steps.signal])
        sim.step()
        self.assertEqual(1, sim.signals[sim.model.steps.signal])
        sim.step()
        self.assertEqual(2, sim.signals[sim.model.steps.signal])

    def test_time(self):
        m = nengo.Model("test_signal_indexing_1")
        sim = m.simulator(sim_class=self.Simulator)
        self.assertEqual(0.00, sim.signals[sim.model.t.signal])
        sim.step()
        self.assertEqual(0.001, sim.signals[sim.model.t.signal])
        sim.step()
        self.assertEqual(0.002, sim.signals[sim.model.t.signal])

    def test_signal_indexing_1(self):
        m = nengo.Model("test_signal_indexing_1")

        one = Signal(shape=1, name='a')
        two = Signal(shape=2, name='b')
        three = Signal(shape=3, name='c')
        tmp = Signal(shape=3, name='tmp')
        m.signals = [one, two, three, tmp]

        m.operators = [
            ProdUpdate(Signal(value=1), three[:1], Signal(value=0), one),
            ProdUpdate(Signal(value=2.0), three[1:], Signal(value=0), two),
            Reset(tmp),
            DotInc(Signal(value=[[0,0,1],[0,1,0],[1,0,0]]), three, tmp),
            Copy(src=tmp, dst=three, as_update=True),
        ]

        sim = m.simulator(sim_class=self.Simulator, builder=testbuilder)
        sim.signals[three] = np.asarray([1, 2, 3])
        sim.step()
        self.assertTrue(np.all(sim.signals[one] == 1))
        self.assertTrue(np.all(sim.signals[two] == [4, 6]))
        self.assertTrue(np.all(sim.signals[three] == [3, 2, 1]))
        sim.step()
        self.assertTrue(np.all(sim.signals[one] == 3))
        self.assertTrue(np.all(sim.signals[two] == [4, 2]))
        self.assertTrue(np.all(sim.signals[three] == [1, 2, 3]))

    def test_simple_direct_mode(self):
        dt = 0.001
        m = nengo.Model("test_simple_direct_mode")

        time = Signal(shape=1, name='time')
        sig = Signal(shape=1, name='sig')
        pop = Direct(n_in=1, n_out=1, fn=np.sin)
        m.signals = [sig, time]
        m.operators = []
        Builder().build_direct(pop, m, dt)
        m.operators += [
            ProdUpdate(
                Signal(value=dt), Signal(value=1), Signal(value=1), time),
            DotInc(Signal(value=[[1.0]]), time, pop.input_signal),
            ProdUpdate(
                Signal(value=[[1.0]]), pop.output_signal, Signal(value=0), sig)
        ]

        sim = m.simulator(sim_class=self.Simulator,
                          dt=dt,
                          builder=testbuilder)
        sim.step()
        for i in range(5):
            sim.step()

            t = (i + 2) * dt
            self.assertTrue(np.allclose(sim.signals[time], t),
                            msg='%s != %s' % (sim.signals[time], t))
            self.assertTrue(
                np.allclose(
                    sim.signals[sig], np.sin(t - dt*2)),
                msg='%s != %s' % (sim.signals[sig], np.sin(t - dt*2)))

    def test_encoder_decoder_pathway(self):
        #
        # This test is a very short and small simulation that
        # verifies (like by hand) that the simulator does the right
        # things in the right order.
        #
        m = nengo.Model("")
        dt = 0.001
        foo = Signal(shape=1, name='foo')
        pop = Direct(n_in=2, n_out=2, fn=lambda x: x + 1, name='pop')

        decoders = np.asarray([.2,.1])
        decs = Signal(value=decoders*0.5)

        m.signals = [foo, decs]
        m.operators = []
        Builder().build_direct(pop, m, dt)
        m.operators += [
            DotInc(Signal(value=[[1.0],[2.0]]), foo, pop.input_signal),
            ProdUpdate(decs, pop.output_signal, Signal(value=0.2), foo)
        ]

        def check(sig, target):
            self.assertTrue(np.allclose(sim.signals[sig], target),
                            "%s: value %s is not close to target %s" %
                            (sig, sim.signals[sig], target))

        sim = m.simulator(sim_class=self.Simulator,
                          dt=dt,
                          builder=testbuilder)

        # -- initialize things
        sim.signals[foo] = np.asarray([1.0])
        check(foo, 1.0)
        check(pop.input_signal, 0)
        check(pop.output_signal, 0)

        sim.step()
        #DotInc to pop.input_signal (input=[1.0,2.0])
        #produpdate updates foo (foo=[0.2])
        #pop updates pop.output_signal (output=[2,3])

        check(pop.input_signal, [1, 2])
        check(pop.output_signal, [2, 3])
        check(foo, .2)
        check(decs, [.1, .05])

        sim.step()
        #DotInc to pop.input_signal (input=[0.2,0.4])
        # (note that pop resets its own input signal each timestep)
        #produpdate updates foo (foo=[0.39]) 0.2*0.5*2+0.1*0.5*3 + 0.2*0.2
        #pop updates pop.output_signal (output=[1.2,1.4])

        check(decs, [.1, .05])
        check(pop.input_signal, [0.2, 0.4])
        check(pop.output_signal, [1.2, 1.4])
        # -- foo is computed as a prodUpdate of the *previous* output signal
        #    foo <- .2 * foo + dot(decoders * .5, output_signal)
        #           .2 * .2  + dot([.2, .1] * .5, [2, 3])
        #           .04      + (.2 + .15)
        #        <- .39
        check(foo, .39)

    def test_encoder_decoder_with_views(self):
        m = nengo.Model("")
        dt = 0.001
        foo = Signal(shape=1, name='foo')
        pop = Direct(n_in=2, n_out=2, fn=lambda x: x + 1, name='pop')

        decoders = np.asarray([.2,.1])

        m.signals = [foo]
        m.operators = []
        Builder().build_direct(pop, m, dt)
        m.operators += [
            DotInc(Signal(value=[[1.0], [2.0]]), foo[:], pop.input_signal),
            ProdUpdate(Signal(value=decoders * 0.5),
                       pop.output_signal,
                       Signal(value=0.2),
                       foo[:])
        ]

        def check(sig, target):
            self.assertTrue(np.allclose(sim.signals[sig], target),
                            "%s: value %s is not close to target %s" %
                            (sig, sim.signals[sig], target))

        sim = m.simulator(sim_class=self.Simulator,
                          dt=dt,
                          builder=testbuilder)

        #set initial value of foo (foo=1.0)
        sim.signals[foo] = np.asarray([1.0])
        #pop.input_signal = [0,0]
        #pop.output_signal = [0,0]

        sim.step()
        #DotInc to pop.input_signal (input=[1.0,2.0])
        #produpdate updates foo (foo=[0.2])
        #pop updates pop.output_signal (output=[2,3])

        check(foo, .2)
        check(pop.input_signal, [1, 2])
        check(pop.output_signal, [2, 3])

        sim.step()
        #DotInc to pop.input_signal (input=[0.2,0.4])
        # (note that pop resets its own input signal each timestep)
        #produpdate updates foo (foo=[0.39]) 0.2*0.5*2+0.1*0.5*3 + 0.2*0.2
        #pop updates pop.output_signal (output=[1.2,1.4])

        check(foo, .39)
        check(pop.input_signal, [0.2, 0.4])
        check(pop.output_signal, [1.2, 1.4])


class TestNonlinear(unittest.TestCase):
    Simulator = simulator.Simulator

    def test_direct(self):
        """Test direct mode"""

        dt = 0.001
        d = 3
        n_steps = 3
        n_trials = 3

        rng = np.random.RandomState(seed=987)

        for i in xrange(n_trials):
            A = rng.normal(size=(d,d))
            fn = lambda x: np.cos(np.dot(A, x))

            x = np.random.normal(size=d)

            m = nengo.Model("")
            ins = Signal(shape=d, name='ins')
            pop = Direct(n_in=d, n_out=d, fn=fn)
            m.signals = [ins]
            m.operators = []
            Builder().build_direct(pop, m, dt)
            m.operators += [
                DotInc(Signal(value=np.eye(d)), ins, pop.input_signal),
                ProdUpdate(Signal(value=np.eye(d)),
                           pop.output_signal,
                           Signal(value=0),
                           ins)
            ]

            sim = m.simulator(sim_class=self.Simulator,
                              dt=dt,
                              builder=testbuilder)
            sim.signals[ins] = x

            p0 = np.zeros(d)
            s0 = np.array(x)
            for j in xrange(n_steps):
                tmp = p0
                p0 = fn(s0)
                s0 = tmp
                sim.step()
                assert np.allclose(s0, sim.signals[ins])
                assert np.allclose(p0, sim.signals[pop.output_signal])

    def _test_lif_base(self, cls=LIF):
        """Test that the dynamic model approximately matches the rates"""
        rng = np.random.RandomState(85243)

        dt = 0.001
        d = 1
        n = 5e3

        m = nengo.Model("")
        ins = Signal(shape=d, name='ins')
        lif = cls(n)
        lif.set_gain_bias(max_rates=rng.uniform(low=10, high=200, size=n),
                          intercepts=rng.uniform(low=-1, high=1, size=n))
        m.signals = [ins]
        m.operators = []
        b = Builder()
        b._builders[cls](lif, m, dt)
        m.operators += [DotInc(Signal(
            value=np.ones((n,d))), ins, lif.input_signal)]

        sim = m.simulator(sim_class=self.Simulator,
                          dt=dt,
                          builder=testbuilder)
        sim.signals[ins] = 0.5 * np.ones(d)

        t_final = 1.0
        spikes = np.zeros(n)
        for i in xrange(int(np.round(t_final / dt))):
            sim.step()
            spikes += sim.signals[lif.output_signal]

        math_rates = lif.rates(sim.signals[lif.input_signal] - lif.bias)
        sim_rates = spikes / t_final
        logger.debug("ME = %f", (sim_rates - math_rates).mean())
        logger.debug("RMSE = %f",
                     rms(sim_rates - math_rates) / (rms(math_rates) + 1e-20))
        self.assertTrue(np.sum(math_rates > 0) > 0.5*n,
                        "At least 50% of neurons must fire")
        self.assertTrue(np.allclose(sim_rates, math_rates, atol=1, rtol=0.02))

    def test_lif(self):
        self._test_lif_base(cls=LIF)

    def test_lif_rate(self):
        self._test_lif_base(cls=LIFRate)


if __name__ == "__main__":
    nengo.log_to_file('log.txt', debug=True)
    unittest.main()
