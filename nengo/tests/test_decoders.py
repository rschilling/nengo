
import numpy as np

import nengo
import nengo.core as core
import nengo.objects as objects
import nengo.helpers
# from nengo.core import ShapeMismatch
# from nengo.objects import Ensemble
import nengo.old_api as nef
import nengo.tests.helpers as helpers
from nengo.tests.helpers import Plotter, SimulatorTestCase, unittest
import nengo.decoders as decsolve

import logging

logger = logging.getLogger(__name__)


class TestDecoders(SimulatorTestCase):

    def _test_decoder_solver(self, solver, seed=None, n_neurons=100):
        name = 'test_decoder_solver_%s' % solver.__name__
        model = nengo.Model(name, seed=seed)
        def input_func(t):
            return (np.asarray(t) - 1.5).clip(-1, 1)
        input = model.make_node('in', output=input_func)
        ensemble = model.make_ensemble('A', nengo.LIF(n_neurons), 1, seed=seed+1)
        model.connect(input, ensemble)

        probe_filter = 0.05
        model.probe(input, filter=probe_filter)

        # model.probe(ensemble, filter=probe_filter, decoder_solver=solver)
        sample_every = 0.001
        probe = objects.Probe('ensemble.decoded_output', sample_every)
        con = ensemble.connect_to(probe, filter=probe_filter, decoder_solver=solver)
        ensemble.probes['decoded_output'].append(probe)
        model.probed[ensemble] = probe

        sim = model.simulator(dt=0.001, sim_class=self.Simulator)
        sim.run(3.0)

        t = sim.data(model.t)
        x = sim.data(input)
        y = sim.data(ensemble)
        return t, x, y

        # sim_ens = sim.get(ensemble)
        # sim_con = sim.get(con)
        # sim_ens = sim.get('A')
        # sim_con = sim.get(con)
        # print type(sim_ens)
        # print sim_ens.encoders
        # eval_points, activities = nengo.helpers.tuning_curves(sim_ens)
        # dec_acts = np.dot(activities, sim_con.decoders)

        # return t, x, y, (eval_points, activities, dec_acts)

    def test_decoder_solvers(self):
        solvers = [decsolve.lstsq,
                   decsolve.lstsq_noisy,
                   decsolve.lstsq_noisy_clip,
                   decsolve.direct_L2,
                   decsolve.direct_L2_low,
                   decsolve.direct_L2_perneuron]
                   # decsolve.dropout]

        # n_neurons = 30
        n_neurons = 50
        n_iters = 1

        seed = 893

        yyy = []
        tunings = []
        for solver in solvers:
            yy = []
            tunings0 = []
            for i in xrange(n_iters):
                t, x, y = self._test_decoder_solver(
                    solver, seed=seed + i, n_neurons=n_neurons)
                # t, x, y, tuning = self._test_decoder_solver(
                #     solver, seed=seed + i, n_neurons=n_neurons)
                yy.append(y.flatten())
                # tunings0.append(tuning)
            yyy.append(yy)
            # tunings.append(tunings0)

        t = t.flatten()
        x = x.flatten()
        yyy = np.asarray(yyy)

        with Plotter(self.Simulator) as plt:
            i = 0
            plt.figure(figsize=(6,10))

            # ax = plt.subplot(311)
            # evals, acts, _ = tunings[0][i]
            # ax.plot(evals, acts)

            # ax = plt.subplot(312)
            # for solver, tuning in zip(solvers, tunings):
            #     evals, _, dec_acts = tuning[i]
            #     ax.plot(evals, dec_acts, label=solver.__name__)

            # ax = plt.subplot(313)
            ax = plt.subplot(211)
            for solver, yy in zip(solvers, yyy):
                ax.plot(t, yy[i], label=solver.__name__)
            ax.plot(t, x, 'k--', label='input')
            ax.legend(loc=2, frameon=False)

            ax = plt.subplot(212)
            for solver, yy in zip(solvers, yyy):
                e = yy[i] - x
                e_sample = e.reshape((-1, 5)).mean(1)
                t_sample = t.reshape((-1, 5)).mean(1)
                ax.plot(t_sample, e_sample, label=solver.__name__)

            plt.savefig('test_decoder_solvers.pdf')
            plt.close()

        def calc_rmses(y):
            mean_std = []
            for t0, t1 in [(0.25, 0.5), (0.5, 2.5), (2.5, 3), (0.25, 3)]:
                m = (t > t0) & (t < t1)
                r = helpers.rms(y[:,m] - x[None,m], axis=-1)
                mean_std.append((r.mean(), r.std()))
            return mean_std

        rmses = map(calc_rmses, yyy)
        for solver, msms in zip(solvers, rmses):
            rmse_str = ', '.join(['%0.3e (%0.3e)' % ms for ms in msms])
            print "%20s: %s" % (solver.__name__, rmse_str)


if __name__ == "__main__":
    nengo.log_to_file('log.txt', debug=True)
    unittest.main()
