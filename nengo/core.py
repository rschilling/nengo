"""
Low-level objects
=================

These classes are used to describe a Nengo model to be simulated.
All other objects use describe models in terms of these objects.
Simulators only know about these objects.
"""

import copy
import logging

import numpy as np
import simulator as sim

import copy
import exception as exc

logger = logging.getLogger(__name__)

"""
Set assert_named_signals True to raise an Exception
if model.signal is used to create a signal with no name.

This can help to identify code that's creating un-named signals,
if you are trying to track down mystery signals that are showing
up in a model.
"""
assert_named_signals = False


def filter_coefs(pstc, dt):
    pstc = max(pstc, dt)
    decay = np.exp(-dt / pstc)
    return decay, (1.0 - decay)


class ShapeMismatch(ValueError):
    pass


class TODO(NotImplementedError):
    """Potentially easy NotImplementedError"""
    pass


class SignalView(object):
    """Base class of Signal, provides indexable view of subset of data."""
    # TODO(arvoelke): Document methods.

    def __init__(self, base, shape, elemstrides, offset, name=None):
        assert is_signal(base)
        self._base = base
        if isinstance(shape, int):
            shape = (shape,)
        self._shape = tuple(shape)
        self._elemstrides = tuple(elemstrides)
        self._offset = offset
        self._name = name

    def __len__(self):
        return self._shape[0]

    def __str__(self):
        return '%s{%s, %s}' % (
            self.__class__.__name__, self.name, self._shape)

    def __repr__(self):
        return '%s{%s, %s}' % (
            self.__class__.__name__, self.name, self._shape)

    @property
    def structure(self):
        return (self._shape, self._elemstrides, self._offset)

    @property
    def dtype(self):
        # Note: This cannot infinitely recurse because the supplied base class
        # must eventually be a derived class (since SignalView requires a base
        # class in order to be instantiated).
        return self.base.dtype

    @property
    def ndim(self):
        return len(self._shape)

    @property
    def size(self):
        return int(np.prod(self._shape))

    @property
    def base(self):
        return self._base

    @property
    def shape(self):
        return self._shape

    @property
    def offset(self):
        return self._offset

    @property
    def elemstrides(self):
        return self._elemstrides

    @property
    def name(self):
        if self._name is not None:
            return self._name
        if self._base is self:
            return '<anon%d>' % id(self)
        return 'View(%s[%d])' % (self._base.name, self._offset)

    @name.setter
    def name(self, value):
        self._name = value

    def same_view_as(self, other):
        return self.structure == other.structure \
           and self._base == other.base

    def rebase(self, newbase, name=None):
        if newbase.base != newbase:
            raise NotImplementedError()
        if newbase.structure != self._base.structure:
            raise NotImplementedError('technically ok but should not happen',
                                      (self._base, newbase))
        return SignalView(
            newbase, self._shape, self._elemstrides, self._offset, name)

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = shape[0]
        if self._elemstrides == (1,):
            size = int(np.prod(shape))
            if size != self.size:
                raise exc.ShapeMismatch(shape, self._shape)
            elemstrides = [1]
            for si in reversed(shape[1:]):
                elemstrides = [si * elemstrides[0]] + elemstrides
            return SignalView(self._base, shape, elemstrides, self._offset)
        elif self.size == 1:
            # -- scalars can be reshaped to any number of (1, 1, 1...)
            size = int(np.prod(shape))
            if size != self.size:
                raise ShapeMismatch(shape, self.shape)
            elemstrides = [1] * len(shape)
            return SignalView(self._base, shape, elemstrides, self._offset)
        else:
            # -- there are cases where reshaping can still work
            #    but there are limits too, because we can only
            #    support view-based reshapes. So the strides have
            #    to work.
            raise TODO('reshape of strided view')

    def transpose(self, neworder=None):
        if neworder:
            raise NotImplementedError()
        return SignalView(
                self.base,
                reversed(self.shape),
                reversed(self.elemstrides),
                self.offset,
                self.name + '.T'
                )

    @property
    def T(self):
        if self.ndim < 2:
            return self
        else:
            return self.transpose()

    def __getitem__(self, item):
        # -- copy the shape and strides
        shape = list(self._shape)
        elemstrides = list(self._elemstrides)
        offset = self._offset
        if isinstance(item, (list, tuple)):
            dims_to_del = []
            for ii, idx in enumerate(item):
                if isinstance(idx, int):
                    dims_to_del.append(ii)
                    offset += idx * elemstrides[ii]
                elif isinstance(idx, slice):
                    start, stop, stride = idx.indices(shape[ii])
                    offset += start * elemstrides[ii]
                    if stride != 1:
                        raise NotImplementedError()
                    shape[ii] = stop - start
            for dim in reversed(dims_to_del):
                shape.pop(dim)
                elemstrides.pop(dim)
            return SignalView(self._base, shape, elemstrides, offset)
        elif isinstance(item, (int, np.integer)):
            if len(self._shape) == 0:
                raise IndexError()
            if not (0 <= item < self._shape[0]):
                raise NotImplementedError()
            shape = self._shape[1:]
            elemstrides = self._elemstrides[1:]
            offset = self._offset + item * self._elemstrides[0]
            return SignalView(self._base, shape, elemstrides, offset)
        elif isinstance(item, slice):
            return self.__getitem__((item,))
        else:
            raise NotImplementedError(item)

    def add_to_model(self, model):
        if self._base not in model.signals:
            raise TypeError('Cannot add signal views. Add the signal instead.')

    def is_contiguous(self, return_range=False):
        def ret_false():
            if return_range:
                return False, None, None
            else:
                return False
        shape, strides, offset = self.structure
        if not shape:
            if return_range:
                return True, offset, offset + 1
            else:
                return True
        if len(shape) == 1:
            if strides[0] == 1:
                if return_range:
                    return True, offset, offset + shape[0]
                else:
                    return True
            else:
                return ret_false()
        if len(shape) == 2:
            if strides == (1, shape[0]) or strides == (shape[1], 1):
                if return_range:
                    return True, offset, offset + shape[0] * shape[1]
                else:
                    return True
            else:
                return ret_false()

        raise NotImplementedError()
        #if self.ndim == 1 and self.elemstrides[0] == 1:
            #return self.offset, self.offset + self.size

    def shares_memory_with(self, other):
        # XXX: WRITE SOME UNIT TESTS FOR THIS FUNCTION !!!
        # Terminology: two arrays *overlap* if the lowermost memory addressed
        # touched by upper one is higher than the uppermost memory address
        # touched by the lower one.
        #
        # np.may_share_memory returns True iff there is overlap.
        # Overlap is a necessary but insufficient condition for *aliasing*.
        #
        # Aliasing is when two ndarrays refer a common memory location.
        if self.base is not other.base:
            return False
        if self is other or self.same_view_as(other):
            return True
        if self.ndim < other.ndim:
            return other.shares_memory_with(self)
        if self.size == 0 or other.size == 0:
            return False

        assert self.ndim > 0
        if self.ndim == 1:
            # -- self is a vector view
            #    and other is either a scalar or vector view
            ae0, = self.elemstrides
            be0, = other.elemstrides
            amin = self.offset
            amax = amin + self.shape[0] * ae0
            bmin = other.offset
            bmax = bmin + other.shape[0] * be0
            if amin <= amax <= bmin <= bmax:
                return False
            elif bmin <= bmax <= amin <= amax:
                return False
            if ae0 == be0 == 1:
                # -- strides are equal, and we've already checked for
                #    non-overlap. They do overlap, so they are aliased.
                return True
            # TODO: look for common divisor of ae0 and be0
            raise NotImplementedError('1d',
                (self.structure, other.structure))
        elif self.ndim == 2:
            # -- self is a matrix view
            #    and other is either a scalar, vector or matrix view
            a_contig, amin, amax = self.is_contiguous(return_range=True)
            if a_contig:
                # -- self has a contiguous memory layout,
                #    from amin up to but not including amax
                b_contig, bmin, bmax = other.is_contiguous(return_range=True)
                if b_contig:
                    # -- other is also contiguous
                    if amin <= amax <= bmin <= bmax:
                        return False
                    elif bmin <= bmax <= amin <= amax:
                        return False
                    else:
                        return True
                raise NotImplementedError('2d self:contig, other:discontig',
                    (self.structure, other.structure))
            raise NotImplementedError('2d',
                (self.structure, other.structure))
        else:
            raise NotImplementedError()

    def to_json(self):
        return {
            '__class__' : '%s.%s' % (self.__module__, self.__class__.__name__),
            'name' : self.name,
            'base' : self.base.name,
            'shape' : list(self._shape),
            'elemstrides' : list(self._elemstrides),
            'offset' : self._offset,
        }


class Signal(SignalView):
    """Interpretable, multi-dimensional valued quantity within NEF."""

    def __init__(self, value, dtype, shape, elemstrides, offset, name):
        if assert_named_signals:
            assert name
        self._value = value
        self._dtype = np.dtype(dtype)
        super(Signal, self).__init__(self, shape, (1,), 0, name)

    @classmethod
    def make_vector(cls, n, name=None):
        return cls(np.zeros(n), np.float64, (n,), (1,), 0, name)

    @classmethod
    def make_constant(cls, x, 

    def __str__(self):
        return 'Signal(%dD, %s)' % (self.size, self.name)

    def __repr__(self):
        return str(self)

    @property
    def dtype(self):
        return self._dtype

    @property
    def value(self):
        return self._value

    def add_to_model(self, model):
        model.signals.append(self)

    def to_json(self):
        return {
            '__class__' : '%s.%s' % (self.__module__, self.__class__.__name__),
            'name' : self.name,
            'size' : self.size,
            'dtype' : str(self._dtype),
        }


class Constant(SignalView):
    """Binds a constant value with a signal."""

    def __init__(self, value, dtype=np.float64, name=None):
        self._dtype = np.dtype(dtype)
        self._value = np.asarray(value)
        s = np.asarray(self._value.strides)
        elemstrides = tuple(map(int, s / self._dtype.itemsize))
        super(Constant, self).__init__(
            self, self._value.shape, elemstrides, 0, name)

    def __str__(self):
        return 'Constant(%s, %s)' % (self._value, self.name)

    def __repr__(self):
        return str(self)

    @property
    def value(self):
        return self._value

    @property
    def dtype(self):
        return self._dtype

    def add_to_model(self, model):
        model.signals.append(self)

    def to_json(self):
        return {
            '__class__' : '%s.%s' % (self.__module__, self.__class__.__name__),
            'name' : self.name,
            'value' : self._value.tolist(),
        }


def is_signal(sig):
    """Return True `iff` sig is (or is derived from) a SignalView."""
    return isinstance(sig, SignalView)


def is_constant(sig):
    """Return True `iff` sig is (or is a view of) a Constant signal."""
    return isinstance(sig.base, Constant)


class Probe(object):
    """A model probe to record a signal."""

    def __init__(self, sig, dt):
        self.sig = sig
        self.dt = dt

    def __str__(self):
        return 'Probing %s' % self.sig

    def __repr__(self):
        return str(self)

    def add_to_model(self, model):
        model.probes.append(self)

    def to_json(self):
        return {
            '__class__' : '%s.%s' % (self.__module__, self.__class__.__name__),
            'sig' : self.sig.name,
            'dt' : self.dt,
        }


class Transform(object):
    """A linear transform from a decoded signal to the signals buffer."""

    def __init__(self, alpha, insig, outsig):
        if is_constant(outsig):
            raise TypeError('transform destination is constant')
        if is_constant(insig):
            raise TypeError('constant input (use filter instead)')

        # TODO(arvoelke): Deduplicate __init__ code between Transform and Filter
        name = '%s>%s.tf_alpha' % (insig.name, outsig.name)

        self.alpha_signal = Constant(alpha, name=name)
        self.insig = insig
        self.outsig = outsig
        if self.alpha_signal.size == 1:
            if self.insig.shape != self.outsig.shape:
                raise exc.ShapeMismatch(self.alpha_signal.shape,
                                        self.insig.shape,
                                        self.outsig.shape)
        elif self.alpha_signal.shape != self.outsig.shape + self.insig.shape:
            raise exc.ShapeMismatch(
                self.alpha_signal.shape, self.insig.shape, self.outsig.shape)

    def __str__(self):
        return 'Transform (id %d) from %s to %s' % (
            id(self), self.insig, self.outsig)

    def __repr__(self):
        return str(self)

    @property
    def alpha(self):
        return self.alpha_signal.value

    def add_to_model(self, model):
        model.signals.append(self.alpha_signal)
        dst = model._get_output_view(self.outsig)

        # XXX: Complicated lookup still necessary?
        if self.insig in model._decoder_outputs:
            insig = model._decoder_outputs[self.insig]
        elif self.insig.base in model._decoder_outputs:
            insig = self.insig.view_like_self_of(
                model._decoder_outputs[self.insig.base])
        else:
            insig = self.insig

        model._operators.append(
            sim.DotInc(self.alpha_signal, insig, dst,
                       tag='transform'))

    def to_json(self):
        return {
            '__class__' : '%s.%s' % (self.__module__, self.__class__.__name__),
            'alpha' : self.alpha.tolist(),
            'insig' : self.insig.name,
            'outsig' : self.outsig.name,
        }


class Filter(object):
    """A linear transform from signals[t-1] to signals[t]."""

    def __init__(self, alpha, oldsig, newsig):
        if is_constant(newsig):
            raise TypeError('filter destination is constant')

        name = '%s>%s.f_alpha' % (oldsig.name, newsig.name)

        self.alpha_signal = Constant(alpha, name=name)
        self.oldsig = oldsig
        self.newsig = newsig
        if self.alpha_signal.size == 1:
            if self.oldsig.shape != self.newsig.shape:
                raise exc.ShapeMismatch(self.alpha_signal.shape,
                                        self.oldsig.shape,
                                        self.newsig.shape)
        elif self.alpha_signal.shape != self.newsig.shape + self.oldsig.shape:
            raise exc.ShapeMismatch(
                self.alpha_signal.shape, self.oldsig.shape, self.newsig.shape)

    def __str__(self):
        return 'Filter (id %d) from %s to %s' % (
            id(self), self.oldsig, self.newsig)

    def __repr__(self):
        return str(self)

    @property
    def alpha(self):
        return self.alpha_signal.value

    def add_to_model(self, model):
        model.signals.append(self.alpha_signal)
        dst = model._get_output_view(self.newsig)

        model._operators.append(
            sim.DotInc(self.alpha_signal, self.oldsig, dst,
                       tag='transform'))

    def to_json(self):
        return {
            '__class__' : '%s.%s' % (self.__module__, self.__class__.__name__),
            'alpha' : self.alpha.tolist(),
            'oldsig' : self.oldsig.name,
            'newsig' : self.newsig.name,
        }


class Encoder(object):
    """A linear transform from a signal to a population."""

    def __init__(self, sig, pop, weights):
        self.sig = sig
        self.pop = pop
        weights = np.asarray(weights)
        if weights.shape != (pop.n_in, sig.size):
            raise ValueError('weight shape', weights.shape)

        name = '%s.encoders' % sig.name
        self.weights_signal = Constant(weights, name=name)

    def __str__(self):
        return 'Encoder (id %d) of %s to %s' % (id(self), self.sig, self.pop)

    def __repr__(self):
        return str(self)

    @property
    def weights(self):
        return self.weights_signal.value

    def add_to_model(self, model):
        model.signals.append(self.weights_signal)
        model._operators.append(
            sim.DotInc(
                self.sig,
                self.weights_signal,
                self.pop.input_signal,
                xT=True,
                tag='encoder'))


    def to_json(self):
        return {
            '__class__' : '%s.%s' % (self.__module__, self.__class__.__name__),
            'sig' : self.sig.name,
            'pop' : self.pop.name,
            'weights' : self.weights.tolist(),
        }


class Decoder(object):
    """A linear transform from a population to a signal."""

    def __init__(self, pop, sig, weights):
        self.pop = pop
        self.sig = sig
        weights = np.asarray(weights)
        if weights.shape != (sig.size, pop.n_out):
            raise ValueError('weight shape', weights.shape)
 
        name = '%s.decoders' % sig.name
        self.weights_signal = Constant(weights, name=name)

    def __str__(self):
        return 'Decoder (id %d) of %s to %s' % (id(self), self.pop, self.sig)

    def __repr__(self):
        return str(self)

    @property
    def weights(self):
        return self.weights_signal.value

    def add_to_model(self, model):
        model.signals.append(self.weights_signal)
        if self.sig.base not in model._decoder_outputs:
            sigbase = Signal(self.sig.base.size,
                             name='%s-decbase' % self.sig.name)
            model.signals.append(sigbase)
            model._decoder_outputs[self.sig.base] = sigbase
            model._operators.append(sim.Reset(sigbase))
        else:
            sigbase = model._decoder_outputs[self.sig.base]
        if self.sig == self.sig.base:
            dec_sig = sigbase
        else:
            dec_sig = self.sig.view_like_self_of(sigbase)
        model._decoder_outputs[self.sig] = dec_sig
        model._operators.append(
            sim.DotInc(
                self.pop.output_signal,
                self.weights_signal,
                dec_sig,
                xT=True,
                tag='Decoder'))

    def to_json(self):
        return {
            '__class__' : '%s.%s' % (self.__module__, self.__class__.__name__),
            'pop' : self.pop.name,
            'sig' : self.sig.name,
            'weights' : self.weights.tolist(),
        }


class Nonlinearity(object):
    """Base class for nonlinearities."""

    def __str__(self):
        return 'Nonlinearity(id=%d)' % id(self)

    def __repr__(self):
        return str(self)

    def add_to_model(self, model):
        # XXX: do we still need to append signals to model?
        model.signals.append(self.bias_signal)
        model.signals.append(self.input_signal)
        model.signals.append(self.output_signal)
        model._operators.append(
            self.operator(
                output=self.output_signal,
                J=self.input_signal,
                nl=self))
        # -- encoders will be scheduled between this copy
        #    and nl_op
        model._operators.append(
            sim.Copy(dst=self.input_signal, src=self.bias_signal))


class Direct(Nonlinearity):
    """A direct nonlinearity."""

    operator = sim.SimDirect

    def __init__(self, n_in, n_out, fn, name=None):
        if name is None:
            name = '<Direct%d>' % id(self)
        self.name = name

        self.input_signal = Signal(n_in, name='%s.input' % name)
        self.output_signal = Signal(n_out, name='%s.output' % name)
        self.bias_signal = Constant(np.zeros(n_in), name='%s.bias' % name)

        self.n_in = n_in
        self.n_out = n_out
        self.fn = fn

    def __deepcopy__(self, memo):
        try:
            return memo[id(self)]
        except KeyError:
            rval = self.__class__.__new__(self.__class__)
            memo[id(self)] = rval
            for k, v in self.__dict__.iteritems():
                if k == 'fn':
                    rval.fn = v
                else:
                    rval.__dict__[k] = copy.deepcopy(v, memo)
            return rval

    def __str__(self):
        return 'Direct(id=%d)' % id(self)

    def __repr__(self):
        return str(self)

    def fn(self, J):
        return J

    def to_json(self):
        return {
            '__class__' : '%s.%s' % (self.__module__, self.__class__.__name__),
            'input_signal' : self.input_signal.name,
            'output_signal' : self.output_signal.name,
            'bias_signal' : self.bias_signal.name,
            'fn' : self.fn.__name__,
        }


class _LIFBase(Nonlinearity):
    """Internal base class for LIF Neuron nonlinearity."""

    def __init__(self, n_neurons, tau_rc=0.02, tau_ref=0.002, name=None):
        if name is None:
            name = '<%s%d>' % (self.__class__.__name__, id(self))
        self.input_signal = Signal(n_neurons, name='%s.input' % name)
        self.output_signal = Signal(n_neurons, name='%s.output' % name)
        self.bias_signal = Constant(np.zeros(n_neurons), name='%s.bias' % name)

        self.name = name
        self._n_neurons = n_neurons
        self.tau_rc = tau_rc
        self.tau_ref = tau_ref
        self.gain = None

    def __str__(self):
        return '%s(id=%d, %dN)' % (
            self.__class__.__name__, id(self), self.n_neurons)

    def __repr__(self):
        return str(self)

    def to_json(self):
        return {
            '__class__' : '%s.%s' % (self.__module__, self.__class__.__name__),
            'input_signal' : self.input_signal.name,
            'output_signal' : self.output_signal.name,
            'bias_signal' : self.bias_signal.name,
            'n_neurons' : self.n_neurons,
            'tau_rc' : self.tau_rc,
            'tau_ref' : self.tau_ref,
            'gain' : self.gain.tolist(),
        }

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value
        self.input_signal.name = '%s.input' % value
        self.output_signal.name = '%s.output' % value
        self.bias_signal.name = '%s.bias' % value

    @property
    def bias(self):
        return self.bias_signal.value

    @property
    def n_in(self):
        return self.n_neurons

    @property
    def n_neurons(self):
        return self._n_neurons

    @property
    def n_out(self):
        return self.n_neurons

    def set_gain_bias(self, max_rates, intercepts):
        """Computes the alpha and bias.

        Parameters
        ---------
        max_rates : list of floats
            Maximum firing rates of neurons.
        intercepts : list of floats
            X-intercepts of neurons.

        Returns gain (alpha) and offset (j_bias) values needed to obtain the
        given max_rate and intercept values of neurons.
        """
        logging.debug('Setting gain and bias on %s', self.name)
        max_rates = np.asarray(max_rates)
        intercepts = np.asarray(intercepts)
        x = 1.0 / (1 - np.exp(
            (self.tau_ref - (1.0 / max_rates)) / self.tau_rc))
        self.gain = (1 - x) / (intercepts - 1.0)
        self.bias = 1 - self.gain * intercepts

    def rates(self, J_without_bias):
        """Returns LIF firing rates for current J in Hz.

        Parameters
        ---------
        J: ndarray of any shape
            membrane voltages
        tau_rc: broadcastable like J
            XXX
        tau_ref: broadcastable like J
            XXX
        """
        old = np.seterr(divide='ignore', invalid='ignore')
        try:
            J = J_without_bias + self.bias
            A = self.tau_ref - self.tau_rc * np.log(
                1 - 1.0 / np.maximum(J, 0))
            # if input current is enough to make neuron spike,
            # calculate firing rate, else return 0
            A = np.where(J > 1, 1 / A, 0)
        finally:
            np.seterr(**old)
        return A


class LIFRate(_LIFBase):
    """LIF Rate Neuron."""

    operator = sim.SimLIFRate

    def math(self, J):
        """Compute rates for input current (incl. bias)"""
        old = np.seterr(divide='ignore')
        try:
            j = np.maximum(J - 1, 0.)
            r = 1. / (self.tau_ref + self.tau_rc * np.log1p(1./j))
        finally:
            np.seterr(**old)
        return r
                
class LIF(_LIFBase):
    """LIF Spiking Neuron."""

    operator = sim.SimLIF

    def __init__(self, n_neurons, upsample=1, **kwargs):
        super(LIF, self).__init__(n_neurons, **kwargs)
        self.upsample = upsample
        self.voltage = Signal(n_neurons)
        self.refractory_time = Signal(n_neurons)

    def add_to_model(self, model):
        # XXX: do we still need to append signals to model?
        model.signals.append(self.bias_signal)
        model.signals.append(self.input_signal)
        model.signals.append(self.output_signal)
        model._operators.append(
            self.operator(
                output=self.output_signal,
                J=self.input_signal,
                nl=self, 
                voltage=self.voltage,
                refractory_time=self.refractory_time))
        # -- encoders will be scheduled between this copy
        #    and nl_op
        model._operators.append(
            sim.Copy(dst=self.input_signal, src=self.bias_signal))
        
    def to_json(self):
        d = _LIFBase.to_json(self)
        d['upsample'] = self.upsample
        return d

    def step_math0(self, dt, J, voltage, refractory_time, spiked):
        if self.upsample != 1:
            raise NotImplementedError()

        # N.B. J here *includes* bias

        # Euler's method
        dV = dt / self.tau_rc * (J - voltage)

        # increase the voltage, ignore values below 0
        v = np.maximum(voltage + dV, 0)

        # handle refractory period
        post_ref = 1.0 - (refractory_time - dt) / dt

        # set any post_ref elements < 0 = 0, and > 1 = 1
        v *= np.clip(post_ref, 0, 1)

        old = np.seterr(all='ignore')
        try:
            # determine which neurons spike
            # if v > 1 set spiked = 1, else 0
            spiked[:] = (v > 1) * 1.0

            # linearly approximate time since neuron crossed spike threshold
            overshoot = (v - 1) / dV
            spiketime = dt * (1.0 - overshoot)

            # adjust refractory time (neurons that spike get a new
            # refractory time set, all others get it reduced by dt)
            new_refractory_time = spiked * (spiketime + self.tau_ref) \
                                  + (1 - spiked) * (refractory_time - dt)
        finally:
            np.seterr(**old)

        # return an ordered dictionary of internal variables to update
        # (including setting a neuron that spikes to a voltage of 0)

        voltage[:] = v * (1 - spiked)
        refractory_time[:] = new_refractory_time
