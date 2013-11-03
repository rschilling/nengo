
# Create the model object
import nengo
model = nengo.Model('Oscillator')

# Create the ensemble for the oscillator
model.make_ensemble('Neurons', nengo.LIF(200), dimensions=2)

import nengo.helpers

# Create an input signal
model.make_node('Input', output=nengo.helpers.piecewise({0:[1,0],0.1:[0,0]}))

# Connect the input signal to the neural ensemble
model.connect('Input','Neurons')

# Create the feedback connection
model.connect('Neurons','Neurons', transform=[[1,1],[-1,1]], filter=0.1)

import matplotlib.pyplot as plt
import nengo.matplotlib

# plt.ion()
plt.figure(1)
# plt.clf()
nengo.matplotlib.networkgraph(model)
plt.show()
