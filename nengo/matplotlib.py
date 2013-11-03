from __future__ import absolute_import

import numpy as np
import matplotlib.pyplot as plt
from . import objects

def rasterplot(time, spikes, ax=None, **kwargs):
    '''Generate a raster plot of the provided spike data

    Parameters
    ----------
    time : array
        Time data from the simulation
    spikes: array
        The spike data with columns for each neuron and 1s indicating spikes
    ax: matplotlib.axes.Axes
        The figure axes to plot into.

    Returns
    -------
    ax: matplotlib.axes.Axes
        The axes that were plotted into

    Examples
    --------
    >>> plt.figure()
    >>> rasterplot(sim.data(sim.model.t), sim.data('A.spikes'))
    '''

    if ax is None:
        ax = plt.gca()

    colors = kwargs.pop('colors', None)
    if colors is None:
        color_cycle = ax._get_lines.color_cycle
        colors = [next(color_cycle) for _ in xrange(spikes.shape[1])]

    if hasattr(ax, 'eventplot'):
        spikes = [time[spikes[:,i] > 0].flatten()
                  for i in xrange(spikes.shape[1])]
        for ix in xrange(len(spikes)):
            if spikes[ix].shape == (0,):
                spikes[ix] = np.array([-1])
        ax.eventplot(spikes, colors=colors, **kwargs)
        ax.set_ylim(len(spikes) - 0.5, -0.5)
        if len(spikes) == 1:
            ax.set_ylim(0.4, 1.6)  # eventplot plots different for len==1
        ax.set_xlim(left=0)

    else:
        # Older Matplotlib, doesn't have eventplot
        for i in xrange(spikes.shape[1]):
            ax.plot(time[spikes[:,i] > 0],
                    np.ones_like(np.where(spikes[:,i] > 0)).T + i, ',',
                    color=colors[i], **kwargs)

    return ax

def networkgraph(model, ax=None):
    import networkx as nx

    if ax is None:
        ax = plt.gca()

    G = nx.DiGraph()

    objs = model.objs.values()
    cons = []
    for o in objs:
        cons.extend(o.connections_out)
    # connections.extend(c for c in o.connections_out for o in objects)
    cons.extend(model.connections)

    # nodes = {}
    # for o in objects:
    #     nodes[o] = G.add_node(o.name)

    # for c in connections:
    #     if c.pre in nodes and c.post in nodes:
    #         G.add_edge(c.pre.name, c.post.name)

    nodes = {}
    for c in cons:
        print c.pre.name, c.post.name
        if c.pre in objs and c.post in objs:
            # if c.pre not in nodes:
            #     nodes[c.pre] = G.add_node(c.pre.name)
            # if c.post not in nodes:
            #     nodes[c.post] = G.add_node(c.post.name)

            if isinstance(c, objects.DecodedConnection):
                if c.function is not None:
                    name = c.function.func_name
                    # fn_str = name if name != '<lambda>' else
                    fn_str = name
                else:
                    fn_str = str(c.transform)
            else:
                fn_str = ''
            G.add_edge(c.pre.name, c.post.name, label=fn_str)

    pos = nx.spring_layout(G)
    nx.draw_networkx(G, pos, ax=ax, node_color='w', node_size=3000)

    edge_labels = dict(((u,v), d['label']) for u,v,d in G.edges(data=True))
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    return ax
