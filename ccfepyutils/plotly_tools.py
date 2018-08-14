"""Convenience functions for common plotly operations"""

from collections import OrderedDict
import numpy as np
import logging

logger = logging.getLogger(__name__)

from ccfepyutils.utils import make_iterable

try:
    import plotly
    import plotly.plotly as py
    import plotly.graph_objs as go
except ImportError as e:
    logger.error('Failed to import plotly')


def hoverinfo_text(names, values, format='0.3g'):
    """Return string suitable for hoverinfo keyword in plotly Scatter3d etc

    :names  - list of strings: names of parameters (will prepend each value)
    :values - list of arrays: values associated with each parameter for each point
    :format - number formating for values

    example call:
    text = hoverinfo_text(['tor', 'dr', 'dtor'], [tor, dr, dtor])  # where tor, dr, dtor are equal length arrays
    example usage:
    trace = go.Scatter3d(x=x, y=y, z=z, mode='markers', text=hoverinfo_text(axis_labels, values), hoverinfo='text')"""
    # from ccfepyutils.ccfe_array import make_iterable
    names, values = make_iterable(names, cast=list), make_iterable(values, cast=list)  # iterable
    for i in np.arange(len(values)):  # make sure values are flattened arrays
        values[i] = values[i].flatten()
    fmt = ''
    # Add information to dictionary
    dic = OrderedDict()
    # Construct the format string
    l = np.max([len(n) for n in names])
    for name, value in zip(names, values):
        dic[name] = value
        fmt += '{name:{l}s}= {{{name}:{format}}}<br>'.format(l=l+1, name=name, format=format)
    fmt = fmt[:-4]  # remove final line break
    hover_info = []
    # Loop over points, adding strings for each
    for i in range(0, len(values[0])):
        # Need to use html line breaks
        d = {k:v[i] if v[i] is not None else np.nan for k, v in dic.items()}
        hover_info.append(fmt.format(**d))
    return hover_info

def input_output_scatter3d(inp, output=None, extra_points={}, axis_labels=None, online=True, join_in_out=True,
                           online_name='scatter3d_tmp', offline_name='input_output_scatter.html', auto_open=False,
                           inp_label='input', output_label='output'):
    # import pdb; pdb.set_trace()
    import plotly.graph_objs as go
    if axis_labels is None:
        axis_labels = ('x', 'y', 'z')
    data = []
    x1, y1, z1 = inp[0].flatten(), inp[1].flatten(), inp[2].flatten()
    if output is None:
        join_in_out = False
    else:
        x2, y2, z2 = output[0].flatten(), output[1].flatten(), output[2].flatten()
    j = 0
    if join_in_out:  # Add lines connecting input and output scatter points
        for i, (xi, yi, zi, xj, yj, zj) in enumerate(zip(x1, y1, z1, x2, y2, z2)):
            if np.any([((v is None) or (np.isnan(v))) for v in (xi, yi, zi, xj, yj, zj)]):
                continue
            trace_line = go.Scatter3d(x=(xi, xj), y=(yi, yj), z=(zi, zj), mode='lines',
                                      line=dict(width=1.0, color='black'), opacity=0.2, hoverinfo='skip', #hoveron=False,
                                      # marker={'size': 0, 'opacity': 0.0},
                                      legendgroup='mappings', showlegend=(i in (0, len(x1)-1)), name='mappings',
                                      visible='legendonly')
                                      # marker={'size': 4, "line": {"width": 1}, 'opacity': 0.9}, name="output")
            data.append(trace_line)

    trace1 = go.Scatter3d(x=x1, y=y1, z=z1, mode='markers', text=hoverinfo_text(axis_labels, inp), hoverinfo='text',
                          marker={'size': 3, "line": {"width": 0.5}, 'opacity': 0.8, 'color': 'green'},
                          name="{} ({})".format(inp_label, len(x1)))
    data += [trace1]
    if output is not None:
        trace2 = go.Scatter3d(x=x2, y=y2, z=z2, mode='markers', text=hoverinfo_text(axis_labels, output),
                              hoverinfo='text',
                              marker={'size': 3, "line": {"width": 0.5}, 'opacity': 0.8, 'color': 'orange'},
                              name="{} ({})".format(output_label, len(x2)))
        data += [trace2]
    for key, (xe,ye,ze) in extra_points.items():
        trace_points = go.Scatter3d(x=xe, y=ye, z=ze, mode='markers', text=hoverinfo_text(axis_labels, (xe,ye,ze)),
                                    hoverinfo='text', marker={'size': 4.5, "line": {"width": 0.5}, 'opacity': 0.9,
                                                              'color': 'red'}, showlegend=True,
                                    name='{} ({})'.format(key, len(xe)))
        data.append(trace_points)
    tickfont=dict(size=16, color='black')#family='Old Standard TT, serif')
    titlefont=dict(size=24, color='black')#family='Old Standard TT, serif')
    legendfont=dict(size=16, color='black')#family='Old Standard TT, serif')
    layout = go.Layout(
                    scene = dict(
                    xaxis = dict(
                        title=axis_labels[0], tickfont=tickfont, titlefont=titlefont),
                    yaxis = dict(
                        title=axis_labels[1], tickfont=tickfont, titlefont=titlefont),
                    zaxis = dict(
                        title=axis_labels[2], tickfont=tickfont, titlefont=titlefont)),
                    # width=1500,
                    margin=dict(
                    r=0, b=10,
                    l=0, t=10),
                    legend=dict(x=.60, y=.80, bgcolor="rgba(240,240,240,0.8)", bordercolor='#FFFFFF', borderwidth=1,
                                font=legendfont)  # '#E2E2E2'
                  )
    if online:
        import plotly.plotly as py
        url = py.plot(data, layout=layout, filename=online_name, auto_open=auto_open)
        logger.info('Plotly input_output_scatter3d plot "{n}" saved online at url: {u}'.format(n=online_name, u=url))
        return url
    else:  # offline
        import plotly
        fig = go.Figure(data=data, layout=layout)
        plotly.offline.plot(fig, filename=offline_name, auto_open=auto_open)
        logger.info('Plotly input_output_scatter3d plot saved to local file: {n}'.format(n=offline_name))
        return fig

def plotly_surface(x, y, z, name='plotly_surface'):
    data = [
        go.Surface(
                x=x,
                y=y,
                z=z
        )
    ]
    layout = go.Layout(
            title=''
            # autosize=False,
            # width=500,
            # height=500,
            # margin=dict(
            #         l=65,
            #         r=50,
            #         b=65,
            #         t=90
            # )
    )
    fig = go.Figure(data=data, layout=layout)
    py.iplot(fig, filename=name)



