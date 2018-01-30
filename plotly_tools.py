import plotly.plotly as py
import plotly.graph_objs as go

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



