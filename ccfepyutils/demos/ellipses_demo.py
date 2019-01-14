import numpy as np
from ccfepyutils.geometry import Ellipses

def single_ellipse():

    # Create Ellipses object, with widths as half widths
    el = Ellipses(4, 2, 15, x=0, y=0, half_widths=False)

    # Create arrays defining grid to plot ellipse over
    x_arr = np.linspace(-20, 20, 501)
    y_arr = np.linspace(-20, 20, 501)

    # Plot ellipse taking widths as 1 sigma widths
    plot = el.plot_3d_superimposed(x_arr, y_arr, 1, show=False, mode='contourf', num='ellipse test_tmp')
    # Add rings at 2 sigma and 3 sigma
    el.plot(plot.ax(), scale_factor=2)
    el.plot(plot.ax(), scale_factor=3)
    plot.show()


def multi_ellipse():

    # Create Ellipses object, with widths as half widths
    el = Ellipses([4, 6], [2, 3], [15, -15], x=[0, 3], y=[0, -8], half_widths=False)

    # Create arrays defining grid to plot ellipse over
    x_arr = np.linspace(-20, 20, 501)
    y_arr = np.linspace(-20, 20, 501)

    # Plot ellipse taking widths as 1 sigma widths
    plot = el.plot_3d_superimposed(x_arr, y_arr, 1, show=False, mode='contourf', num='ellipse test_tmp', scale_factor=[1,2,3])
    # Add rings at 2 sigma and 3 sigma
    # el.plot(plot.ax(), scale_factor=2)
    # el.plot(plot.ax(), scale_factor=3)
    plot.show()


if __name__ == '__main__':
    single_ellipse()
    multi_ellipse()