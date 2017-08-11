"""Demo script for """
from ccfepyutils.classes.scan import ParameterScan
from collections import OrderedDict
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import sys

def spike(x):
    x = x -0.0
    return 100*np.exp(-(((x/0.7)**2+5*x+1))) + 3*x - np.exp(x/3.) #+ 4*np.sin(x)

def dec_exp(x):
    y = np.exp(-2*x)
    return y

def wavy(x):
    y = np.cos(x) - x**2 / 40
    return y

def down_spike(x):
    x = x -0.0
    return -100*np.exp(-(((x/0.6)**2+7*x+1))) + 5*x - np.exp(x/3.) #+ 4*np.sin(x)

def wait_for_enter():
    if sys.version_info > (3, 0):
        input('Press enter to continue')
    else:
        raw_input('Press enter to continue')

def demo(func, lims, noise=0.05, time=30, end_pause=15, **kwargs):
    """Demo class on known function to demonstrate capabilties

    lims:       Limits of data to perfom scan over
    noise:      Level of gaussian noise to add to evaluted values to emulate experimental noise
    time:       Duration of main demo sequence
    end_pause:  Time to pause at end of demo
    """
    plt.ion()  # interactive mode

    # ** Instantiate the ParameterScan object with your settings **
    ps = ParameterScan(lims, **kwargs)
    print('Performing parameter scan of function {} with {} points'.format(func.__name__, ps.n_scan))

    # Time step size for interactive plot
    dt = (time-2.) / (ps.n_scan)

    # Evaluate function at fine resolution to fully resolve structure
    fine = np.linspace(lims[0], lims[1], 1000)
    fine = [fine, func(fine)]

    # Find points that would be returned used simple linear scan using the same number of points
    linear_scan = np.linspace(lims[0], lims[1], ps.n_scan)
    linear_scan = [linear_scan, func(linear_scan)]

    # Add random guassian 'experiemntal' noise to measurement
    linear_scan[1] = linear_scan[1] + np.random.randn(len(linear_scan[1]))*noise*np.mean(linear_scan[1])
    noisy_interp = sp.interpolate.interp1d(linear_scan[0], linear_scan[1])

    fig, ax = plt.subplots(num=func.__name__)

    # Plot the funtion we wish to accurately describe with out reduced sample
    art_true, = plt.plot(fine[0], fine[1], label='True distribution')
    plt.legend(loc='best')
    plt.pause(1)
    art_linear, = plt.plot(linear_scan[0], linear_scan[1], 'o-', label='Linear scan results')
    plt.legend(loc='best')
    plt.pause(1)
    art_true.set_visible(False)
    art_linear.set_visible(False)

    model = ps.model_data  # fitted spline data to existing points sampled by scan class
    art_model, = plt.plot(model[0], model[1], '--r', label='model')  # spline curve
    art_results, = plt.plot(ps._x[:-1], ps._y[:-1], '.k', label='results')  # black dots for points evaluated
    art_next = plt.axvline(np.mean(ps.lims), color='k', ls='--')  # vertical line marking where next point will be
    art_text = plt.annotate('', xy=(0.05, 0.95), xycoords=("axes fraction"))
    art_next.set_visible(False)
    for x in ps:  # ** Iterate over the scan object to return the next point to evaluate **
        # Show where the next point will be evaluated
        art_next.set_xdata(x)
        art_next.set_visible(True)
        art_text.set_text(ps._current_type)  # Update label for type of point being added
        plt.pause(dt)
        art_next.set_visible(False)

        # Calculate the value at x and update the scan object
        if noise > 0:
            y = noisy_interp(x)  # update with experimental/sim data + noise
        else:
            y = func(x)
        ps.add_data(x, y)  # ** Add new date to the scan object to update it's model to inform selection of next x **

        # Plot the new data
        model = ps.model_data
        art_model.set_data(model)  # update spline fit data
        art_results.set_data(ps._x[:], ps._y[:])  # mark previous points (excluding new point)
        plt.legend(loc='best')

    # Plot summary comparison of linear scan and parameter scan results with same number of points
    art_results.set_visible(False)
    art_model.set_visible(False)
    art_final, = plt.plot(ps.x, ps.y, 'o-g', label='model')  # final set of fitted points
    art_linear.set_visible(True)
    art_true.set_visible(True)
    plt.legend(loc='best')

    if end_pause == 'input':
        wait_for_enter()
    else:
        plt.pause(end_pause)

    print('Demo finished for function {}!'.format(func.__name__))
    ps.plot_data()

if __name__ == '__main__':

    functions = OrderedDict([(spike, dict(n_linear=5, n_refine=5, n_spread=4, n_extreema=2, order=1)),
                             (down_spike, dict(n_linear=5, n_refine=5, n_spread=4, n_extreema=2, order=1)),
                             (dec_exp, dict(n_linear=5, n_refine=5, n_spread=4, n_extreema=2, order=1)),
                             (wavy, dict(n_linear=5, n_refine=5, n_spread=4, n_extreema=2, order=1))])

    noise = 0.0
    lims = [-10, 10]
    time = 15
    end_pause = 5  # 'input'
    extrema_frac = 0.05

    for func, kwargs in functions.items():
        demo(func, lims, noise=noise, time=time, end_pause=end_pause, extrema_frac=extrema_frac, **kwargs)

    wait_for_enter()