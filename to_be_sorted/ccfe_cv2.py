#!/usr/bin/env python
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
"""
Re-written version of James Young's filament tracking routines. Replaces trackingScripts2.py
"""
__author__ = 'tfarley'

from copy import deepcopy
import numpy as np
from scipy import interpolate as spinterpolate
from scipy.interpolate import InterpolatedUnivariateSpline
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

import cv2

try: import cpickle as pickle
except: import pickle

def scalled_ellipse(contour, x, y, stretch=True, verbose=False):
    """ Take thresholded image and fit ellipse

    For angle=0: minor axis = width, Major axis = height
    matplotlib ellipse rotates anticlockwise in degrees
    """
    from EllipseFitter import EllipseFitter
    import numbers
    error = []

    # Grid spacing
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    # Stretch for equal aspect ratio. NOTE: TODO: Stretch factor must be integer to avoid integer rounding errors!
    if stretch:
        stretch = dy/dx
        assert np.any(np.isclose(np.repeat(stretch % 1, 2), [0, 1])), \
            'Stretch factor must be integer to avoid integer rounding errors! Stretch = {}'.format(stretch)
        stretch = int(np.round(stretch))
    else:
        stretch = 1
    ear_contour = deepcopy(contour).astype(np.int32)
    ear_contour[:, 0, 1] *= stretch  # stretch 2nd coordinate by required amount for equal aspect

    thresh = np.zeros((len(y)*stretch, len(x)), dtype=np.int32)
    cv2.fillPoly(thresh, pts =[ear_contour[:,0,:]], color=(255,255,255))
    try:
        angle,imajor,iminor,xCenter,yCenter = EllipseFitter(thresh)
        icentre = [yCenter-0.5, xCenter-0.5]
        # icentre, (iminor, imajor), angle = cv2.fitEllipse(ear_contour.astype(np.int64))  # Potential rounding ERRORS!
    except Exception as e:
        print('Failed to fit ellipse to contour: {}\n{}'.format(e, contour))
        raise e
    icentre = np.array(icentre)

    if iminor < 1e-5:
        import pdb; pdb.set_trace()

    ## Coordiantes of extrema at ends of minor and major axes in ear index coords
    it = [0, imajor/2]  # top
    ib = [0, -imajor/2]  # bottom
    il = [-iminor/2, 0]  # left
    ir = [iminor/2, 0]  # right

    iextrema_unrot = np.matrix([it,ib,il,ir]).T  # unrotated extrema
    angle_rad = np.deg2rad(angle)
    rot_matrix = np.matrix([[np.cos(angle_rad), -np.sin(angle_rad)], [np.sin(angle_rad), np.cos(angle_rad)]])

    ## Rotate the extrema points about the ellipse centre
    iextrema = rot_matrix * iextrema_unrot  # rotate pixel coords relative to ellipse centre

    ## Now rotation is complete add on centre coords
    iextrema[0] += icentre[0]
    iextrema[1] += icentre[1]
    iextrema_unrot[0] += icentre[0]
    iextrema_unrot[1] += icentre[1]

    ## Reverse stretch in phi direction
    icentre[1] /= stretch
    iminor /= stretch
    iextrema[1] /= stretch
    iextrema_unrot[1] /= stretch

    ## Interpolate/extrapolate to convert to Rphi coords (can use fractional coords)
    fR = InterpolatedUnivariateSpline(np.arange(len(x)), x, k=2)  # quad spline interpolation+extrapolation
    fphi = InterpolatedUnivariateSpline(np.arange(len(y)), y, k=2)  # quad spline interpolation+extrapolation

    centre = np.array([fR(icentre[0]), fphi(icentre[1])])  # Centre point of ellipse

    extrema = np.array([[fR(v) for v in np.array(iextrema[0])[0]], [fphi(v) for v in np.array(iextrema[1])[0]]])
    # extrema_unrot = np.array([[fR(v) for v in np.array(iextrema_unrot[0])[0]],
    #                           [fphi(v) for v in np.array(iextrema_unrot[1])[0]]])

    major = np.linalg.norm(extrema[:,0]-extrema[:,1])  # top - bottom
    minor = np.linalg.norm(extrema[:,3]-extrema[:,2])  # right - left
    # assert major == imajor * dx
    # assert minor == iminor * stretch * dx

    # Zweeben2016 method: Find length of axes spanned by ellipse centred at origin
    x_width_ax = 2 * ((np.cos(angle_rad)/major)**2 + (np.sin(angle_rad)/minor)**2)**(-0.5)
    y_width_ax = 2 * ((np.sin(angle_rad)/major)**2 + (np.cos(angle_rad)/minor)**2)**(-0.5)

    # major_unrot = np.linalg.norm(extrema_unrot[:,0]-extrema_unrot[:,1])  # top - bottom
    # minor_unrot = np.linalg.norm(extrema_unrot[:,3]-extrema_unrot[:,2])  # right - left

    # Use parametric coords to find extrema of ellipse (ie full x and y extent)
    from numpy import arctan2, arctan, tan, cos, sin
    tx = np.arctan2(-minor*np.tan(angle_rad), major)  # p47 of labbook
    ty = np.arctan2(major, minor*np.tan(angle_rad))
    x_width_full = np.abs(minor*cos(tx)*cos(angle_rad) - major*sin(tx)*sin(angle_rad))  # * 2
    y_width_full = np.abs(minor*cos(ty)*sin(angle_rad) + major*sin(ty)*cos(angle_rad))  # * 2

    ellipticity = np.sqrt(1-(minor/major)**2) if major >= minor else np.sqrt(1-(major/minor)**2)

    if True:
        thresh = np.zeros((len(y), len(x)), dtype=np.int32)
        cv2.fillPoly(thresh, pts=[contour[:, 0, :]], color=(255, 255, 255))
        iangle, imajor, iminor, xCenter, yCenter = EllipseFitter(thresh)
    else:
        iangle = 0

    ellipse = {'centre': centre, 'dx': dx, 'dy': dy, 'angle': angle, 'radians': angle_rad,
               'width': minor, 'height': major, 'x_width_ax': x_width_ax, 'y_width_ax': y_width_ax,
               'x_width_full': x_width_full, 'y_width_full': y_width_full,
               'ellipticity': ellipticity, 'rot_extrema': extrema, 'error': error}  # ellipse in R-phi coords
    i_ellipse = {'centre': icentre, 'width': iminor, 'height': imajor, 'angle': iangle, 'radians': angle_rad,
                 'elliptcicity': ellipticity, 'rot_extrema': iextrema}  # ellipse in index coords

    if verbose:
        print('centre: {}, dR: {}, dphi: {}, angle: {} deg'.format(centre, dy, dy, angle))
        # print('box:\n', box)
        # print('ellipse:\n', ellipse)
    return ellipse, i_ellipse

def scalled_ellipse_cv2(contour, x, y, verbose=False):
    """ Take contour from cv2.fitContour and convert to ellipse parameters in R-phi space
    cv2.fitEllipse(cnt) gives: (angle in degrees)
    box = ((iR_centre, iphi_centre), (minor axis, Major axis), rotation_angle)
    For angle=0: minor axis = width, Major axis = height
    matplotlib ellipse rotates anticlockwise in degrees
    """
    error = []  # record of operations that have failed

    ## Convert contour to equal aspect ratio (ear) for ellipse fitting to get correct tilt angle etc
    ## Find local dphi
    # moments = cv2.moments(contour)
    # try:
    #     cx = moments['m10']/moments['m00']
    #     cy = moments['m01']/moments['m00']
    # except ZeroDivisionError:
    #     cx = np.mean(contour[:,:,0])
    #     cy = np.mean(contour[:,:,1])
    # dR = R[cx+1] - R[cx]
    # dphi = phi[cy+1] - phi[cy]

    # Grid spacing
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    # Stretch for equal aspect ratio. NOTE: TODO: Stretch factor must be integer to avoid integer rounding errors!
    stretch = dy/dx
    ear_contour = deepcopy(contour).astype(float)
    ear_contour[:, 0, 1] *= stretch  # stretch 2nd coordinate by required amount for equal aspect

    ## Returns the rotated rectangle in which the ellipse is inscribed in index coordinates
    try:
        icentre, (iminor, imajor), angle = cv2.fitEllipse(ear_contour.astype(np.int64))  # Potential rounding ERRORS!
    except Exception as e:
        print('Failed to fit ellipse to contour: {}\n{}'.format(e, contour))
        raise e
    icentre = np.array(icentre)

    if iminor < 1e-5:
        print('ERROR: cv2.fitEllipse failed to fit contour correctly')
        # import pdb; pdb.set_trace()

    ## Coordiantes of extrema at ends of minor and major axes in ear index coords
    it = [0,imajor/2]  # top
    ib = [0,-imajor/2]  # bottom
    il = [-iminor/2, 0]  # left
    ir = [iminor/2, 0]  # right

    iextrema_unrot = np.matrix([it,ib,il,ir]).T  # unrotated extrema
    angle_rad = np.deg2rad(angle)
    rot_matrix = np.matrix([[np.cos(angle_rad), -np.sin(angle_rad)], [np.sin(angle_rad), np.cos(angle_rad)]])

    ## Rotate the extrema points about the ellipse centre
    iextrema = rot_matrix * iextrema_unrot  # rotate pixel coords relative to ellipse centre

    ## Now rotation is complete add on centre coords
    iextrema[0] += icentre[0]
    iextrema[1] += icentre[1]
    iextrema_unrot[0] += icentre[0]
    iextrema_unrot[1] += icentre[1]

    ## Reverse stretch in phi direction
    icentre[1] /= stretch
    iminor /= stretch
    iextrema[1] /= stretch
    iextrema_unrot[1] /= stretch

    ## Interpolate/extrapolate to convert to Rphi coords (can use fractional coords)
    fR = InterpolatedUnivariateSpline(np.arange(len(x)), x, k=2)  # quad spline interpolation+extrapolation
    fphi = InterpolatedUnivariateSpline(np.arange(len(y)), y, k=2)  # quad spline interpolation+extrapolation

    centre = np.array([fR(icentre[0]), fphi(icentre[1])])  # Centre point of ellipse

    extrema = np.array([[fR(v) for v in np.array(iextrema[0])[0]], [fphi(v) for v in np.array(iextrema[1])[0]]])
    # extrema_unrot = np.array([[fR(v) for v in np.array(iextrema_unrot[0])[0]],
    #                           [fphi(v) for v in np.array(iextrema_unrot[1])[0]]])

    major = np.linalg.norm(extrema[:,0]-extrema[:,1])  # top - bottom
    minor = np.linalg.norm(extrema[:,3]-extrema[:,2])  # right - left
    # assert major == imajor * dx
    # assert minor == iminor * stretch * dx

    # Zweeben2016 method: Find length of axes spanned by ellipse centred at origin
    x_width_ax = 2 * ((np.cos(angle_rad)/major)**2 + (np.sin(angle_rad)/minor)**2)**(-0.5)
    y_width_ax = 2 * ((np.sin(angle_rad)/major)**2 + (np.cos(angle_rad)/minor)**2)**(-0.5)

    # major_unrot = np.linalg.norm(extrema_unrot[:,0]-extrema_unrot[:,1])  # top - bottom
    # minor_unrot = np.linalg.norm(extrema_unrot[:,3]-extrema_unrot[:,2])  # right - left

    # Use parametric coords to find extrema of ellipse (ie full x and y extent)
    from numpy import arctan2, arctan, tan, cos, sin
    tx = np.arctan2(-minor*np.tan(angle_rad), major)  # p47 of labbook
    ty = np.arctan2(major, minor*np.tan(angle_rad))
    x_width_full = 2*np.abs(minor*cos(tx)*cos(angle_rad) - major*sin(tx)*sin(angle_rad))
    y_width_full = 2*np.abs(minor*cos(ty)*sin(angle_rad) + major*sin(ty)*cos(angle_rad))

    ellipticity = np.sqrt(1-(minor/major)**2) if major >= minor else np.sqrt(1-(major/minor)**2)

    ellipse = {'centre': centre, 'dR': dy, 'dtor': dy, 'angle': angle, 'radians': angle_rad,
               'width': minor, 'height': major, 'x_width_ax': x_width_ax, 'y_width_ax': y_width_ax,
               'x_width_full': x_width_full, 'y_width_full': y_width_full,
               'ellipticity':ellipticity, 'rot_extrema':extrema, 'error': error}  # ellipse in R-phi coords
    i_ellipse = {'centre': icentre, 'width': iminor, 'height': imajor, 'angle': angle, 'radians': angle_rad,
                 'elliptcicity': ellipticity, 'rot_extrema': iextrema}  # ellipse in index coords

    if verbose:
        print('centre: {}, dR: {}, dphi: {}, angle: {} deg'.format(centre, dy, dy, angle))
        # print('box:\n', box)
        # print('ellipse:\n', ellipse)
    return ellipse, i_ellipse


def contour_info(contour, image=None, stats=True, shape=True, verbose=False):
    """ Return dictionary of information about the contour
    """
    cnt = {}
    for key in ['corners','ix_corners','iy_corners','COM','area','perimeter','solidity',
                'mask','pix','ix','iy','npoints',
                'Imin','Imax','min_loc','max_loc','Imean']:
        cnt[key] = None

    cnt['corners'] = contour
    cnt['ix_corners'] = contour[:,:,0]  # x coords of contour points for plotting etc
    cnt['iy_corners'] = contour[:,:,1]  # y coords of contour points for plotting etc

    if shape:
        moments = cv2.moments(contour)
        try:
            cx = moments['m10']/moments['m00']
            cy = moments['m01']/moments['m00']
        except ZeroDivisionError:
            cx = np.mean(contour[:,:,0])
            cy = np.mean(contour[:,:,1])

        cnt['COM'] = (cx,cy)

        ## Get total extent in x and y directions of contour (rectangle not rotated)
        x, y, bounding_width, bounding_height = cv2.boundingRect(contour)
        cnt['bound_width'] = bounding_width
        cnt['bound_height'] = bounding_height

        area = cv2.contourArea(contour)
        cnt['area'] = area

        perimeter = cv2.arcLength(contour,True)
        cnt['perimeter'] = perimeter

        hull = cv2.convexHull(contour) # Area of elastic band stretched around point set
        hull_area = cv2.contourArea(hull)
        if area>0 and hull_area>0:
            solidity = float(area)/hull_area # Measure of how smooth the outer edge is
        elif area == 0:  # only one point or line?
            solidity = 1
        else:
            solidity = 0.0
        cnt['solidity'] = solidity

        if verbose: print('COM: ({}, {}), area: {}, perimeter: {}, solidity: {},'.format(cx,cy,area,perimeter,solidity))

    if image is not None:
        mask = np.zeros(image.shape,np.uint8)
        cv2.drawContours(mask,[contour],0,255,-1)
        cnt['mask']  = mask
        pixelpoints = np.transpose(np.nonzero(mask))
        pixelpoints = pixelpoints[:,::-1] # swap columns: iphi, iR -> iR, iphi
        cnt['pix'] = pixelpoints
        cnt['npoints'] = len(pixelpoints)
        cnt['ix'] = pixelpoints[:,0]
        cnt['iy'] = pixelpoints[:,1]
        if verbose:
            pass
            # print('pixelpoints:', pixelpoints)

        if stats:  # bug: max_loc giving values outside of image! x,y reversed?
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(image, mask=mask)
            cnt['Imin'] = min_val  # min and max WITHIN contour (not relative to background)
            cnt['Imax'] = max_val
            cnt['min_loc'] = min_loc
            cnt['max_loc'] = max_loc

            mean_val = np.mean(image[mask.astype(bool)])
            cnt['Imean'] = mean_val

            if verbose: print('npoints: {}, Imean: {}, Imax: {}'.format(cnt['npoints'], mean_val, max_val))

    return cnt

def plot_contour(cnt, image, x, y, ellipse=None, enlarge=False, quad_minima=None, title='Contour points', cmap='viridis'):
    ## Plot the region being analysed
    fig = plt.figure(title)
    ax = fig.add_subplot(111)
    ms = 6
    ax.imshow(image, cmap, interpolation='none', origin='lower')
    ax.plot(cnt['ix'], cnt['iy'], 'g.', ms=ms)
    ax.plot(cnt['ix_corners'], cnt['iy_corners'], 'b.', ms=ms)
    # ax.contourf(x, y, image, 200, interpolation='none', origin='lower')
    # ax.plot(cnt['x'], cnt['y'], 'g.', ms=ms)
    # ax.plot(cnt['x_corners'], cnt['y_corners'], 'b.', ms=ms)
    if cnt['max_loc'] is not None:
        ax.plot(cnt['max_loc'][0], cnt['max_loc'][1], color='orange', marker='.', ms=ms + 5)
    if ellipse is not None:
        # print('Angle changed from {:0.1f} to 0.0'.format(ellipse['angle']))
        # ellipse['angle'] = 0
        if type(ellipse) is not list:
            ellipse = [ellipse]

        for el in ellipse:
            ax.add_artist(Ellipse(xy=el['centre'], width=el['width'], height=el['height'],
                                  angle=el['angle'], alpha=1, facecolor='none', edgecolor='k', linewidth=1))
            if enlarge:
                ax.add_artist(Ellipse(xy=el['centre'],
                                      width=el['width']*enlarge, height=el['height']*enlarge,
                                      angle=el['angle'],
                                      alpha=1, facecolor='none', edgecolor='k', linewidth=0.4, linestyle='dotted'))
            # tmp
            # axI.plot(el['rot_extrema'][0,:], el['rot_extrema'][1,:], '.', color='k', ms=6)
            
            ax.plot(el['rot_extrema'][0][0:2], el['rot_extrema'][1][0:2], 'k-', marker='.', ms=ms+1)
            ax.plot(el['rot_extrema'][0][2:], el['rot_extrema'][1][2:], 'k-', marker='.', ms=ms+1)
            
            ax.plot(el['centre'][0], el['centre'][1], 'w.', ms=ms+2)
            # axI.plot(el['rot_rhs'][0], el['rot_rhs'][1], 'g.', ms=8)
            # axI.plot(el['rot_top'][0], el['rot_top'][1], 'g.', ms=8)
    if quad_minima is not None:
        for quad in quad_minima.values():
            if quad != {}:
                ax.plot(quad['ix'], quad['iy'], 'r.', ms=ms+5)
        x = np.arange(image.shape[1])
        y = x - cnt['max_loc'][0] + cnt['max_loc'][1]
        ax.plot(x,y, 'k:')
        y = -x + cnt['max_loc'][0] + cnt['max_loc'][1]
        ax.plot(x,y, 'k:')

    ax.set_xlim(-0.5, image.shape[1]-0.5)
    ax.set_ylim(-0.5, image.shape[0]-0.5)
    plt.show()

def test_ellipse_fit():
    from copy import copy
    from EllipseFitter import EllipseFitter
    im = np.zeros((10,10)).astype(np.uint8)
    # shape = 'tilt'
    # shape = 'tilt-small'
    # shape = 'horizontal'
    # shape = 'vertical'
    shape = '2-row'
    # shape = '2-col'
    # shape = '2-col+1'
    # shape = 'square'
    if shape == 'tilt':
        im[4:6, 4:8] = 1
        im[3, 3:7] = 1
        im[2:4, 3:5] = 1
    if shape == 'tilt-small':
        im[2:4, 4:8] = 1
        im[3, 3:7] = 1
        im[4:6, 3:5] = 1
    elif shape == 'horizontal':
        im[4:7, 3:8] = 1
    elif shape == 'vertical':
        im[3:8, 4:7] = 1
    elif shape == '2-row':  # fails to fit if only spans one row or column
        im[4:6, 3:8] = 1
    elif shape == '2-col':
        im[3:8, 4:6] = 1
    elif shape == '2-col+1':  # works as long as spans 3rd col/row
        im[3:8, 4:6] = 1
        im[3, 6] = 1
    elif shape == 'square':  # needs to be at elast 3x3
        im[4:7, 4:7] = 1

    im_out, cnt, hiera = cv2.findContours(copy(im), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cnt = cnt[0]
    icentre, (iminor, imajor), angle = cv2.fitEllipse(cnt)
    angle2, major2, minor2, xCenter, yCenter = EllipseFitter(im)
    centre2 = [yCenter-0.5, xCenter-0.5]
    # returns angle in degrees, zero angle for vertical major axis

    print(icentre, (iminor, imajor), angle)

    f, ax = plt.subplots()
    plt.imshow(im, interpolation='none')
    plt.scatter(cnt[:,:,0], cnt[:,:,1])
    # Plot with angle in degrees (between vertical and major axis), with major axis y value
    ax.add_artist(Ellipse(icentre, iminor, imajor, angle=angle, facecolor='none', edgecolor='b', lw=2, alpha=0.7))
    ax.add_artist(Ellipse(centre2, minor2, major2, angle=angle2, facecolor='none', edgecolor='g', lw=2, alpha=0.7))
    plt.show()


if __name__ == '__main__':
    test_ellipse_fit()
    print('\n*** Done! ***\n')