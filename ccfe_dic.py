#!/usr/bin/env python

""" tf_dic.py: Frequently used dictionary tools.

Detailed description:

Notes:
    @bug:

Todo:
    @todo:

Info:
    @since: 18/09/2015
"""

import numpy as np
import matplotlib as mpl
import pylab
import os
import shutil

from . import tf_debug
import pprint as pp

__author__ = 'Tom Farley'
__copyright__ = "Copyright 2015, TF Library Project"
__credits__ = []
__email__ = "farleytpm@gmail.com"
__status__ = "Development"
__version__ = "1.0.1"

# print dir(tf_debug)

db = tf_debug.Debug(0,1,1)
# Debug = 0

def key_search(dict, *args, **kwargs):
    ## Return array of deepest nested VALUES that match key criteria
    print('key_search function is incomplete!')
    # subdict = dict
    # for key, value in dict.iteritems():
    #     for arg in args:
    #         dict = subdict
    #             if arg in key:
    #                 subdict = value
    # return subdic


def key_contains( dict, *args ):
    ## Return subset of dictionary with only keys that contain the required strings
    # print dict
    # print args
    if args == () or not args:
        return dict
    else:
        subdict = {}
        for key, value in dict.items():     ## Loop through keys in main dictionary
            dprint(debug, '')
            dprint(debug, 'Checking whether "{}"" should be kept'.format(key))
            # dprint( Debug, key=key)#, value=value )
            nested_dict = dict
        
            for i, (key_nest, value_nest) in enumerate(nested_dict.items()):    ## Loop through nested dictionary to compare with current argument
                # dprint(Debug, 'Searching for keys containing "{}" in: {}'.format(arg, nested_dict.keys()))
                dprint(debug, 'Does "{}" contain "{}"? '.format( key_nest, arg) )
                # dprint( Debug, key_nest=key_nest)#, value_nest=value_nest )
                if arg in key_nest:
                    # if i!=len(args)-1:
                    dprint(debug, 'Yes!')
                    nested_dict = nested_dict[key]
                    add = 1
                    dprint( debug, add=add )
                else:
                    dprint(debug, 'No')
                    add = 0
                    dprint( debug, add=add )
                    break
            if add==1: 
                subdict[key] = value
        return subdict




# def key_contains( dict, *args ):
#     ## Return subset of dictionary with only keys that contain the required strings
#     # print dict
#     # print args
#     if args == () or not args:
#         return dict
#     else:
#         subdict = {}
#         for key, value in dict.iteritems():     ## Loop through keys in main dictionary
#             dprint(Debug, '')
#             dprint(Debug, 'Checking whether "{}"" should be kept'.format(key))
#             # dprint( Debug, key=key)#, value=value )
#             nested_dict = dict
#             for i, arg in enumerate(args):      ## Loop through arguments to check at each nested dictionary
#                 # dprint( Debug, arg=arg )
#                 for key_nest, value_nest in nested_dict.iteritems():    ## Loop through nested dictionary to compare with current argument
#                     # dprint(Debug, 'Searching for keys containing "{}" in: {}'.format(arg, nested_dict.keys()))
#                     dprint(Debug, 'Does "{}" contain "{}"? '.format( key_nest, arg) )
#                     # dprint( Debug, key_nest=key_nest)#, value_nest=value_nest )
#                     if arg in key_nest:
#                         # if i!=len(args)-1:
#                         dprint(Debug, 'Yes!')
#                         nested_dict = nested_dict[key]
#                         add = 1
#                         dprint( Debug, add=add )
#                     else:
#                         dprint(Debug, 'No')
#                         add = 0
#                         dprint( Debug, add=add )
#                         break
#                 if add == 0:
#                     break
#             if add==1: 
#                 subdict[key] = value
#         return subdict

def key_contains_all1( dict, string ):
    ## Return subset of dictionary containing keys that contain string
    subdict = {}
    for key, value in dict.items():
        if string in key:
            subdict[key] = value
    return subdict

def key_contains_sub1( dict, string ):
    ## Return subset of dictionary containing keys that contain string
    subdict = {}
    for key, value in dict.items():
        if string in key:
            subdict = value
    return subdict

def key_contains_subn( dict, *args ):
    ## Return subset of dictionary containing keys that contain string
    pprint = 0

    subdict = dict
    for i, arg in enumerate(args):
        # dprint(Debug, '\n', arg = arg, subdict = subdict)
        db('\n', arg = arg, subdict = subdict)
        if i!=len(args)-1 :
            subdict = key_contains_sub1( subdict, arg )
        else:
            subdict = key_contains_all1( subdict, arg )

    if pprint:
        pp.pprint(subdict)
    return subdict

def main():
    dict = { 'chi_one':  { 'im_1234': [1,7,8,9], 're_1234': [79,657,987] }, 
             'chi_two':  { 'im_7888': [1,7,8,9], 're_8874': [79,657,987] }, 
             'chi_bone': { 'im_gfhfg':[1,7,8,9], 're_tyrt': [79,657,987] } }
    # print key_contains(dict, 'im')
    # print
    # print key_contains(dict, 'on')
    # print
    print('OUTPUT: ', key_contains_subn(dict, 'chi' ))
    print()
    print('OUTPUT: ', key_contains_subn(dict, 'on', '12'))
    print()
    print('OUTPUT: ', key_contains_subn(dict, 'tw', '88'))
    print()
    print('OUTPUT: ', key_contains_subn(dict, 'tw', '78'))
    pass
	

if __name__ == "__main__":
    main()
