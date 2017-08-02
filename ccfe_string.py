#!/usr/bin/env python
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

""" tf_string.py: Frequently used string operations and wrappers.

Detailed description:

Notes:
    @bug:

Todo:
    @todo: Use to_precision
    @todo: Extend strnsignif to arrays of values

Info:
    @since: 18/09/2015
"""

from itertools import repeat
import datetime
import numpy as np
import matplotlib.pyplot as plt
import math
import re

## CAN import:    tf_array
## CANNOT import:
# from tf_libs import tf_array
import tf_libs.tf_array as tf_array
from tf_libs.tf_debug import Debug

__author__ = 'Tom Farley'
__copyright__ = "Copyright 2015, TF Library Project"
__credits__ = []
__email__ = "farleytpm@gmail.com"
__status__ = "Development"
__version__ = "1.0.1"

db = Debug(debug_ON=1, lines_ON = 1, plot_ON=False)
db.force_all(on=1)

def to_precision(x,p):
    """ Return a string representation of x formatted with a precision of p

    strnsignif also adds trailing 0s for correnct # of sf to functionality is same

    From: http://randlet.com/blog/python-significant-figures-format/
    Based on the webkit javascript implementation taken from here:
    https://code.google.com/p/webkit-mirror/source/browse/JavaScriptCore/kjs/number_object.cpp
    """

    x = float(x)

    if x == 0.:
        return "0." + "0"*(p-1)

    out = []

    if x < 0:
        out.append("-")
        x = -x

    e = int(math.log10(x))
    tens = math.pow(10, e - p + 1)
    n = math.floor(x/tens)

    if n < math.pow(10, p - 1):
        e = e -1
        tens = math.pow(10, e - p+1)
        n = math.floor(x / tens)

    if abs((n + 1.) * tens - x) <= abs(n * tens -x):
        n = n + 1

    if n >= math.pow(10,p):
        n = n / 10.
        e = e + 1

    m = "%.*g" % (p, n)

    if e < -2 or e >= p:
        out.append(m[0])
        if p > 1:
            out.append(".")
            out.extend(m[1:p])
        out.append('e')
        if e > 0:
            out.append("+")
        out.append(str(e))
    elif e == (p -1):
        out.append(m)
    elif e >= 0:
        out.append(m[:e+1])
        if e+1 < len(m):
            out.append(".")
            out.extend(m[e+1:])
    else:
        out.append("0.")
        out.extend(["0"]*-(e+1))
        out.append(m)
    return "".join(out)


_ftod_r = re.compile(
    r'^(-?)([0-9]*)(?:\.([0-9]*))?(?:[eE]([+-][0-9]+))?$')
def ftod(f):
    """Print a floating-point number in the format expected by PDF:
    as short as possible, no exponential notation.
    Taken from: https://stackoverflow.com/questions/8345795/force-python-to-not-output-a-float-in-standard-form-scientific-notation-expo/30754233#30754233"""
    s = str(f)
    m = _ftod_r.match(s)
    if not m:
        raise RuntimeError("unexpected floating point number format: {!a}"
                           .format(s))
    sign = m.group(1)
    intpart = m.group(2)
    fractpart = m.group(3)
    exponent = m.group(4)
    if ((intpart is None or intpart == '') and
        (fractpart is None or fractpart == '')):
        raise RuntimeError("unexpected floating point number format: {}"
                           .format(s))

    # strip leading and trailing zeros
    if intpart is None: intpart = ''
    # else: intpart = intpart.lstrip('0') # keep zero before dp
    if fractpart is None: fractpart = ''
    # else: fractpart = fractpart.rstrip('0') # keep sig fig zeros

    if intpart == '' and fractpart == '':
        # zero or negative zero; negative zero is not useful in PDF
        # we can ignore the exponent in this case
        return '0'

    # convert exponent to a decimal point shift
    elif exponent is not None:
        exponent = int(exponent)
        exponent += len(intpart)
        digits = intpart + fractpart
        if exponent <= 0:
            return sign + '.' + '0'*(-exponent) + digits
        elif exponent >= len(digits):
            return sign + digits + '0'*(exponent - len(digits))
        else:
            return sign + digits[:exponent] + '.' + digits[exponent:]

    # no exponent, just reassemble the number
    elif fractpart == '':
        return sign + intpart # no need for trailing dot
    else:
        return sign + intpart + '.' + fractpart

def _char_before_dp(x):
    """ digits before decimal point """
    x = np.abs(x)
    n = int(np.floor(np.log10(x)))
    if x >= 1:
        return n+1
    elif x < 1:
        return 0

def _lead_zeros_after_dp(x):
    """ Number of zero characters after dp and before 1st sf """
    x = np.abs(x)
    n = int(np.floor(np.log10(x)))
    # db(n=n)
    if x < 1:
        return abs(n+1)
    elif x >= 1:
        return 0

def strnsignif(x, nsf=3, scientific=False, _verbatim=False, standard_form=False):
    """ Return string format of number to supplied no. of significant figures
    Ideas from: http://stackoverflow.com/questions/3410976/how-to-round-a-number-to-significant-figures-in-python
    Note: Does not round up for: strnsignif(-55.55,nsf=3 ==> -55.5  -- Not sure why...?"""
    assert nsf >= 0, 'Negative number of significant figures supplied'
    nsf = int(nsf)

    ## For zero sig fig just return 0.0 (needed for str_err)
    if nsf == 0:
        x=0
        nsf = 1

    if _verbatim: print('_before_dp: ',_char_before_dp(x))

    if not standard_form:
        ## %s %g rounds, but will give standard form notation (e-7 etc) in some cases
        ## therefore use ftod to always get normal form
        format_str = '%.'+str(nsf)+'g' # Use 5g to do rounding (very occasional small bugs?)
        strn = '%s' % float(format_str % x)
        strn = ftod(float(strn)) # make sure in 'normal' form
    else:
        format_str = '%.'+str(nsf)+'g' # Use 5g to do rounding (very occasional small bugs?)
        strn = '%e' % float(format_str % x)

    ## Remove -ve sign and add it at the end to simplify counting no of characters in string
    if x<0: strn = strn[1:]
    if standard_form:
        exp_r = re.compile(r'^(-?[0-9]*\.[0-9]*)([eE]([+-][0-9]+))?$')
        m = exp_r.match(strn)
        strn = m.group(1)
        exponent = m.group(2)
        if strn == '' or exponent == '':
            raise RuntimeError("unexpected standard form number format: {}"
                           .format(m.string))
        ## Remove false trailing 0s giving false impression of precision
        if len(strn)-1 > nsf: # -1 for dp
            print(strn)
            strn = strn[0:nsf+1] # +1 for dp
            print(strn)

    ## Add trailing 0s of precision if they are missing (only used for %s above, not %f)
    if (len(strn)-1) < nsf:
        strn += '0'*(nsf-(len(strn)-1))
    ## %s always formats with a decimal point ie .0
    ## Remove false trailing .0 giving false impression of precision if %s used above
    elif _char_before_dp(x) >= nsf: # important if %s used above
        strn = strn.split(sep='.')[0]

    ## Add -ve sign if removed
    if x<0: strn = '-'+strn
    if standard_form: strn += exponent

    return strn


def str_err(value, error, nsf = 1, max_lead_zero = 6, latex=False, separate=False):
    """ Given a value and error, finds the position of the first sifnificant
        digit of the error and returns a string contianing both value and
        error to the same number of sifnificant digits separated by a +/-
        Inputs:
        c      value 		value to be rounded
            error 		error to determine rounding
            nsf			number significant figures
            latex		use latex \pm symbol
            separated 	return separate value and error strings (overrides '\latex')
    """

    ## Check for sensible inputs
    assert error > 0, "str_err received negative error"
    assert not (error == 0), "str_err received zero valued error"
    assert nsf >= 1, "str_err received nsf less than 1"

    ## Find number of additional digits needed in value than error assuming 1sf required
    nsf_val = np.floor(np.log10(abs(value))) - np.floor(np.log10(error))

    ## Add additional sf as requested
    nsf_val += (nsf)

    ## If last required sf in error is at least an order of magnitude greater than the error display
    ## no additional digits (not negative additional digits)
    if nsf_val < 0: nsf_val = 0
    
    ## Keyword dependent output format
    if   (latex == True):  # for single string containing latex \pm
        pm = r" $\pm$ "
    else:  # for  single string containing +/-
        pm = r" +/- "

    ## If displaying in normal form requires too many digits (zeros after dp) use exponential format
    if (_lead_zeros_after_dp(error) < max_lead_zero) or separate:
        value_str = strnsignif(value, nsf_val)
        err_str = strnsignif(error, nsf)
        if (separate == True): 	# Return value and error in separate strings
            return value_str, err_str    
        else:
            comb_str = value_str + pm + err_str
            return comb_str
    else: # Too many dp: display as (4.12+/-0.03)e-7
        pow = int(np.floor(np.log10(np.abs(value))))
        shift = 10**(-(pow))
        value_str = strnsignif(value*shift, nsf_val)
        err_str = strnsignif(error*shift, nsf)
        str_exp = '('+value_str+pm+err_str+')e%d' % pow
        return str_exp


def _str_err_old(value, error, nsf = 1, latex=False, separate=False):
	""" Given a value and error, finds the position of the first sifnificant 
		digit of the error and returns a string contianing both value and
		error to the same number of sifnificant digits separated by a +/-
		Inputs:
			value 		value to be rounded
			error 		error to determine rounding
			nsf			number significant figures
			latex		use latex \pm symbol
			separated 	return separate value and error strings (over-rides '\latex')
	"""
	from numpy import floor, log10, sign
	## REMEMBER: Look at IDL routine for how to extend!
	## NOTE: Pass keywords by reference doesn't work for value_str=value_str, err_str=err_str
	## TODO: implement
	# In [79]: '{:.{s}f}'.format(1.234, s = 2)
	# Out[# 79]: '1.23'

	## Default format code when error cannot be used to determine sig figs
	fmt_dflt = ':0.3g'

	## Check input is sensible - could use assert
	if error < 0:
		print("WARNING: error passed to str_err is -ve")
		fmt_str = '{'+fmt_dflt+'} +/- (-ve err!)'
		return fmt_str.format(value)
	elif error == 0:
		print("WARNING: error passed to str_err is 0.0")
		fmt_str = '{'+fmt_dflt+'} +/- 0.0 (!)'
		return fmt_str.format(value)

	## Find first significant figure of error
	if error < 1.0:

		## Put number of significant digits to display into a string
		sf_err = abs(floor(log10(error)))
		try:
			sf_str = str(int(sf_err))
		except OverflowError as detail:
			## err is probably floating infinity
			return '{} +/- inf ({})'.format(value, detail)

		## Create separate value and error strings
		format_string = r"{:0."+sf_str+"f}"
		value_str 	= format_string.format(value)
		err_str 	= format_string.format(error)

		## Return keyword depended output format
		if   (latex == True)  and (separate == False):  # Return single string contianing \pm
			comb_str = value_str + r" $\pm$ " + err_str
			return comb_str
		elif (latex == False) and (separate == False):  # Return single string contianing +/-
			comb_str = value_str + r" +/- " + err_str
			return comb_str
		elif (separate == True): 	# Return value and error in separate strings
			return value_str, err_str

	elif error >= 1.0:
		## NEED  TO FIX CASE WHERE ERROR IS BIGGER THAN VALUE!


		## Put number of significant digits to display into a string
		# sign_value = sign(value)
		sf_err = floor(log10(error))+1
		sf_value = floor(log10(abs(value)))+1
		try:
			sf_str = str(int(sf_err))
		except OverflowError as detail:
			## err is probably floating infinity
			return '{} +/- inf ({})'.format(value, detail)

		## Create separate value and error strings
		format_string = r"{:0.0f}"
		value_str 	= format_string.format(round(value*10**-(sf_value-sf_err))*10**(sf_value-sf_err))
		err_str 	= format_string.format(round(error*10**-(sf_err-1))*10**(sf_err-1))

		## Return keyword depended output format
		if   (latex == True)  and (separate == False):  # Return single string contianing \pm
			comb_str = value_str + r" $\pm$ " + err_str
			return comb_str
		elif (latex == False) and (separate == False):  # Return single string contianing +/-
			comb_str = value_str + r" +/- " + err_str
			return comb_str
		elif (separate == True): 	# Return value and error in separate strings
			return value_str, err_str
	else: 
		return "(str_err cannot handel this value-error combination yet)"
		
			
			# if latex == True:
			# 	format_string = r"{:0."+sf_str+"f} $\pm$ {:0."+sf_str+"f}"
			# 	str_err = format_string.format(value, error)
			# else: 
			# 	format_string =  "{:0."+sf_str+"f} +/- {:0."+sf_str+"f}"
			# 	str_err = format_string.format(value, error)


def str_popt(popt, pcov, check=[0,1,2], strings=None, 
			units = None, latex=False):
	""" Print fit parameters from sciipy.curvefit 

	"""
	if units == None:
		units = list(repeat('', len(check)))
	if strings == None:
		strings = ['popt['+str(i)+']' for i in check]
	nl = list(repeat('\n', len(check)-1)) + [''] # new lines
	string = ''
	for i in check:
		string += (strings[i] + '\t = ' + str_err(popt[i], np.sqrt(pcov[i][i]), latex=latex) + ' ' +
			units[i] + nl[i])
	return string

def scs(str1, str2=False, append = False, separator='/'):
    """ String character separator: 
    -If only str1 is supplied, appends a separator (default:'/') to the string 
    if not already pressent.  
    -If both str1 and str2 supplied, the two strings are concatenated such that 
    they are separated by one separator characher (default:'/')
    *If append is true only the second argument (str2) is returned w or w/o the
    required separator """
    # print str1, str2
    if str2: # combine two path strings
        ## Do not add or remove any separators if one string is empty - ideally remove separators at join...
        if (str1 == '') or (str2 == ''): ## NOTE: This may not work properly in str_comb! ******
            return str1+str2
        end = str1.endswith(separator)      # True if ends in '/'
        start = str2.startswith(separator)  # True if starts with '/'
        if (not end) and (not start): # add slash between path elements
            if append: return separator+str2
            else:      return str1+separator+str2
        elif end and start: # remove slash between path elements
            if append: return str2 #[1:]
            else:      return str1+str2[1:]
        else: # Either begining or end already has a slash
            if append: return str2
            else:      return str1+str2

    else: # check end of one string
        if ~str1.endswith(separator):
            return str1+separator
        else:
            return str1
        
def comb_str(*args, **kwargs):
    """ Combine strings into one string, each element separated by separator character (dflt:'/') """
    separator = kwargs.pop("separator", '/') # Need to pop to prevent separator becoming a positional argument!
    assert (len(args)>1 and type(args[0])==str), 'comb_str requires at least two string arguements'
    comb_str = args[0]
    for str1, str2 in zip(args, args[1:]): # create tuples of adjacent strings
        # db(str1=str1, str2=str2)
        comb_str += scs(str1, str2=str2, append=True, separator=separator)
    return comb_str

def sort_dates(dates_in, format = "%Y-%m-%d", reverse = True):
    """Sort dates chronologically
    Note: This will add padded zeros if they are nor already present therefore output not always equal to input
    basic method from: http://stackoverflow.com/questions/5166842/sort-dates-in-python-array """
    dates = [datetime.datetime.strptime(d, format) for d in dates_in]
    indices = tf_array.argsort(dates, reverse=reverse)
    dates.sort(reverse = reverse)
    sorted = [datetime.datetime.strftime(d, format) for d in dates]
    return sorted, indices


def str_cols(*args, **kwargs):
    """ Return string containing properly aligned columns of strings. Each supplied list becomes a column in the
    output string
    kwargs:
    sep:    str/tuple     separator between each column
    """
    sep = kwargs.pop("sep", ' ') # Need to pop to prevent becoming a positional argument!
    append = kwargs.pop("append", '') # Need to pop to prevent becoming a positional argument!

    if len(args) == 1:
        assert [type(a) == True for a in args], "Cannot combine non strings"
        return '\n'.join(args[0])

    if isinstance(sep, (tuple,list)) and (len(sep) == len(args)-1): # list of column specific separators
        sep_out = sep[1:]
        sep = sep[0]
    elif type(sep) == str: # if only one sepatator supplied use it between each columns
        sep_out = sep
    else:
        raise('sep must be a string or a tuple of strings')

    if type(append) == str: # use the same appendor for all but last column
        append = list(repeat(append,len(args)-1))
        append.append('')
        db(app=append, rep=repeat(append,len(args)-2))
        app1 = append[0]
        app2 = append[1]
        append = append[2:]
        db(append=append, args=args)
    elif isinstance(append, (tuple,list)) and (len(append) == len(args)-0): # individual appendors supplied for each
        # column
        app1 = append[0]
        app2 = append[1]
        append = append[2:]
    elif isinstance(append, (tuple,list)) and (len(append) == len(args)-1): # nested call - 1st col already appended
        app1 = ''
        app2 = append[0]
        append = append[1:]
    else:
        db(append=append, args=args)
        raise('append must be a string or a tuple of strings')

    # extract first two args - these two colums will first be combined into one column string
    list1 = args[0] # can't use pop as tuple immutable
    list2 = args[1]
    args = args[2:] # Remove them from args for next pass

    assert len(list1) == len(list2), "inconsistent number of strings in list1 and list2"

    width1 = np.max([len(s) for s in list1])+len(app1) # length of longest first string
    width2 = np.max([len(s) for s in list2])+len(app2)
    list1_out = []
    list2_out = []
    str_out = []
    for s1, s2 in zip(list1, list2):
        # str_out += '{2:{0}}{3:{1}}\n'.format(width1, width2, name+sep, value)
        s1 = '{1:{0}}'.format(width1, s1+app1)
        s2 = '{1:{0}}'.format(width2, s2+app2)
        list1_out.append(s1)
        list2_out.append(s2)
        str_out.append( sep.join([s1, s2]) )

    return str_cols(str_out, sep = sep_out, append = append, *args)


def str_name_value(names, values, errors=None, dp=1, sep=':'):
    """ Return string containing properly formatted names and their values
    Makes sure that decimal points line up in values
    eg
    value1:    23.5
    val2:   34654.5
    ^      ^     ^ lined up using as little horizontal space as possible (better than tabs)
    """
    assert len(names) == len(values), "inconsistent number of name strings and values"
    pad1 = np.max([len(s) for s in names])+1+len(sep) # length of longest name
    pad2 = str(int(np.floor(np.log10(np.max(values))))+3) # pad spaces before dp to align dp
    str_out = ''
    for name, value in zip(names, values):
        str_out += '{2:{0}}{3:{1}.1f}\n'.format(pad1, pad2, name+sep, value)
    str_out = str_out.rstrip()

    return str_out

def str_moment(arr, ax=None, xy=(0.04, 0.7)):
    """ Return string containing statistical information about the contents of arr
    ax: matplotlib axis to annotate stats to """

    mean = np.mean(arr)
    # mode = sp.stats.mode(arr)
    min = np.min(arr)
    max = np.max(arr)
    range = max-min
    stdev = np.std(arr)

    values = [mean, min, max, range, stdev]
    names = ['Mean', 'Min', 'Max', 'Range', 'Stdev']

    stats_str = str_name_value(names, values)

    if ax:
        plt.annotate(stats_str, xy=xy, xycoords='axes fraction', family='monospace', bbox=dict(boxstyle="round", alpha= 0.5, fc="0.8"))

    return stats_str

def str2tex(string):
    """Convert common mathematical strings to latex friendly strings for plotting"""
    def insert(string, char, i):
        """ insert character after index """
        if i == -1:  # effecively last character: -1-1
            return string + char
        # elif i == 0:  # effecively first character: 0-1
        #     return char + string
        else:
            return string[0:i+1] + char + string[i+1:]

    first = 1000
    last = -1
    close_chars = [' ', '=', '^', '_']

    for char_enclose in ['_', '^']:
        start = 0
        while string.find(char_enclose, start+1, len(string)) != -1:
            if string.find(char_enclose, start+1, len(string)) == string.find(char_enclose+'{', start+1, len(string)):
                continue
            start = string.find(char_enclose, start, len(string))
            if start < first:
                first = np.max(start-1, 0)
            string = insert(string, '{', start)
            found = False
            end_chars = deepcopy(close_chars)
            end_chars.remove(char_enclose)
            for char in end_chars:
                if string.find(char, start, len(string)) != -1:
                    end = string.find(char, start, -1)
                    if end > last:
                        last = end -1
                    string = insert(string, '}', end-1)
                    found = True
                    break
            if found is False:
                string += '}'
                last = 1000

    for char_escape in ['%', 'Psi', 'phi', 'theta', 'sigma']:
        start = 0
        while string.find(char_escape, start, len(string)) != -1:
            if string.find(char_escape, start+1, len(string)) == string.find('\\'+char_escape, start+1, len(string))-1:
                continue
            start = string.find(char_escape, start, len(string))
            if start < first:
                first = np.max(start-1, 0)
            string = insert(string, '\\', start-1)
            start += 2
            if start+len(char_escape) > last:
                last = start+len(char_escape)

    if first!= 1000:
        first = string.rfind(' ', 0, first+1)
        if first == -1:
            first = 0
        if last == 1000:
            last = len(string)
        string = insert(string, '$', first)
        string = insert(string, '$', last)

    string = re.sub('nounits', 'N/A', string)

    return string

def str_replace(string, pattern, replacement='', unique=True):
    """ Find occurence of pattern in string and replace with replacement: UNFINISHED"""  # TODO: UNFINISHED
    pat = re.compile(pattern)
    m = pat.match(string)
    print(m.expand())

    ## Check how many times pattern was found
    if m == None:
        print('No matches for %s in %s' % (pattern, string))
    if len(m.groups()) > 1:
        print('Multiple (%d) matches for %s in %s' % (len(m.groups), pattern, string))

    # re.s
    #
    # strn = m.group(1)


def test_print():
	print("Here is a string")

if __name__ == "__main__":
    print('*** tf_string.py demo ***')
    x = np.linspace(0,10,100)
    y = np.linspace(10,30,100)

    a = [1,2,3]
    b = [a,a,a]
    print(str_popt(a,b))


    pass
