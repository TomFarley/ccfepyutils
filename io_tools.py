import configparser
import os
import logging

from nested_dict import nested_dict

logger = logging.getLogger(__name__)

def create_config_file(fn, dic):
    """Create ini config file structured around supplied dictionary"""
    config = configparser.ConfigParser()
    for key, value in dic.items():
        if isinstance(value, dict):
            config[key] = {}
            for k, v in value.items():
                config[key][k] = v
        else:
            config[key] = value
    with open(fn, 'w') as configfile:
        config.write(configfile)
    logging.info('Wrote config file to {}. Sections: {}'.format(fn, config.sections()))


def get_from_ini(config, setting, value):
    """Return value for setting from config ini file if value is None"""

if __name__ == '__main__':
    fn = os.path.expanduser('~/repos/elzar2/elzar2/default_settings/elzar_defaults.ini')
    file = nested_dict()
    file['Paths']['elzar_path'] = '~/elzar/:'
    file['Paths']['data'] = ''

    file['Movie']['source'] = 'repeat'

    file['Invertor']['type'] = 'PsfInvertor'
    file['Invertor']['settings'] = 'repeat'
    file['Invertor']['resolution'] = 'repeat'

    file['Detector']['type'] = 'QuadMinEllipseDetector'
    file['Detector']['settings'] = 'repeat'

    file['Tracker']['type'] = 'NormedVariationTracker'
    file['Tracker']['settings'] = 'repeat'

    file['Benchmarker']['type'] = 'ProximityBenchmarker'
    file['Tracker']['settings'] = 'repeat'
    # file['elzar_path']['path'] = os.path.expanduser('~/elzar/')
    create_config_file(fn, file)