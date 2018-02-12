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
    logging.info('Wrote config file to {}'.format(fn))

def get_from_ini(config, setting, value):
    """Return value for setting from config ini file if value is None"""

if __name__ == '__main__':
    fn = os.path.expanduser('~/repos/elzar2/default_settings/elzar_defaults.ini')
    file = nested_dict()
    file['DEFAULTS']['elzar_path'] = '~/elzar/:'
    file['DEFAULTS']['machine'] = 'MAST'
    file['DEFAULTS']['camera'] = 'SA1.1'
    file['DEFAULTS']['settings'] = 'elzar_default'
    file['user_settings']['elzar_user.ini'] = '~/elzar/user_settings/elzar.ini:'
    # file['elzar_path']['path'] = os.path.expanduser('~/elzar/')
    create_config_file(fn, file)