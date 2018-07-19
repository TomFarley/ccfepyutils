#!/usr/bin/env python

""" 
Author: T. Farley
"""
import logging, os, shutil
logger = logging.getLogger(__name__)

# TODO: get base path from config file?
settings_dir = os.path.expanduser('~/.ccfetools/settings/')

def check_ccfepyutils_dir_struct(template_settings_dirs=()):
    if not os.path.isdir(settings_dir):
        os.mkdir(settings_dir)
        os.mkdir(settings_dir+'values')
        os.mkdir(settings_dir+'log_files')
        os.mkdir(settings_dir+'hash_records')
        logger.info('Created ~/.ccfetools directory')
    copy_template_settings(template_settings_dirs=template_settings_dirs)

def copy_template_settings(template_settings_dirs=()):
    for template_settings_dir in template_settings_dirs:
        # Loop over folders in template settings dir
        for src_dir in os.walk(template_settings_dir).next()[1]:
            # If folder for this setting does not already exist, copy the template values
            new_dir = os.path.join(settings_dir, os.path.split(src_dir)[-1])
            if os.path.isdir(new_dir):
                continue
            logger.debug('Copying {} to {}'.format(src_dir, new_dir))
            shutil.copytree(os.path.join(template_settings_dir, src_dir), new_dir)
            logger.info('Copied template settings from "{}" to: {}'.format(template_settings_dir, settings_dir))