import json
import meta.config as config
from meta.log import logger
import os

class StoreControlClient(object):
    """
    Controlling Snark Pods through rest api
    """
    def __init__(self):
        super(StoreControlClient, self).__init__()

    def is_authenticated(cls):
        return os.path.getsize(config.TOKEN_FILE_PATH) > 10

    def get_config(self):
        try:
            with open(config.TOKEN_FILE_PATH, 'r') as file:
                details = file.readlines()
                details = json.loads(''.join(details))

            with open(config.CLOUDVOLUME_PATH, 'r') as file:
                details = file.readlines()
                details = json.loads(''.join(details))
        except:
            logger.error("No AWS credentials found. Please configure credentials '> meta configure'")
            return {
                'AWS_ACCESS_KEY_ID':'',
                'AWS_SECRET_ACCESS_KEY': '',
                'bucket': ''
            }
        return details

    def save_config(self, details):
        try:
            os.mkdir('/'.join(config.TOKEN_FILE_PATH.split('/')[:-1]))
        except OSError:
            pass
        
        try:
            os.mkdir('/'.join(config.CLOUDVOLUME_PATH.split('/')[:-2]))
        except OSError:
            pass
            
        try:
            os.mkdir('/'.join(config.CLOUDVOLUME_PATH.split('/')[:-1]))
        except OSError:
            pass

        with open(config.TOKEN_FILE_PATH, 'w') as file:
            file.writelines(json.dumps(details))

        with open(config.CLOUDVOLUME_PATH, 'w') as file:
            file.writelines(json.dumps(details))

    def purge_token(self):
        if os.path.isfile(config.TOKEN_FILE_PATH):
            os.remove(config.TOKEN_FILE_PATH)
        if os.path.isfile(config.CLOUDVOLUME_PATH):
            os.remove(config.CLOUDVOLUME_PATH)