import json
import hub.config as config
from hub.log import logger
import os
from hub.utils.verify import Verify
import base64
from hub.utils.base import gen
from hub.exceptions import NotAuthorized


class StoreControlClient(object):
    """
    Controlling Snark Pods through rest api
    """

    def __init__(self):
        super(StoreControlClient, self).__init__()

    def is_authenticated(cls):
        return os.path.getsize(config.TOKEN_FILE_PATH) > 10

    @classmethod
    def lookup_aws_creds(self):
        with open(config.AWSCRED_PATH, 'r') as file:
            creds = file.readlines()

            for line in creds:
                line = ''.join(line.split())
                if 'aws_access_key_id' in line:
                    access_key = line.split('=')[1]
                if 'aws_secret_access_key' in line:
                    secret_key = line.split('=')[1]

        success, creds = Verify(access_key, secret_key).verify_aws(bucket=None)
        if success:
            self.save_config(creds)
        else:
            raise Exception(
                'Error in lookup AWS credentials or in having AWS S3 bucket access.')
        return creds

    @classmethod
    def get_config(self, public=False):
        try:
            if os.path.exists(config.TOKEN_FILE_PATH):
                with open(config.TOKEN_FILE_PATH, 'r') as file:
                    details = file.readlines()
                    details = json.loads(''.join(details))
                    return details
            else:
                return gen()

            # Disabled AWS credential lookup
            if False and os.path.exists(config.AWSCRED_PATH):
                return self.lookup_aws_creds()
            raise Exception
        except Exception as err:
            raise NotAuthorized(
                "No Hub or AWS credentials found. Please provide credentials by running '> hub configure'")
            return {
                'AWS_ACCESS_KEY_ID': '',
                'AWS_SECRET_ACCESS_KEY': '',
                'BUCKET': ''
            }
        return details

    @classmethod
    def save_config(self, details):
        try:
            os.mkdir('/'.join(config.TOKEN_FILE_PATH.split('/')[:-1]))
        except OSError:
            pass

        with open(config.TOKEN_FILE_PATH, 'w') as file:
            file.writelines(json.dumps(details))

    def purge_token(self):
        if os.path.isfile(config.TOKEN_FILE_PATH):
            os.remove(config.TOKEN_FILE_PATH)
