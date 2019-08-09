import click
from meta.log import logger
from meta.utils.store_control import StoreControlClient
from meta.utils.verify import Verify

@click.command()
@click.option('--username', '-u', default=None, help='Your AWS access key')
@click.option('--password', '-p', default=None, help='Your AWS secret key')
@click.option('--bucket',   '-b', default=None, help='Desired bucket name')
def configure(username, password, bucket):
    """ Logs in to Meta"""
    logger.info("Please log in using your AWS credentials.")
    if not username:
        logger.debug("Prompting for Access Key")
        username = click.prompt('AWS Access Key ID', type=str, hide_input=False)
    access_key = username.strip()

    if not password:
        logger.debug("Prompting for Secret Key")
        password = click.prompt('AWS Secret Access Key', type=str, hide_input=False)
    secret_key = password.strip()

    if not bucket:
        logger.debug("Prompting for bucket name")
        bucket = click.prompt('Bucket Name (e.g. company-name)', type=str, hide_input=False)
    bucket = bucket.strip()
    
    success, creds = Verify(access_key, secret_key).verify_aws(bucket)
    if success:
        StoreControlClient().save_config(creds)
        logger.info("Login Successful.")
    else:
        logger.error("Login error, please try again")
        
def logout():
    """ Logs out of Snark AI"""
    # TODO remove scripts
    TokenManager.purge_token()
