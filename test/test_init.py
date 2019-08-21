import hub
import pytest
import os

# Scenario 1. User has AWS credentials, no hub creds


def test_aws_wo_hub_creds():
    os.system('mv ~/.hub ~/.hub_arxiv')
    import hub
    x = hub.array((100, 100, 100), 'image/test:smth', dtype='uint8')
    print(x.shape)
    os.system('mv ~/.hub_arxiv ~/.hub')

# Scenario 2. User has Hub credentials, no aws creds

# Scenario 3. User has no credentials but wants to access public datasets


def test_public_access_no_creds():
    x = hub.load('imagenet')
    assert x[0].mean() == 1

# Scenario 4. User has no credentials and wants to create an array


def test_wo_aws_or_hub_creds():
    os.system('mv ~/.aws ~/.aws_arxiv')
    os.system('mv ~/.hub ~/.hub_arxiv')
    try:
        import hub
        x = hub.array((100, 100, 100), 'image/test:smth', dtype='uint8')
        print(x.shape)
    except Exception as err:
        print('pass', err)
        pass
    os.system('mv ~/.hub_arxiv ~/.hub')
    os.system('mv ~/.aws_arxiv ~/.aws')

# Scenario 5. Provided AWS credentials do not provide enough permission


if __name__ == "__main__":

    print('Running Basic Tests')
    test_aws_wo_hub_creds()
    test_wo_aws_or_hub_creds()
    test_public_access_no_creds()
