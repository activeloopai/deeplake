## Local Development
Developing for Hub is as you expect for most Python packages.

### Prepare Environment
It is highly recommended you use an environment manager when developing for any project, Hub is no exception. For more information on how conda works, please visit [their website](https://docs.conda.io/projects/conda/en/latest/user-guide/install/). You can create a new conda environment with the following (replace [version] with your version, must be greater than python3.6)
```bash
conda create -n hub python=[version]
```

Activate the environment
```bash
conda activate hub
```

**Note: If you don't like using conda, you can use [venv](https://docs.python.org/3/library/venv.html) instead.**

### Install Package
When installing the package, make sure you use the `-e` flag, this allows you to `import hub` from another project (with the same conda/venv environment) & make changes directly to the source with immediate updates.
```bash
git clone https://github.com/activeloopai/Hub
cd Hub
pip install -e .
```

### Example
Make sure that, after following the preceeding steps, you can now run this python script

```python
from hub import Dataset

mnist = Dataset("activeloop/mnist")  # loading the MNIST data lazily
# saving time with *compute* to retrieve just the necessary data
mnist["image"][0:1000].compute()
```

If you receive the error

```bash
botocore.exceptions.ClientError: An error occurred (InvalidAccessKeyId) when calling the GetObject operation: The AWS Access Key Id you provided does not exist in our records.
```

Double check that you included `aws_session_token` within your `~/.aws/credentials` profile. More information regarding AWS credentials can be found [here](https://docs.aws.amazon.com/sdk-for-php/v3/developer-guide/guide_credentials_profiles.html).

If you are having a related problem, please read/comment on the [reference issue](https://github.com/activeloopai/Hub/issues/633) or you can setup with docker.

### Documentation
Using the same development environment, you can locally build & contribute to the documentation (what you are on right now)

#### Setup:

```bash
cd docs
pip install -r requirements.txt
```

#### Build:

```bash
make html
```

#### View:

Navigate to `Hub/docs/build` & open the `index.html` file in a browser window.
