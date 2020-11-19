## Using Docker

### Installing Dataflow using Docker

In order to install Dataflow with Docker you need to first install the latter.

You can install Docker by following the official guide [here](https://docs.docker.com/get-docker/).

### Login to Docker

Now that you have Docker installed, it's time to login.

If you don't have an account yet, feel free to create one [here](https://hub.docker.com/signup).

You can easily login by running the following command:

```bash
docker login -u <USERNAME> -p <PASSWORD>
```

### Setup your project environment

We can now begin our project setup process. First, create a directory for your project anywhere you see fit.

In your project directory create a file called `Dockerfile`. This file will tell `Docker` how the image is built.

Add the following to the `Dockerfile`:

```docker
FROM snarkai/hub:dataflow-latest

ADD . /dataflow/project
WORKDIR /dataflow
```

Next you need to create a file for your AWS credentials. This is what the file should contain:

```
[default]
aws_access_key_id = <AWS_ACCESS_KEY>
aws_secret_access_key = <AWS_SECRET_ACCESS_KEY>
region = <REGION>
```

You can now build your `Docker` image and tag it. You need to tag the image so you can reference it later with ease. The name of the tag can be anything.

```bash
docker build -t <IMAGE_TAG> .
```

Now that the image is built, you can run the container. This command attaches your project directory and your credentials file to the container as volumes. Everything inside your project directory will be reflected inside the container.

```bash
docker run \
-v <PROJECT_ABSOLUTE_PATH>:/dataflow/project \
-v <CREDS_ABSOLUTE_PATH>:/root/.aws/credentials:ro \
-it <IMAGE_TAG> bash
```

You should now be inside the `Docker` container.

To execute a `python` file that uses `dataflow` simply run:

```python
python <PATH_TO_FILE>/<FILE>
```
