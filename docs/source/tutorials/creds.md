## Setup Credentials

This guide will walk you through setting up your credentials so you can start using Snark: Dataflow right away.

In your project directory create `<your_credential_file_path>` file and fill the following:

```
[default]
aws_access_key_id = <your_access_key_id>
aws_secret_access_key = <your_access_key>
region: us-east-1
```

Run `usecases/setup_creds.py` to set up your Agmri and Intelinair credentials (the script is in the `dataflow` directory).

A prompt will appear asking for your AgMRI credentials.

```
> Admin_username:
> Admin_password:
> Environment: production
```

The prompt will then ask you for the IntelinAir AWS credentials for field access.

```
> Aws_access_key_id:
> Aws_secret_access_key:
```

That's it!
