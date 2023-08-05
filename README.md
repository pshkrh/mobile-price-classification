# Mobile Price Classification

A simple mobile price classifier using Amazon SageMaker.

## Installation

``pip install -r requirements.txt``


## Usage

Create `env-vars.yml` and use the following structure:

```
sagemaker-role-arn: "<enter sagemaker role arn here>"
sagemaker-profile: "<aws cli profile name goes here>"
s3-bucket-name: "<s3 bucket name goes here>"
```

The `sagemaker-profile` is the name of the profile used in AWS CLI with all the relevant permissions.

Run all the cells in the Jupyter Notebook.

Don't forget to delete the endpoint once you are done with it!