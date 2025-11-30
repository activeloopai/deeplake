---
seo_title: "Azure Workload Identities | Secure Role Management"
description: "Manage Azure Workload Identities For Role-Based Access To Deep Lake, Providing Secure Data Access And Ensuring Safety Across Machine Learning Workflows."
---

# Azure Workload Identities

How to authenticate using workload identities instead of user credentials.

## Authenticating Using Workload Identities Instead of User Credentials

Workload identities enable you to define a cloud workload that will have access to your Deep Lake organization without authenticating using Deep Lake user tokens. This enables users to manage and define Deep Lake permissions for jobs that many not be attributed to a specific user.

Set up a Workload Identity using the following steps:

1. Define an Azure Managed Identity in your cloud
1. Attached the Azure Managed Identity to your workload
1. Create a Deep Lake Workload Identity using the Azure Managed Identity

1. Run the workload in Azure

## Step 1: Define the workload identity in Azure

1. Navigate to Managed Identities in Azure

    <img src="../images/workload_step_1_1.png" class="screenshot">

1. Click `Create` a Managed Identity

    <img src="../images/workload_step_1_2.png" class="screenshot">

1. Select the `Subscription` and `Resource Group` containing the workload, and give the Managed Identity a `Name`. Click `Review + Create`.

    <img src="../images/workload_step_1_3.png" class="screenshot">


## Step 2: Attached the Azure Managed Identity to your workload

When creating or updating a resource that will serve as the Client running Deep Lake, assign the Managed Identity from Step 1 to this resource.

For example, in Azure Machine Learning Studio, when creating a compute instance, toggle `Assign Identity` and select the `Managed Identity` from Step 1.

<img src="../images/workload_step_2.png" class="screenshot">

## Step 3: Create a Deep Lake Workload Identity using the Azure Managed Identity

Navigate to the `Permissions` tab for your organization in the [Deep Lake App](https://app.activeloop.ai/), locate the `Workload Identities`, and select `Add`.

<img src="../images/workload_step_3_1.png" class="screenshot">

Specify a `Display Name`, `Client ID` (for the Managed Identity), and `Tenant ID`. The `Client ID` can be found in the main page for the Managed Identity, and the `Tenant ID` can be found in `Tenant Properties` in Azure. Click `Add`.

<img src="../images/workload_step_3_2.png" class="screenshot">

## Step 4: Run the workload

Specify the environmental variables below in the Deep Lake client and run other Deep APIs as normal.

<!-- test-context
```python
import os
azure_client_id = os.environ["AZURE_CLIENT_ID"]

```
-->

```python
#### THIS IS THE CLIENT_ID FOR THE COMPUTE INSTANCE
#### NOT THE MANAGED IDENTITY 
os.environ["AZURE_CLIENT_ID"] = azure_client_id

os.environ["ACTIVELOOP_AUTH_PROVIDER"] = "azure"
```

Specifying the `AZURE_CLIENT_ID` is not necessary in some environments because the correct value may automatically be set.

For a compute instance in the Azure Machine Learning Studio, the Client ID can be found in instance settings below:

<img src="../images/workload_step_4.png" class="screenshot">
