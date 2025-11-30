---
seo_title: "Azure CORS Setup | Enable Secure Data Access"
description: "Set Up Azure CORS With Deep Lake To Enable Secure Cross-Origin Requests, Providing Efficient Access To Training Data For Machine Learning Projects."
---

# Enabling CORS in Azure

Cross-Origin Resource Sharing (CORS) is typically enabled by default in Azure. If that's not the case in your Azure account, please enable [CORS](https://en.wikipedia.org/wiki/Cross-origin\_resource\_sharing) in order to use the [Deep Lake app](https://app.activeloop.ai/) to visualize Deep Lake datasets stored in your own Azure storage. [CORS](https://en.wikipedia.org/wiki/Cross-origin\_resource\_sharing) should be enabled in the storage account containing the Deep Lake dataset and any linked data.

## Steps for enabling CORS in Azure

1\. Login to the Azure.

2\. Navigate to the `Storage account` with the relevant data.

3\. Open the `Resource sharing (CORS)` section on the left nav.

<figure><img src="images/Azure_CORS_1.png" alt=""><figcaption></figcaption></figure>

4\. Add the following items to the permissions.

<figure><img src="images/Azure_CORS_1.png" alt=""><figcaption></figcaption></figure>

| Allowed origins           | Allowed methods | Allowed headers |
| ------------------------- | --------------- | --------------- |
| https://app.activeloop.ai | GET, HEAD       | \*              |

## Next Steps

- [Provisioning Federated Credentials](provisioning.md)