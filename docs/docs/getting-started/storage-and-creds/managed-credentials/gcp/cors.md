---
seo_title: "GCP CORS Setup | Cross-Origin Access For AI Data"
description: "Set Up CORS For Google Cloud Platform With Deep Lake To Securely Handle Cross-Origin Requests, Ensuring Compliance And Easy Access To Machine Learning Data."
---

# Enabling CORS in GCP

In order to visualize Deep Lake datasets stored in your own GCP buckets in the [Deep Lake app](https://app.activeloop.ai/), please enable [Cross-Origin Resource Sharing (CORS)](https://en.wikipedia.org/wiki/Cross-origin\_resource\_sharing) in the buckets containing the Deep Lake dataset and any linked data, by inserting the snippet below in the CORS section of the Permissions tab for the bucket:

```
[
    {
      "origin": ["https://app.activeloop.ai"],
      "method": ["GET", "HEAD"],
      "responseHeader": ["*"],
      "maxAgeSeconds": 3600
    }
]
```

## Next Steps

- [Provisioning Federated Credentials](provisioning.md)