---
seo_title: "AWS CORS Setup | Configure Secure Cross-Origin Access"
description: "Follow This Guide To Configure CORS For AWS With Deep Lake 4.0, Ensuring Secure Cross-Origin Resource Sharing And Compliance For Machine Learning Training Data."
---

# Enabling CORS in S3

In order to visualize Deep Lake datasets stored in your own S3 buckets in the [Deep Lake app](https://app.activeloop.ai/), please enable [Cross-Origin Resource Sharing (CORS)](https://en.wikipedia.org/wiki/Cross-origin\_resource\_sharing) in the buckets containing the Deep Lake dataset and any linked data, by inserting the snippet below in the CORS section of the Permissions tab for the bucket:

```
[
    {
        "AllowedHeaders": [
            "*"
        ],
        "AllowedMethods": [
            "GET",
            "HEAD"
        ],
        "AllowedOrigins": [
            "*.activeloop.ai"
        ],
        "ExposeHeaders": []
    }
] 
```

## Next Steps

- [Provisioning Role-Based Access](./provisioning.md)
