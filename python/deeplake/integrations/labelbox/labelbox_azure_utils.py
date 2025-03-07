def load_blob_file_paths_from_azure(
    storage_account_name,
    container_name,
    parent_path,
    sas_token,
    predicate=lambda x: True,
    sign_urls=True,
):
    from azure.storage.blob import BlobServiceClient

    # Construct the account URL with the SAS token
    account_url = f"https://{storage_account_name}.blob.core.windows.net"
    # Service client to connect to Azure Blob Storage using SAS token
    blob_service_client = BlobServiceClient(
        account_url=account_url, credential=sas_token
    )
    # Get a reference to the container
    container_client = blob_service_client.get_container_client(container_name)
    # List blobs in the directory
    blob_list = container_client.list_blobs(name_starts_with=parent_path)
    file_url_list = [
        f"{account_url}/{container_name}/{blob.name}"
        + (f"?{sas_token}" if sign_urls else "")
        for blob in blob_list
        if predicate(blob.name)
    ]
    return file_url_list
