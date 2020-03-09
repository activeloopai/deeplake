import hub


def test_gs():
    bucket = hub.gs('snark_waymo_open_dataset').connect()
    txt = bucket.blob_get('temporary_blob.txt').decode()
    print(txt)


if __name__ == '__main__':
    test_gs()