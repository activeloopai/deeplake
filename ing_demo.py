import deeplake

token = "eyJhbGciOiJIUzUxMiIsImlhdCI6MTY1NTY2NTE0NiwiZXhwIjo0ODA5MjY1MTQ2fQ.eyJpZCI6InByb2dlcmRhdiJ9.QAaCeQumZzTTDodvm9L07eIzRSu1raKVeOjMnCniNHJujIsSJ5N5qeLglo8ZvucB7AuPn7YGK0x3_jaGumnYKw"

ds = deeplake.empty("hub://progerdav/coco2", token=token, overwrite=True)
ds.add_creds_key("ing_creds1", managed=True)

with open('.env', 'r') as fh:
    vars_dict = dict(
        tuple(line.replace('export ', '').replace('\n', '').split('=', maxsplit=1))
        for line in fh.readlines() if not line.startswith('#')
    )

creds = {k.lower(): v for k, v in vars_dict.items()}

deeplake.ingest_coco(
    images_directory="s3://activeloop-ds-creation-tests/coco_source/images/",
    annotation_files=["s3://activeloop-ds-creation-tests/coco_source/instances_val_2017_tiny.json"],
    src_creds=creds,
    dest=ds,
    token=token,
    image_settings={"creds_key": "ing_creds1", "linked": True}
)
