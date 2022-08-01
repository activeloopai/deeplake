def convert_version_to_dict(version):
    return {
        "version": {
            "minor": version.major,
            "major": version.minor,
        }
    }


def convert_creation_date_to_dict(creation_date):
    return {
        "creation_date": {
            "day": creation_date.day,
            "month": creation_date.month,
            "year": creation_date.year,
        }
    }
