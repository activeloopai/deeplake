from setuptools import find_packages, setup

setup(
    name='al-social',
    packages=find_packages(),
    include_package_data=True,
    entry_points={
        'mkdocs.plugins': [
            'al-social = social.plugin:SocialPlugin'
        ]
    }
)
