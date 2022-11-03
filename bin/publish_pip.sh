rm -rf dist deeplake.egg-info
python3 setup.py sdist
twine upload dist/*
