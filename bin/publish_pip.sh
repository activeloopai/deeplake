rm -rf dist hub.egg-info
python3 setup.py sdist bdist_wheel
twine upload dist/*
