rm -rf dist hub_array.egg-info
python3 setup.py sdist bdist_wheel
twine upload dist/*
