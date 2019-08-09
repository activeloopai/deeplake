rm -rf dist meta_array.egg-info
python3 setup.py sdist bdist_wheel
twine upload dist/*
