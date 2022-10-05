rm -rf dist deeplake.egg-info
python3 setup.py sdist bdist_wheel
twine upload dist/*