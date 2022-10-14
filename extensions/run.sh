rm -rf build
pip uninstall brainpylib -y
python setup_cuda.py bdist_wheel
pip install dist/brainpylib*