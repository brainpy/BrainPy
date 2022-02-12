rm -rf build
pip uninstall brainpylib -y
python setup_cuda.py bdist_wheel
pip install dist/brainpylib-0.0.3+cuda115-cp39-cp39-linux_x86_64.whl