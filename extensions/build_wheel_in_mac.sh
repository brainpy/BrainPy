# M1
#conda activate base
#python setup_mac.py bdist_wheel
#
#conda activate py3.8
#python setup_mac.py bdist_wheel
# Intel
conda activate py3.9-x86
python setup_mac.py bdist_wheel

conda activate py3.8-x86
python setup_mac.py bdist_wheel

conda activate py3.7-x86
python setup_mac.py bdist_wheel

conda activate py3.6-x86
python setup_mac.py bdist_wheel

# repair wheel
conda activate rosetta
delocate-wheel -w fixed_wheels -v dist/*.whl