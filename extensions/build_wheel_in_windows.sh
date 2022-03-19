#mkdir "build/lib.win-amd64-3.6"
#cp build/win_dll/*  build/lib.win-amd64-3.6/
#python setup.py bdist_wheel

conda activate py37
mkdir "build/lib.win-amd64-3.7/brainpylib"
cp build/win_dll/*  build/lib.win-amd64-3.7/brainpylib
python setup.py bdist_wheel

conda activate py38
mkdir "build/lib.win-amd64-3.8/brainpylib"
cp build/win_dll/*  build/lib.win-amd64-3.8/brainpylib
python setup.py bdist_wheel

conda activate py39
mkdir "build/lib.win-amd64-3.9/brainpylib"
cp build/win_dll/*  build/lib.win-amd64-3.9/brainpylib
python setup.py bdist_wheel

conda activate py310
mkdir "build/lib.win-amd64-3.10/brainpylib"
cp build/win_dll/*  build/lib.win-amd64-3.10/brainpylib
python setup.py bdist_wheel
