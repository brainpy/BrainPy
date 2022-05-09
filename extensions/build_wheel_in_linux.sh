#docker run -ti -v $(pwd):/io quay.io/pypa/manylinux2010_x86_64 /bin/bash
#cd /io/

#docker run -ti -v $(pwd):/io quay.io/pypa/manylinux2014_x86_64 /bin/bash
#cd /io/


version=0.0.5
linux_version=manylinux2010_x86_64

linux_version=manylinux2014_x86_64


# py36
/opt/python/cp36-cp36m/bin/python -m pip install pybind11 numpy  jax jaxlib
/opt/python/cp36-cp36m/bin/python setup.py bdist_wheel
auditwheel repair --plat $linux_version dist/brainpylib-$version-cp36-cp36m-linux_x86_64.whl

# py37
/opt/python/cp37-cp37m/bin/python -m pip install pybind11 numpy  jax jaxlib
/opt/python/cp37-cp37m/bin/python setup.py bdist_wheel
auditwheel repair --plat $linux_version dist/brainpylib-$version-cp37-cp37m-linux_x86_64.whl

# py38
/opt/python/cp38-cp38/bin/python -m pip install pybind11 numpy  jax jaxlib scipy==1.7.1
/opt/python/cp38-cp38/bin/python setup.py bdist_wheel
auditwheel repair --plat $linux_version dist/brainpylib-$version-cp38-cp38-linux_x86_64.whl

# py39
/opt/python/cp39-cp39/bin/python -m pip install pybind11 numpy  jax jaxlib scipy==1.7.1
/opt/python/cp39-cp39/bin/python setup.py bdist_wheel
auditwheel repair --plat $linux_version dist/brainpylib-$version-cp39-cp39-linux_x86_64.whl


# py310
/opt/python/cp310-cp310/bin/python -m pip install pybind11 numpy  jax jaxlib scipy==1.7.2
/opt/python/cp310-cp310/bin/python setup.py bdist_wheel
auditwheel repair --plat $linux_version dist/brainpylib-$version-cp310-cp310-linux_x86_64.whl
