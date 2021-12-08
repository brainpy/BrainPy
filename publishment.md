
# Git 使用

1. 切换分支

```bash
git checkout xxxx
```

2. 合并分支

```bash
git merge dev  # 把dev分支的工作成果合并到master分支上
```

3. 推送到远程仓库

```bash
git push origin master
```

```bash
find . -type d -name "__pycache__" -exec rm -rf "{}" \;

find . -type d -name "__pycache__" -exec rm -rf {} +
```


# Conda package

1, All the metadata in the conda-build recipe is specified 
in the `meta.yaml` file. See 
[LINK](https://docs.conda.io/projects/conda-build/en/latest/resources/define-metadata.html).

2, Compile 

```bash
conda build conda-recipe
```

3, Converting a package for use on all platforms
```bash
conda convert --platform all .../click-7.0-py37_0.tar.bz2 -o outputdir/
```

4, Upload

```bash
anaconda login
anaconda upload /usr/local/anaconda/conda-bld/osx-64/???.tar.bz2
```



# Pypi package

1, write `~/.pypirc` file. 

```text
[distutils]
index-servers=pypi

[pypi]
repository = https://upload.pypi.org/legacy/
username = xxxx  # pypi登录用户名
password = xxxx  # pypi登录密码
```

2, create `MANIFEST.in` file.

```text
include *.in
include *.ini
include *.rst
include *.txt
include LICENSE

global-exclude __pycache__ *.py[cod]
global-exclude *.so *.dylib
```

3, build 

```bash
python setup.py sdist
```

4, upload

```text
twine upload dist/* # 也可以单独指定 dist 文件夹中的某个版本的发布包
```



https://pypi.org/classifiers/


# GitHub

```bash

git config --global --unset-all remote.origin.proxy
```


