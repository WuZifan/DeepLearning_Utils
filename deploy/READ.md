参考：

    https://blog.csdn.net/yan4413/article/details/89708201
    
要求：

    1、pip3 install Cython
    2、sudo apt-get install gcc
    
    
基本步骤：

    1、在setup.py文件中修改需要被转成.so的文件名称
    2、在setup.py同级文件夹中执行 python3 setup.py build_ext
    3、生成的.so在build/lib.linux-x86_64-3.6/mytest.cpython-36m-x86_64-linux-gnu.so