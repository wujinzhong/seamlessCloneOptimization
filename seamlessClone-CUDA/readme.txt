first: 参考这里安装依赖库，https://github.com/wujinzhong/seamlessCloneOptimization。
goto file seamlessClone_imp.h and set "#define SCDEBUG true"

to build executive:
>./seamlessClone_main.sh
#the output is ./output/ucRGB_Output.bmp

>to build .so for python wrapping.
>./make
>cd ./seamlessClone-python-binding
>make && python SeamlessClone_test.py
#the output is at ./ seamlessClone-python-binding/output/blendedMat_0/1.jpg

While profiling performance, set "#define SCDEBUG false" and do it again.

Or, you can perform all these steps just by run ./seamlessClone_main.sh




