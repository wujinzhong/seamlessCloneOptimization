PYTHON_INC=$(shell python2-config --includes)
PYTHON_LIBS=$(shell python2-config --libs)
BOOST_LIBS=-lboost_python

export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig

all: SeamlessClone.cpp SeamlessClone.h demoUtilities.cpp demoUtilities.h
	g++ -O3 -I/usr/include/python2.7 -I/usr/local/lib/python2.7/dist-packages/numpy/core/include/ -shared -fPIC `pkg-config --cflags opencv` SeamlessClone.cpp demoUtilities.cpp `pkg-config --libs opencv` -o SeamlessClone.so $(PYTHON_INC) $(PYTHON_LIBS) $(BOOST_LIBS) ../seamlessclone_cuda.so
