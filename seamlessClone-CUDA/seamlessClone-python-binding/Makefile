PYTHON_INC=$(shell python3-config --includes)
PYTHON_LIBS=$(shell python3-config --libs)
BOOST_LIBS=-lboost_python38

export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig

all: SeamlessClone.cpp SeamlessClone.h demoUtilities.cpp demoUtilities.h
	g++ -O3 -I/usr/include/python3.8 -I/opt/conda/lib/python3.8/site-packages/numpy -shared -fPIC `pkg-config --cflags opencv` SeamlessClone.cpp demoUtilities.cpp `pkg-config --libs opencv` -o SeamlessClone.so $(PYTHON_INC) $(PYTHON_LIBS) $(BOOST_LIBS) ./seamlessclone_cuda.so
