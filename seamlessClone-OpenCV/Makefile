CXX ?= g++

CXXFLAGS += -c -Wall $(shell pkg-config --cflags opencv)
LDFLAGS += $(shell pkg-config --libs --static opencv)

all: seamlessClone_OpenCV
all: jpg2yaml


seamlessClone_OpenCV: seamlessClone_OpenCV.o; $(CXX) $< -o $@ $(LDFLAGS)
jpg2yaml: jpg2yaml.o; $(CXX) $< -o $@ $(LDFLAGS)

%.o: %.cpp; $(CXX) $< -o $@ $(CXXFLAGS)

clean: ; rm -f seamlessClone_OpenCV.o seamlessClone_OpenCV jpg2yaml.o jpg2yaml
