/* 
 * File:   SeamlessClone.h
 * Author: mbustreo
 *
 * Created on 19 November 2015
 *
 * This class allows to move data from C++ to Python and viceversa in realtime.
 * In this way it is possible to reuse OpenCV C++ code in Python.
 * See pyttest for howTo use.
 * Personal adaptation of CvBridge - module_opencv2.cpp functions (http://wiki.ros.org/cv_bridge)
 *
 * The demo function multiRotPersDet implements a Deformable Part Model Person Detection using 
 * OpenCv3. It looks for person lying in the ground, therefore it checks if person are presents 
 * varying input image orientation. confidenceThr controls the minimum acceptable person detector confidence.
 */

#ifndef CVPYMAT_H
#define CVPYMAT_H

#include <opencv2/core/mat.hpp>
#include <boost/python.hpp>

using namespace cv;

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarrayobject.h>

#if PYTHON3
	static int do_numpy_import( )
	{
	    import_array( );
	}
#else
	static void do_numpy_import( )
	{
	    import_array( );
	}
#endif


class SeamlessClone
{
public:
    SeamlessClone();
    virtual ~SeamlessClone();

	// Converting from PyObject* to Mat and viceversa
	PyObject* 	mat2py(const cv::Mat& mat);
    void 		py2mat(const PyObject* o, cv::Mat &m);

    void 		loadMatsInSeamlessClone(PyObject* oface, PyObject* obody, PyObject* omask, int centerX, int centerY, int gpu_id);
    PyObject*   seamlessClone();
    
    // Demo functions
    PyObject* 	loadImageInCpp_Demo(std::string imagePath);	
    void 		multiRotPersDet(PyObject* o, int rotationStep, std::string detectorPath, float confidenceThr, std::string outputPath, bool visualizeResults);
    
    Mat face, body, mask, blendedMat; 
    int centerX, centerY, gpu_id;
};


BOOST_PYTHON_MODULE(SeamlessClone)
{
    Py_Initialize();
    import_array();

    // functions exposed in Python        
    boost::python::class_<SeamlessClone>("SeamlessClone", boost::python::init< >())        
        
        .def("mat2py", &SeamlessClone::mat2py)
        .def("py2mat", &SeamlessClone::py2mat)
        .def("loadMatsInSeamlessClone", &SeamlessClone::loadMatsInSeamlessClone)
        .def("seamlessClone", &SeamlessClone::seamlessClone)
        .def("loadImageInCpp_Demo", &SeamlessClone::loadImageInCpp_Demo)        
		.def("multiRotPersDet", &SeamlessClone::multiRotPersDet)
    ;
}

#endif  /* CVPYMAT_H */

    