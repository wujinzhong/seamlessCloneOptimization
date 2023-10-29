/* 
 * File:   SeamlessClone.cpp
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

#include "SeamlessClone.h"
#include "demoUtilities.h"

#include <iostream>
#include <fstream>

#include <vector>

#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <numpy/ndarrayobject.h>

using namespace boost::python;
using namespace std;

using namespace cv;
//using namespace cv::dpm;

extern "C"
{
//void cuda_sum(float *a, float *b, float *c, size_t size);
void* my_seamlessclone_api_imp_create_instance( int gpu_id );
Mat my_seamlessclone_api_imp_run( void* instance_ptr, void* face, void* body, void* mask, int centerX, int centerY, int gpu_id, bool bSync );
void my_seamlessclone_api_imp_destroy( void* instance_ptr );
void my_seamlessclone_api_imp_sync( void* instance_ptr );
}

static int failmsg(const char *fmt, ...)
{
    char str[1000];

    va_list ap;
    va_start(ap, fmt);
    vsnprintf(str, sizeof(str), fmt, ap);
    va_end(ap);

    PyErr_SetString(PyExc_TypeError, str);
    return 0;
}

SeamlessClone::SeamlessClone()
{
    //cuda_sum(NULL, NULL, NULL, 5);
    this->instance_ptr = NULL;
    this->bSync = false;
}

SeamlessClone::~SeamlessClone() 
{}

PyObject* SeamlessClone::loadImageInCpp_Demo(std::string imagePath)
{
    cv::Mat m = cv::imread(imagePath);

    PyObject* res = mat2py(m);

    return res;
}

void SeamlessClone::loadMatsInSeamlessClone(PyObject* oface, PyObject* obody, PyObject* omask, int centerX, int centerY, int gpu_id)
{
    py2mat(oface, this->face);
    //imwrite( "./face.jpg", this->face );

    py2mat(obody, this->body);
    //imwrite( "./body.jpg", this->body );

    py2mat(omask, this->mask);
    //imwrite( "./mask.jpg", this->mask );

    this->centerX = centerX;
    this->centerY = centerY;
    this->gpu_id = gpu_id;
    return;
}

void SeamlessClone::destroy()
{
    my_seamlessclone_api_imp_destroy( this->instance_ptr );
    this->instance_ptr = NULL;
    return;
}

void SeamlessClone::sync()
{
    my_seamlessclone_api_imp_sync( this->instance_ptr );
    return;
}

PyObject* SeamlessClone::seamlessClone()
{
    if (this->instance_ptr==NULL)
    {
        this->instance_ptr = my_seamlessclone_api_imp_create_instance( this->gpu_id );
    }
    this->blendedMat = my_seamlessclone_api_imp_run( this->instance_ptr, (void*)&this->face, (void*)&this->body, (void*)&this->mask, this->centerX, this->centerY, this->gpu_id, this->bSync );
    
    PyObject* res = mat2py(this->blendedMat);
    return res;
}

PyObject* SeamlessClone::mat2py(const cv::Mat& mat)
{
    npy_intp dims[] = {mat.rows, mat.cols, mat.channels()};

    PyObject *res = 0 ;
    if (mat.depth() == CV_8U)
        res = PyArray_SimpleNew(3, dims, NPY_UBYTE);
    else if (mat.depth() == CV_8S)
        res = PyArray_SimpleNew(3, dims, NPY_BYTE);
    else if (mat.depth() == CV_16S)
        res = PyArray_SimpleNew(3, dims, NPY_SHORT);
    else if (mat.depth() == CV_16U)
        res = PyArray_SimpleNew(3, dims, NPY_USHORT);
    else if (mat.depth() == CV_32S)
        res = PyArray_SimpleNew(3, dims, NPY_INT);
    else if (mat.depth() == CV_32F)
        res = PyArray_SimpleNew(3, dims, NPY_CFLOAT);
    else if (mat.depth() == CV_64F)
        res = PyArray_SimpleNew(3, dims, NPY_CDOUBLE);

    std::memcpy(PyArray_DATA((PyArrayObject*)res), mat.data, mat.step*mat.rows);

    return res;
}

void SeamlessClone::py2mat(const PyObject* o, cv::Mat &m)
{    
    // to avoid PyArray_Check() to crash even with valid array
    do_numpy_import();
 
    if(!o || o == Py_None)  
        return;

    if( !PyArray_Check(o) )
    {
        failmsg("Not a numpy array");
        return; 
    }

    // NPY_LONG (64 bit) is converted to CV_32S (32 bit)
    int typenum = PyArray_TYPE((PyArrayObject*) o);
    int type = typenum == NPY_UBYTE ? CV_8U : typenum == NPY_BYTE ? CV_8S :
        typenum == NPY_USHORT ? CV_16U : typenum == NPY_SHORT ? CV_16S :
        typenum == NPY_INT || typenum == NPY_LONG ? CV_32S :
        typenum == NPY_FLOAT ? CV_32F :
        typenum == NPY_DOUBLE ? CV_64F : -1;

    if( type < 0 )
    {
        failmsg("data type = %d is not supported", typenum);
        return; 
    }

    int ndims = PyArray_NDIM((PyArrayObject*) o);
    if(ndims >= CV_MAX_DIM)
    {
        failmsg("dimensionality (=%d) is too high", ndims);
        return; 
    }

    int size[CV_MAX_DIM+1];
    size_t step[CV_MAX_DIM+1], elemsize = CV_ELEM_SIZE1(type);
    const npy_intp* _sizes = PyArray_DIMS((PyArrayObject*) o);
    const npy_intp* _strides = PyArray_STRIDES((PyArrayObject*) o);
    bool transposed = false;

    for(int i = 0; i < ndims; i++)
    {
        size[i] = (int)_sizes[i];
        step[i] = (size_t)_strides[i];
    }

    if( ndims == 0 || step[ndims-1] > elemsize ) {
        size[ndims] = 1;
        step[ndims] = elemsize;
        ndims++;
    }
   
    if( ndims >= 2 && step[0] < step[1] )
    {
        std::swap(size[0], size[1]);
        std::swap(step[0], step[1]);
        transposed = true;
    }

    if( ndims == 3 && size[2] <= CV_CN_MAX && step[1] == elemsize*size[2] )
    {
        ndims--;
        type |= CV_MAKETYPE(0, size[2]);
    }

    if( ndims > 2 )
    {
        failmsg("more than 2 dimensions");
        return; 
    }

    m = cv::Mat(ndims, size, type, PyArray_DATA((PyArrayObject*) o), step);
   
    
    if( transposed )
    {
        cv::Mat tmp;
       transpose(m, tmp);
        m = tmp;
    }
}

/*void SeamlessClone::multiRotPersDet(PyObject* o, int rotationStep, std::string detectorPath, float confidenceThr, std::string outputPath, bool visualizeResults)
{
    // Initializing CSV output
    ofstream outFile;
    outFile.open (outputPath.c_str());
    outFile << "rotDeg, x_origin, y_origin, height, width, confidence\n";
 
    // Initializing Deformable Part Model Detector
    cv::Ptr<DPMDetector> detector = DPMDetector::create(vector<string>(1, detectorPath));

    //namedWindow("DPM Cascade Detection", 1);
    
    // Receiving image from Pyhton
    Mat image; 
    py2mat(o, image);

    // Generating rotated versiong of input image
    //tic();
    vector<Mat> images;      
    demo_utils::generateRotatedImages(image, images, rotationStep);

    // Looking for persons in the images
    for(int i=0;i<images.size();i++)
    {
        // Structure for storing the detector results
        vector<DPMDetector::ObjectDetection>  ds;

        printf ("Rotation: %3d - Working on image %3d out of %3d ...", rotationStep*i, i+1, (int)images.size());
        if(visualizeResults)
        {
            Mat im2sh = images[i].clone();
        
            // Detecting persons
            detector->detect(images[i], ds);
            
            // Visualizing detected persons
            demo_utils::drawBoxes(im2sh, ds, confidenceThr, Scalar(0, 0, 255));
            imshow("DPM Cascade Detection", im2sh);
            waitKey(1) ;
        }
        else
        {
            // Detecting persons
            detector->detect(images[i], ds);        
        }

        printf (" done.\n");

        // Saving results to CSV file
        int rot = rotationStep*i;
        demo_utils::saveDetections(rot, ds, outFile, confidenceThr);
     }

    outFile.close();

    return;
}*/