/*
 * Copyright 1993-2018 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

// This sample needs at least CUDA 10.0. It demonstrates usages of the nvJPEG
// library nvJPEG supports single and multiple image(batched) decode. Multiple
// images can be decoded using the API for batch mode

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/photo.hpp"
#include<iostream>
#include<string>
#include <cuda_runtime_api.h>
#include "nvJPEG_helper.hxx"
#include "seamlessClone.cpp"

using namespace std;
using namespace cv;

struct seamlessClone_params_t {
  int dev;
  cudaStream_t stream;
};

int load_inputs( Mat& dst, Mat& patch, Mat& mask,
 string dst_file, string patch_file, string mask_file)
 {
    dst = readFromYaml(dst_file.c_str());
    patch = readFromYaml(patch_file.c_str());
    mask = readFromYaml(mask_file.c_str());

    assert( patch.rows==mask.rows &&
     patch.rows==mask.rows &&
     patch.channels()==3 &&
     mask.channels()==1 );

     return 0;
 }

// parse parameters
int findParamIndex(const char **argv, int argc, const char *parm) {
  int count = 0;
  int index = -1;

  for (int i = 0; i < argc; i++) {
    if (strncmp(argv[i], parm, 100) == 0) {
      index = i;
      count++;
    }
  }

  if (count == 0 || count == 1) {
    return index;
  } else {
    std::cout << "Error, parameter " << parm
              << " has been specified more than once, exiting\n"
              << std::endl;
    return -1;
  }

  return -1;
}

void writeSCImage( const char* writeto, SCImage* img, float scale )
{
	if( img->mDType==SCImageDataType_UC )
	{
		writeBMP( writeto,
		  	img->mData + 0*img->mHeight*img->pitch(),
                  	img->pitch(),
                  	img->mData + 1*img->mHeight*img->pitch(),
                  	img->pitch(),
                  	img->mData + 2*img->mHeight*img->pitch(),
                  	img->pitch(),
                  	img->mWidth,
                  	img->mHeight);
	}
	else if(img->mDType==SCImageDataType_Float)
	{
		SCImage tmp;
		tmp.resize( img->mWidth, img->mHeight, img->mChannel, SCImageDataType_UC, SCImageOrder_Row );
		MyNPP myNpp;
		myNpp.convertFloat2UC( &tmp, img, scale );
		writeSCImage( writeto, &tmp, -1.0f );
		tmp.destroy();
	}
	return;
}

inline int findSampleId(int argc, const char **argv) {
    int sampleId = 6;
    if (checkCmdLineFlag(argc, argv, "sample")) {
        sampleId = getCmdLineArgumentInt(argc, argv, "sample=");
    }
    return sampleId;
}

int main(int argc, const char *argv[]) {
  int pidx;
  if ((pidx = findParamIndex(argv, argc, "-h")) != -1 ||
      (pidx = findParamIndex(argv, argc, "--help")) != -1) {
    std::cout << "Usage: " << argv[0]
              << " [-device=device_id]"
              << std::endl;
    return EXIT_SUCCESS;
  }

  seamlessClone_params_t params;
  params.dev = 0;
  params.dev = findCudaDevice(argc, argv);
  int sampleId = findSampleId(argc, argv);
  cudaDeviceProp props;
  checkCudaErrors(cudaGetDeviceProperties(&props, params.dev));
  printf("Using GPU %d (%s, %d SMs, %d th/SM max, CC %d.%d, ECC %s)\n",
         params.dev, props.name, props.multiProcessorCount,
         props.maxThreadsPerMultiProcessor, props.major, props.minor,
         props.ECCEnabled ? "on" : "off");

  // stream for decoding
  checkCudaErrors(
      cudaStreamCreateWithFlags(&params.stream, cudaStreamNonBlocking));

  SeamlessClone *seamlessClone = new SeamlessClone( params.stream, 
		  props.multiProcessorCount, 
		  props.maxThreadsPerMultiProcessor );
  cudaEvent_t start, stop;
  checkCudaErrors( cudaEventCreate(&start) );
  checkCudaErrors( cudaEventCreate(&stop) );

#if SCDEBUG
  const int LOOPS = 1;
#else
  const int LOOPS = 50;
#endif
  int centerX = 800, centerY = 100;

  // align API with OpenCV //////////////////////////////////////////////////////////////
  Mat destMat, patchMat, maskMat, _blend;
  vector<string> ymls = {
  "./images/dst.yml",
  "./images/src_154x100.yml",
  "./images/src_mask_154x100.yml",
  
  "./images/dst.yml",
  "./images/src.yml",
  "./images/src_mask_rect255.yml",

  "./images/dst.yml",
  "./images/src_592x592.yml",
  "./images/src_mask_592x592.yml",

  "./images/dst_4800x2694.yml",
  "./images/src_2400x1552.yml",
  "./images/src_mask_2400x1552.yml",
  };
  int file_idx = sampleId;
  if( file_idx!=3 ) centerY = 450;
  if( file_idx==9 ) { centerX = 2400; centerY = 1347; }
  if( load_inputs(destMat, patchMat, maskMat,
                  ymls[file_idx+0],
                  ymls[file_idx+1],
                  ymls[file_idx+2]))
        return EXIT_FAILURE;
  Point p(centerX, centerY);
  int flags = NORMAL_CLONE;

#if !SCDEBUG
  seamlessClone->seamlessCloneGPU( destMat, patchMat, maskMat, p, _blend, flags ); // warm up
#endif

  int   maskWidth = seamlessClone->ucMask.mWidth,
        maskHeight = seamlessClone->ucMask.mHeight;
  checkCudaErrors(cudaStreamSynchronize(params.stream));
  checkCudaErrors( cudaEventRecord(start, params.stream) );

  for( int l=0; l<LOOPS; l++ )
  {
  	seamlessClone->seamlessCloneGPU( destMat, patchMat, maskMat, p, _blend, flags );
#if SCDEBUG
	std::stringstream ss;
	if( l==0 )
		ss<<"./output/ucRGB_Output"<<".bmp";
	else
		ss<<"./output/ucRGB_Output"<<l<<".bmp";
	writeSCImage(ss.str().c_str(), &seamlessClone->ucRGB_Output, 0 );
    	imwrite( ss.str().c_str(), _blend );
	if( l<LOOPS-1 ) seamlessClone->ucRGB_Output.setConstant(0, params.stream);
#endif
	checkCudaErrors(cudaStreamSynchronize(params.stream));
  }

  checkCudaErrors(cudaEventRecord(stop, params.stream));
  checkCudaErrors(cudaEventSynchronize(stop));

  float msCompute = 0.0f;
  checkCudaErrors(cudaEventElapsedTime(&msCompute, start, stop));
  printf( "Compute stage performance time= %.3f msec, patch size=%dx%d\n",
            msCompute/LOOPS, maskWidth, maskHeight);
  //writeSCImage("./output/ucRGB_Output.bmp", &seamlessClone->ucRGB_Output, 0 );
  printf( "total device memory used: %d\n", SCImage::getTotalDeviceMemoryOccupy() );
  checkCudaErrors(cudaStreamDestroy(params.stream));
  checkCudaErrors(cudaEventDestroy(start));
  checkCudaErrors(cudaEventDestroy(stop));
  delete seamlessClone;
  return EXIT_SUCCESS;
}
