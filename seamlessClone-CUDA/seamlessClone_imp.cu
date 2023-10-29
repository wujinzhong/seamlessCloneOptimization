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
//#include "nvJPEG_helper.hxx"

#include "helper_cuda.h"
#include <npp.h>
#include <nppi.h>
#include <npps.h>
#include <stdio.h>
#include "seamlessClone_imp.h"

#include "seamlessClone_imp.cpp"

using namespace std;
using namespace cv;

struct seamlessClone_params_t {
  int dev;
  cudaStream_t stream;
};

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

// write bmp, input - RGB, device
int writeBMP(const char *filename, const unsigned char *d_chanR, int pitchR,
             const unsigned char *d_chanG, int pitchG,
             const unsigned char *d_chanB, int pitchB, int width, int height) {
  unsigned int headers[13];
  FILE *outfile;
  int extrabytes;
  int paddedsize;
  int x;
  int y;
  int n;
  int red, green, blue;

  std::vector<unsigned char> vchanR(height * width);
  std::vector<unsigned char> vchanG(height * width);
  std::vector<unsigned char> vchanB(height * width);
  unsigned char *chanR = vchanR.data();
  unsigned char *chanG = vchanG.data();
  unsigned char *chanB = vchanB.data();
  checkCudaErrors(cudaMemcpy2D(chanR, (size_t)width, d_chanR, (size_t)pitchR,
                               width, height, cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy2D(chanG, (size_t)width, d_chanG, (size_t)pitchR,
                               width, height, cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy2D(chanB, (size_t)width, d_chanB, (size_t)pitchR,
                               width, height, cudaMemcpyDeviceToHost));

  extrabytes =
      4 - ((width * 3) % 4);  // How many bytes of padding to add to each
  // horizontal line - the size of which must
  // be a multiple of 4 bytes.
  if (extrabytes == 4) extrabytes = 0;

  paddedsize = ((width * 3) + extrabytes) * height;

  // Headers...
  // Note that the "BM" identifier in bytes 0 and 1 is NOT included in these
  // "headers".

  headers[0] = paddedsize + 54;  // bfSize (whole file size)
  headers[1] = 0;                // bfReserved (both)
  headers[2] = 54;               // bfOffbits
  headers[3] = 40;               // biSize
  headers[4] = width;            // biWidth
  headers[5] = height;           // biHeight

  // Would have biPlanes and biBitCount in position 6, but they're shorts.
  // It's easier to write them out separately (see below) than pretend
  // they're a single int, especially with endian issues...

  headers[7] = 0;           // biCompression
  headers[8] = paddedsize;  // biSizeImage
  headers[9] = 0;           // biXPelsPerMeter
  headers[10] = 0;          // biYPelsPerMeter
  headers[11] = 0;          // biClrUsed
  headers[12] = 0;          // biClrImportant

  if (!(outfile = fopen(filename, "wb"))) {
    std::cerr << "Cannot open file: " << filename << std::endl;
    return 1;
  }

  //
  // Headers begin...
  // When printing ints and shorts, we write out 1 character at a time to avoid
  // endian issues.
  //
  fprintf(outfile, "BM");

  for (n = 0; n <= 5; n++) {
    fprintf(outfile, "%c", headers[n] & 0x000000FF);
    fprintf(outfile, "%c", (headers[n] & 0x0000FF00) >> 8);
    fprintf(outfile, "%c", (headers[n] & 0x00FF0000) >> 16);
    fprintf(outfile, "%c", (headers[n] & (unsigned int)0xFF000000) >> 24);
  }

  // These next 4 characters are for the biPlanes and biBitCount fields.

  fprintf(outfile, "%c", 1);
  fprintf(outfile, "%c", 0);
  fprintf(outfile, "%c", 24);
  fprintf(outfile, "%c", 0);

  for (n = 7; n <= 12; n++) {
    fprintf(outfile, "%c", headers[n] & 0x000000FF);
    fprintf(outfile, "%c", (headers[n] & 0x0000FF00) >> 8);
    fprintf(outfile, "%c", (headers[n] & 0x00FF0000) >> 16);
    fprintf(outfile, "%c", (headers[n] & (unsigned int)0xFF000000) >> 24);
  }

  //
  // Headers done, now write the data...
  //

  for (y = height - 1; y >= 0;
       y--)  // BMP image format is written from bottom to top...
  {
    for (x = 0; x <= width - 1; x++) {
      red = chanR[y * width + x];
      green = chanG[y * width + x];
      blue = chanB[y * width + x];

      if (red > 255) red = 255;
      if (red < 0) red = 0;
      if (green > 255) green = 255;
      if (green < 0) green = 0;
      if (blue > 255) blue = 255;
      if (blue < 0) blue = 0;
      // Also, it's written in (b,g,r) format...

      fprintf(outfile, "%c", blue);
      fprintf(outfile, "%c", green);
      fprintf(outfile, "%c", red);
    }
    if (extrabytes)  // See above - BMP lines must be of lengths divisible by 4.
    {
      for (n = 1; n <= extrabytes; n++) {
        fprintf(outfile, "%c", 0);
      }
    }
  }

  fclose(outfile);
  return 0;
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
		NPPUtil nppUtil;
		nppUtil.convertFloat2UC( &tmp, img, scale );
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

static Mat readFromYaml( const char* file_path )
{
	Mat mat;
	printf("begin reading file: %s\n", file_path);

	cv::FileStorage fs( file_path, cv::FileStorage::READ);
	fs["data"]>>mat;

    printf("mat shape: %d, %d, %d\n", mat.cols, mat.rows, mat.channels());

	return mat;
}

void* seamlessClone_imp_create_instance( int gpu_id )
{
  int gpu = gpu_id;

  seamlessClone_params_t params;
  params.dev = gpu;
  cudaDeviceProp props;
  checkCudaErrors(cudaGetDeviceProperties(&props, params.dev));
  printf("Using GPU %d (%s, %d SMs, %d th/SM max, CC %d.%d, ECC %s)\n",
         params.dev, props.name, props.multiProcessorCount,
         props.maxThreadsPerMultiProcessor, props.major, props.minor,
         props.ECCEnabled ? "on" : "off");

  // create cuda stream
  checkCudaErrors(
      cudaStreamCreateWithFlags(&params.stream, cudaStreamNonBlocking));

  SeamlessClone *seamlessClone = (SeamlessClone*)malloc(sizeof(SeamlessClone));
  memset( seamlessClone, 0, sizeof(SeamlessClone) );
  seamlessClone->init( params.stream, 
                        props.multiProcessorCount, 
                        props.maxThreadsPerMultiProcessor );
  
  return (void*)seamlessClone; //return EXIT_SUCCESS;
}

Mat seamlessClone_imp_run( void* instance_ptr, void* face, void* body, void* mask, int centerX, int centerY, int gpu_id, bool bSync=true ){
  Mat patchMat=*((Mat*)face), destMat=*((Mat*)body), maskMat=*((Mat*)mask);

  //patchMat = readFromYaml("./images/src.yml");
  //destMat = readFromYaml("./images/dst.yml");
  //maskMat = readFromYaml("./images/src_mask.yml");

  assert( patchMat.rows==maskMat.rows &&
    patchMat.rows==maskMat.rows &&
    patchMat.channels()==3 &&
    maskMat.channels()==1 );

  SeamlessClone *seamlessClone = (SeamlessClone*)instance_ptr;
  assert(seamlessClone!=NULL);
  

  cudaEvent_t start, stop;
  if(bSync)
  {
    checkCudaErrors( cudaEventCreate(&start) );
    checkCudaErrors( cudaEventCreate(&stop) );
  }
  

#if SCDEBUG
  const int LOOPS = 1;
#else
  //const int LOOPS = 50;
  const int LOOPS = 1;
#endif

  // align API with OpenCV //////////////////////////////////////////////////////////////
  // Mat destMat, patchMat, maskMat;
  Mat retMat, _blend;
  //
  Point p(centerX, centerY);
  int flags = NORMAL_CLONE;

#if !SCDEBUG
  retMat = seamlessClone->seamlessCloneGPU( destMat, patchMat, maskMat, p, _blend, flags ); // warm up
#endif

  int   maskWidth = seamlessClone->ucMask.mWidth,
        maskHeight = seamlessClone->ucMask.mHeight;
  
  if(bSync)
  {
    checkCudaErrors(cudaStreamSynchronize(seamlessClone->mStream));
    checkCudaErrors( cudaEventRecord(start, seamlessClone->mStream) );
  }  

  for( int l=0; l<LOOPS; l++ )
  {
    retMat = seamlessClone->seamlessCloneGPU( destMat, patchMat, maskMat, p, _blend, flags );
#if SCDEBUG
    std::stringstream ss;
    if( l==0 )
      ss<<"./output/ucRGB_Output"<<".bmp";
    else
      ss<<"./output/ucRGB_Output"<<l<<".bmp";
    //writeSCImage(ss.str().c_str(), &seamlessClone->ucRGB_Output, 0 );
    imwrite( ss.str().c_str(), _blend );
    if( l<LOOPS-1 ) seamlessClone->ucRGB_Output.setConstant(0, seamlessClone->mStream);
#endif
	  
    if(bSync)
    {
      checkCudaErrors(cudaStreamSynchronize(seamlessClone->mStream));
    }
  }

  if(bSync)
  {
    checkCudaErrors(cudaEventRecord(stop, seamlessClone->mStream));
    checkCudaErrors(cudaEventSynchronize(stop));

    float msCompute = 0.0f;
    checkCudaErrors(cudaEventElapsedTime(&msCompute, start, stop));
    printf( "Compute stage performance time= %.3f msec, patch size=%dx%d\n",
              msCompute/LOOPS, maskWidth, maskHeight);
    //writeSCImage("./output/ucRGB_Output.bmp", &seamlessClone->ucRGB_Output, 0 );
    printf( "total device memory used: %d\n", SCImage::getTotalDeviceMemoryOccupy() );
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));
  }

  return _blend; //return EXIT_SUCCESS;
}

void seamlessClone_imp_destroy( void* instance_ptr ){
  
  SeamlessClone *seamlessClone = (SeamlessClone*)instance_ptr;
  
  checkCudaErrors(cudaStreamDestroy(seamlessClone->mStream));
  seamlessClone->delete_();
  free(seamlessClone); seamlessClone = NULL;

  return; //return EXIT_SUCCESS;
}

void seamlessClone_imp_sync( void* instance_ptr ){
  SeamlessClone *seamlessClone = (SeamlessClone*)instance_ptr;
  checkCudaErrors(cudaStreamSynchronize(seamlessClone->mStream));

  return;
}