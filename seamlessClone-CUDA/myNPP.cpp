#ifndef __MY_NPP_CPP__
#define __MY_NPP_CPP__

#include <npp.h>
#include <nppi.h>
#include <npps.h>
#include <stdio.h>
#include "SC_common.h"

void checkNPPErrors( NppStatus error );

class MyNPP{
public:
	MyNPP(){ printNPPVersion(); }
	~MyNPP(){}

	void setStream( cudaStream_t stream ){ if( nppGetStream()!=stream ) nppSetStream(stream); }
	void printNPPVersion();
	void copyROI( SCImage& dst, int x0, int y0,
                SCImage& src, int x1, int y1,
                int width, int height, cudaStream_t stream );
	void convertFloat2UC( SCImage* dst, SCImage* src, Npp32f scale );
public:


};

void MyNPP::printNPPVersion()
{
	const NppLibraryVersion *libVer   = nppGetLibVersion();
    	printf("NPP Library Version %d.%d.%d\n", libVer->major, libVer->minor, libVer->build);
}

void MyNPP::copyROI( SCImage& dst, int x0, int y0,
                SCImage& src, int x1, int y1,
                int width, int height, cudaStream_t stream )
{
	assert( x0>=0 && y0>=0 &&
		x1>=0 && y1>=0 &&
		dst.mWidth>=x0+width && dst.mHeight>=y0+height &&
		src.mWidth>=x1+width && src.mHeight>=y1+height &&
                width>0 && height>0 );

        for( int c=0; c<dst.mChannel; c++ )
        {
                unsigned char* td = dst.mData + c*dst.sliceSz() + y0*dst.pitch() + x0*dst.elmtSz();
                unsigned char* ts = src.mData + c*src.sliceSz() + y1*src.pitch() + x1*src.elmtSz();
                checkCudaErrors( cudaMemcpy2DAsync((void*)td, dst.pitch(),
                                        	(void*)ts, src.pitch(),
                                        	width*dst.elmtSz(), height,
                                        	cudaMemcpyDeviceToDevice,
						stream) );
#if SCDEBUG
		printf( "nppGetStream()= %p\n", nppGetStream() );
#endif
	}
}

void MyNPP::convertFloat2UC( SCImage* dst, SCImage* src, Npp32f scale )
{
	assert( src->mDType==SCImageDataType_Float );
	SCImage tmpSrc; tmpSrc.resize( src->mWidth, src->mHeight, src->mChannel, src->mDType, SCImageOrder_Row );

	int nSrcStep = src->mWidth*src->elmtSz();
        NppiSize oSizeROI; oSizeROI.width = src->mWidth; oSizeROI.height = src->mHeight;
        //printf("roi(%d,%d)\n", oSizeROI.width, oSizeROI.height);
	for( int ch=0; ch<src->mChannel; ch++ )
	{
		Npp32f* pSrc = (Npp32f*)src->mData   + src->mHeight  *src->mWidth  *ch;
		Npp32f* pDst = (Npp32f*)tmpSrc.mData + tmpSrc.mHeight*tmpSrc.mWidth*ch;
        	checkNPPErrors(nppiMulC_32f_C1R( pSrc, nSrcStep, scale, pDst, nSrcStep, oSizeROI ));
	}
	
	int nLength = src->mWidth*src->mHeight*src->mChannel;
        //checkNPPErrors(nppsConvert_32f8u_Sfs( (const Npp32f*)tmpSrc.mData, dst->mData, nLength, NPP_RND_NEAR, 1 ));
	checkNPPErrors(nppsConvert_32f8u_Sfs( (const Npp32f*)tmpSrc.mData, dst->mData, nLength, NPP_RND_ZERO, 1 ));
	tmpSrc.destroy();
}

void checkNPPErrors( NppStatus error )
{
	if( error==NPP_SUCCESS ) return;

	switch(error)
	{
case_print(NPP_NOT_SUPPORTED_MODE_ERROR )
case_print(NPP_INVALID_HOST_POINTER_ERROR )
case_print(NPP_INVALID_DEVICE_POINTER_ERROR )
case_print(NPP_LUT_PALETTE_BITSIZE_ERROR )
case_print(NPP_ZC_MODE_NOT_SUPPORTED_ERROR )
case_print(NPP_NOT_SUFFICIENT_COMPUTE_CAPABILITY )
case_print(NPP_TEXTURE_BIND_ERROR )
case_print(NPP_WRONG_INTERSECTION_ROI_ERROR )
case_print(NPP_HAAR_CLASSIFIER_PIXEL_MATCH_ERROR )
case_print(NPP_MEMFREE_ERROR )
case_print(NPP_MEMSET_ERROR )
case_print(NPP_MEMCPY_ERROR )
case_print(NPP_ALIGNMENT_ERROR )
case_print(NPP_CUDA_KERNEL_EXECUTION_ERROR )
case_print(NPP_ROUND_MODE_NOT_SUPPORTED_ERROR )
case_print(NPP_QUALITY_INDEX_ERROR )
case_print(NPP_RESIZE_NO_OPERATION_ERROR )
case_print(NPP_OVERFLOW_ERROR )
case_print(NPP_NOT_EVEN_STEP_ERROR )
case_print(NPP_HISTOGRAM_NUMBER_OF_LEVELS_ERROR )
case_print(NPP_LUT_NUMBER_OF_LEVELS_ERROR )
case_print(NPP_CORRUPTED_DATA_ERROR )
case_print(NPP_CHANNEL_ORDER_ERROR )
case_print(NPP_ZERO_MASK_VALUE_ERROR )
case_print(NPP_QUADRANGLE_ERROR )
case_print(NPP_RECTANGLE_ERROR )
case_print(NPP_COEFFICIENT_ERROR )
case_print(NPP_NUMBER_OF_CHANNELS_ERROR )
case_print(NPP_COI_ERROR )
case_print(NPP_DIVISOR_ERROR )
case_print(NPP_CHANNEL_ERROR )
case_print(NPP_STRIDE_ERROR )
case_print(NPP_ANCHOR_ERROR )
case_print(NPP_MASK_SIZE_ERROR )
case_print(NPP_RESIZE_FACTOR_ERROR )
case_print(NPP_INTERPOLATION_ERROR )
case_print(NPP_MIRROR_FLIP_ERROR )
case_print(NPP_MOMENT_00_ZERO_ERROR )
case_print(NPP_THRESHOLD_NEGATIVE_LEVEL_ERROR )
case_print(NPP_THRESHOLD_ERROR )
case_print(NPP_CONTEXT_MATCH_ERROR )
case_print(NPP_FFT_FLAG_ERROR )
case_print(NPP_FFT_ORDER_ERROR )
case_print(NPP_STEP_ERROR )
case_print(NPP_SCALE_RANGE_ERROR )
case_print(NPP_DATA_TYPE_ERROR )
case_print(NPP_OUT_OFF_RANGE_ERROR )
case_print(NPP_DIVIDE_BY_ZERO_ERROR )
case_print(NPP_MEMORY_ALLOCATION_ERR )
case_print(NPP_NULL_POINTER_ERROR )
case_print(NPP_RANGE_ERROR )
case_print(NPP_SIZE_ERROR )
case_print(NPP_BAD_ARGUMENT_ERROR )
case_print(NPP_NO_MEMORY_ERROR )
case_print(NPP_NOT_IMPLEMENTED_ERROR )
case_print(NPP_ERROR )
case_print(NPP_ERROR_RESERVED )
//case_print(NPP_SUCCESS )
case_print(NPP_NO_OPERATION_WARNING )
case_print(NPP_DIVIDE_BY_ZERO_WARNING )
case_print(NPP_AFFINE_QUAD_INCORRECT_WARNING )
case_print(NPP_WRONG_INTERSECTION_ROI_WARNING )
case_print(NPP_WRONG_INTERSECTION_QUAD_WARNING )
case_print(NPP_DOUBLE_SIZE_WARNING )
case_print(NPP_MISALIGNED_DST_ROI_WARNING)
		default:
			printf("NPP_UNDEFINED_RETURN_FLAG\n");
			break;
	}

}

#endif
