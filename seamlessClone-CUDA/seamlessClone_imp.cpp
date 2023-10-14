#ifndef __SEAMLESS_CLONE_CPP__
#define __SEAMLESS_CLONE_CPP__

#include "seamlessClone_imp.h"
#include <cublas_v2.h>
#include <math.h>
#include <cufft.h>
#include <vector>
#include <sstream>
#include "opencv2/core.hpp"

#include <npp.h>
#include <nppi.h>
#include <npps.h>
#include <stdio.h>

#if SC_Enable_Cooperative_Group
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
#endif

#define SC_Test false

static inline int iDivUp(int n, int m)
{
    return (n + m - 1) / m;
}

typedef struct SCComplex{
    float r, c;
}SCComplex;

dim3 calCooperativeGroupBlocks( void* func, int sharedMem, int& scale, dim3& threads, int width, int height, int SM );

class FFTParams{
public:
    FFTParams():ny(0), type(cufftType(0)), nx(0), plan(0){}
    FFTParams( int ny_, cufftType type_, int nx_, cufftHandle plan_):
    ny(ny_), type(type_), nx(nx_), plan(plan_){}
    bool isEqualTo( int ny, cufftType type, int nx )
    {
        return ny==this->ny && type==this->type && nx==this->nx;
    }
    int ny;
    cufftType type;
    int nx;
    cufftHandle plan;
};

void checkNPPErrors( NppStatus error );

void NPPUtil::printNPPVersion()
{
	const NppLibraryVersion *libVer   = nppGetLibVersion();
    	printf("NPP Library Version %d.%d.%d\n", libVer->major, libVer->minor, libVer->build);
}

void NPPUtil::copyROI( SCImage& dst, int x0, int y0,
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

void NPPUtil::convertFloat2UC( SCImage* dst, SCImage* src, Npp32f scale )
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


class SeamlessClone{
public:
	SeamlessClone( /*cudaStream_t stream, 
			int multiProcessorCount,
                	int maxThreadsPerMultiProcessor*/ )
	{
		// init(stream, multiProcessorCount, maxThreadsPerMultiProcessor);
	}

    void init( cudaStream_t stream, 
			int multiProcessorCount,
                	int maxThreadsPerMultiProcessor )
    {
        mStream = stream; 
        mSM = multiProcessorCount;
        mMaxThdPerSM = maxThreadsPerMultiProcessor; 
        {
            assert(mStream);
            nppUtil.setStream(mStream);
            checkCublasErrors(cublasCreate(&mCublasHandle));
            checkCublasErrors(cublasSetStream( mCublasHandle, mStream ));

            for( int i=0; i<4; i++ )
            {
                blasParamsA[i] = NULL;
                blasParamsB[i] = NULL;
                blasParamsC[i] = NULL;
            }
        }
    }
	// ~SeamlessClone()
	// {
    //    delete_();
	// }

	void init_resize();
	void run();
	cv::Mat seamlessCloneGPU( Mat dst, Mat patch, Mat mask, Point point, Mat& blend, int flag );
    void delete_()
    {
        RELEASE_DEV_PTR( ucRGB_Bkgrd.mData );
		RELEASE_DEV_PTR( ucRGB_Patch.mData );
		RELEASE_DEV_PTR( Tmp_Bkgrd.mData );
		RELEASE_DEV_PTR( Tmp_Patch.mData );
		RELEASE_DEV_PTR( gdX.mData );
        RELEASE_DEV_PTR( gdY.mData );
        RELEASE_DEV_PTR( ucMask_Org.mData );
		RELEASE_DEV_PTR( ucMask.mData ); //ucMask is referred from outside.
        RELEASE_DEV_PTR( ucMaskDup.mData );
		RELEASE_DEV_PTR( lapXY.mData );
		RELEASE_DEV_PTR( g.mData );
		RELEASE_DEV_PTR( u.mData );
		RELEASE_DEV_PTR( u_.mData );
		RELEASE_DEV_PTR( Vn1_1.mData );
		RELEASE_DEV_PTR( Vn2_1.mData );
		RELEASE_DEV_PTR( Tmp_n2_1.mData );
		RELEASE_DEV_PTR( lambda_n1_1.mData );
		RELEASE_DEV_PTR( lambda_n2_1.mData );
		RELEASE_DEV_PTR( lambda_matrix_n2_1.mData );
		RELEASE_DEV_PTR( ucRGB_Output.mData );
		RELEASE_DEV_PTR( mRect.mData );
		RELEASE_DEV_PTR( blasParams.mData );
		RELEASE_DEV_PTR( filter_X.mData );
		RELEASE_DEV_PTR( filter_Y.mData );
		RELEASE_DEV_PTR( tempComplex0.mData );
		RELEASE_DEV_PTR( tempComplex1.mData );

		checkCublasErrors( cublasDestroy(mCublasHandle) );
		cufftDestroy(fftParams0.plan);
		cufftDestroy(fftParams1.plan);
    }
private:
    void Mat2SCImage_resize( Mat dst, Mat patch, Mat mask, Mat blend,
                            SCImage& tmp_bkgrd, SCImage& tmp_patch,
                            cudaStream_t stream);
	void poissonSolver2D( SCImage& g, SCImage& u );
	void poissonSolver2D_FFT( SCImage& g, SCImage& u );
	void solve(SCImage& g, SCImage& u, int channelIdx);
	void dst(SCImage& g, SCImage& u, bool invert/* = false*/, int channelIdx);
	void ImageMultiplyPerSlice( SCImage& C, SCImage& A, SCImage& B, int blasIdx, cublasOperation_t transa=CUBLAS_OP_N );
	void initMask( ); // same as in OpenCV for 3 times erode.
	void initBlas( SCImage* images, int imageNum );
	void initBlas_resize( SCImage* images, int imageNum );
	void pre_process_v2();
#if SC_Test
	void testDST( SCImage& g, SCImage& u );
	void resetLambdaMatrix( float* lambdaMatrix, int n );
#endif
	void resetDSTMatrix( float* dstm, int n );
	void resetLambda( float* lambda, int n );
	void initDSTMatrix( float* Vn1_1, float* Vn2_1, float* lambda_n1_1, float* lambda_n2_1, float* lambda_matrix_n2_1, int nx, int ny );
    void initDSTMatrix_resize( float* Vn1_1, float* Vn2_1, float* lambda_n1_1, float* lambda_n2_1,
                                float* lambda_matrix_n2_1,
                                float* filter_X, float* filter_Y,
                                int nx, int ny );

public:
	cudaStream_t mStream;

	SCImage ucRGB_Bkgrd, ucRGB_Patch, Tmp_Bkgrd, Tmp_Patch;
	SCImage gdX, gdY;
	SCImage ucMask, ucMaskDup, ucMask_Org;
	SCImage lapXY;
	SCImage g, u, u_;
	SCImage ucRGB_Output;
	SCImage Vn1_1, Vn2_1, lambda_n1_1, lambda_n2_1, lambda_matrix_n2_1;
	SCImage Tmp_n2_1;
	SCImage mRect;
	SCImage blasParams;
	SCImage filter_X, filter_Y;
	SCImage tempComplex0, tempComplex1;
	FFTParams fftParams0, fftParams1;

	cublasHandle_t mCublasHandle;
	float** blasParamsA[4];
	float** blasParamsB[4];
	float** blasParamsC[4];
	Point2D centerPt, leftTop, mPatchOffset;
	NPPUtil nppUtil;
	int mSM, mMaxThdPerSM;
};

void Mat2SCImage( SCImage& dst, unsigned char* ptr, Mat src, cudaStream_t mStream )
{
    int sliceSz = src.rows*src.cols*sizeof(unsigned char);
    const int CH = src.channels();
    const int ROWS = src.rows;
    const int COLS = src.cols;
    unsigned char* tp = ptr;

    if( src.channels()==3 )
    {
        for( int row=0; row<ROWS; row++ )
        {
            unsigned char* line = src.ptr<unsigned char>(row);
            for( int col=0; col<COLS; col++ )
            {
                tp[0*sliceSz] = line[2];
                tp[1*sliceSz] = line[1];
                tp[2*sliceSz] = line[0];
                line+=3;
                tp++;
            }
        }
    }
    else if( src.channels()==1 )
    {
        for( int row=0; row<ROWS; row++ )
        {
            unsigned char* line = src.ptr<unsigned char>(row);
            for( int col=0; col<COLS; col++ )
            {
                *tp++ = *line++;
            }
        }
    }
    else
    {
        assert(false);
        for( int row=0; row<ROWS; row++ )
        {
            unsigned char* line = src.ptr<unsigned char>(row);
            for( int col=0; col<COLS; col++ )
            {
                for( int ch=0; ch<CH; ch++ )
                {
                    tp[ch*sliceSz] = line[CH-ch-1];
                }
                line+=CH;
                tp++;
            }
        }
    }
    dst.resize( src.cols, src.rows, src.channels(), SCImageDataType_UC, SCImageOrder_Row );
    checkCudaErrors( cudaMemcpyAsync( dst.mData, ptr, dst.sz(), cudaMemcpyHostToDevice, mStream ) );
    return;
}

__global__ void Mat2SCImage_kernel( unsigned char* dst2, unsigned char* dst1, uchar3* dst0, int dstWidth, int dstHeight,
                                    unsigned char* patch1, uchar3* patch0, int patchWidth, int patchHeight,
                                    uchar1* mask2, uchar1* mask1, uchar1* mask0, int maskWidth, int maskHeight)
{
    int x = threadIdx.x + blockDim.x*blockIdx.x;
    int y = threadIdx.y + blockDim.y*blockIdx.y;

    if( x<dstWidth && y<dstHeight )
    {
	int idx = y*dstWidth+x;
        uchar3 uc3 = dst0[idx];

        int sliceSz = dstWidth*dstHeight;
        dst2[idx+0*sliceSz] = uc3.z;
        dst2[idx+1*sliceSz] = uc3.y;
        dst2[idx+2*sliceSz] = uc3.x;

        dst1[idx+0*sliceSz] = uc3.z;
        dst1[idx+1*sliceSz] = uc3.y;
        dst1[idx+2*sliceSz] = uc3.x;
    }

    if( x<patchWidth && y<patchHeight )
    {
	int idx = y*patchWidth+x;
        uchar3 uc3 = patch0[idx];
        
        int sliceSz = patchWidth*patchHeight;
        patch1[idx+0*sliceSz] = uc3.z;
        patch1[idx+1*sliceSz] = uc3.y;
        patch1[idx+2*sliceSz] = uc3.x;
    }

    if( x<maskWidth && y<maskHeight )
    {
	int idx = y*maskWidth+x;
        uchar1 uc1 = mask0[idx];
        
        mask1[idx] = uc1;
        mask2[idx] = uc1;
    }
}

void SeamlessClone::Mat2SCImage_resize( Mat dst, Mat patch, Mat mask, Mat blend,
                            SCImage& tmp_bkgrd, SCImage& tmp_patch,
                            cudaStream_t stream)
{
    //printf("call %s\n", __FUNCTION__);
    tmp_bkgrd   .resize( dst.cols, dst.rows, dst.channels(), SCImageDataType_UC, SCImageOrder_Row );
    ucRGB_Bkgrd .resize( dst.cols, dst.rows, dst.channels(), SCImageDataType_UC, SCImageOrder_Row );
    ucRGB_Output.resize( dst.cols, dst.rows, dst.channels(), SCImageDataType_UC, SCImageOrder_Row );
    tmp_patch  .resize( patch.cols, patch.rows, patch.channels(), SCImageDataType_UC, SCImageOrder_Row );
    
    ucRGB_Patch.resize( patch.cols, patch.rows, patch.channels(), SCImageDataType_UC, SCImageOrder_Row );
    ucMask     .resize( mask.cols, mask.rows, mask.channels(), SCImageDataType_UC, SCImageOrder_Row );
    ucMask_Org .resize( mask.cols, mask.rows, mask.channels(), SCImageDataType_UC, SCImageOrder_Row );
    ucMaskDup  .resize( mask.cols, mask.rows, mask.channels(), SCImageDataType_UC, SCImageOrder_Row );

    checkCudaErrors( cudaMemcpyAsync( tmp_bkgrd.mData, dst.ptr<unsigned char>(0), tmp_bkgrd.sz(), cudaMemcpyHostToDevice, stream) );
    checkCudaErrors( cudaMemcpyAsync( tmp_patch.mData, patch.ptr<unsigned char>(0), tmp_patch.sz(), cudaMemcpyHostToDevice, stream) );
    checkCudaErrors( cudaMemcpyAsync( ucMask.mData, mask.ptr<unsigned char>(0), ucMask.sz(), cudaMemcpyHostToDevice, stream) );

    dim3 threads(16, 16);
    dim3 blocks(iDivUp(tmp_bkgrd.mWidth, threads.x), iDivUp(tmp_bkgrd.mHeight, threads.y));
    Mat2SCImage_kernel<<<blocks, threads, 0, stream>>>( ucRGB_Bkgrd.mData, ucRGB_Output.mData, (uchar3*)tmp_bkgrd.mData, ucRGB_Bkgrd.mWidth, ucRGB_Bkgrd.mHeight,
                                                        ucRGB_Patch.mData, (uchar3*)Tmp_Patch.mData, ucRGB_Patch.mWidth, ucRGB_Patch.mHeight,
                                                        (uchar1*)ucMask_Org.mData, (uchar1*)ucMaskDup.mData, (uchar1*)ucMask.mData, ucMask.mWidth, ucMask.mHeight);
}

cv::Mat SeamlessClone::seamlessCloneGPU( Mat dst, Mat patch, Mat mask, Point point, Mat& blend, int flag )
{
    assert( dst.type()== CV_8UC3 &&
            patch.type() == CV_8UC3 &&
            mask.type() == CV_8UC1);
    assert( dst.rows*dst.cols>=patch.rows*patch.cols );
    assert( dst.rows*dst.cols>=mask.rows*mask.cols );

    if( dst.isContinuous() &&
        patch.isContinuous() &&
        mask.isContinuous() )
    {
        Mat2SCImage_resize( dst, patch, mask, blend, Tmp_Bkgrd, Tmp_Patch, mStream );
    }
    else
    {
        // set background
        unsigned char * ptr = new unsigned char[dst.rows*dst.cols*sizeof(unsigned char)*dst.channels()];
        Mat2SCImage( ucRGB_Bkgrd, ptr, dst, mStream );

        // set patch
        Mat2SCImage( ucRGB_Patch, ptr, patch, mStream );

        // set mask
        Mat2SCImage( ucMask, ptr, mask, mStream );
        Mat2SCImage( ucMask_Org, ptr, mask, mStream );

        ucRGB_Output.resize( ucRGB_Bkgrd.mWidth, ucRGB_Bkgrd.mHeight, ucRGB_Bkgrd.mChannel, SCImageDataType_UC, SCImageOrder_Row );
        ucRGB_Output.copyFrom( ucRGB_Bkgrd.mData, mStream );
        delete[] ptr;
    }

    centerPt = Point2D( point.x, point.y );

    _print( ucRGB_Bkgrd, mStream );
    _print( ucRGB_Patch, mStream );

    run();

    // copy to Host
    blend = dst; assert( blend.channels()==3 && u_.mChannel==3 );
    unsigned char* hPtr = u_.copyD2H( mStream );
    unsigned char* hTmpPtr = hPtr;
    for( int row=0; row<g.mHeight; row++ )
    {
        unsigned char* dp = blend.ptr<unsigned char>(row+leftTop.y+1);
        dp += (leftTop.x+1)*3;
        for( int col=0; col<g.mWidth; col++ )
        {
            *dp++ = *hTmpPtr++;
            *dp++ = *hTmpPtr++;
            *dp++ = *hTmpPtr++;
        }
    }
    delete[] hPtr;
    return dst;
}

void SeamlessClone::resetDSTMatrix( float* dstm, int n )
{
        float* tmpDst = new float[n*n];
        for( int j=0; j<n; j++ )
        {
                for( int i=0; i<n; i++ )
                {
			int idx = j*n + i;
			tmpDst[idx] = sin((i+1)*(j+1)*PI/(n+1)) * sqrt(2.0f/(n+1));
                }
        }
	//normalization
	/*
	for( int j=0; j<n; j++ )
	{
		float sum2 = 0.0f;
		for( int i=0; i<n; i++ )
		{
			int idx = j*n + i;
			sum2 += tmpDst[idx]*tmpDst[idx];
		}
		float scale = 1.0f/sqrt(sum2);
		for( int i=0; i<n; i++ )
                {
                        int idx = j*n + i;
                        tmpDst[idx] *= scale;
                }
	}
	*/
        checkCudaErrors( cudaMemcpyAsync( dstm, tmpDst, n*n*sizeof(float), cudaMemcpyHostToDevice, mStream) );
        delete[] tmpDst;
}

void SeamlessClone::resetLambda( float* lambda, int n )
{
    float* tmpDst = new float[n];
    for( int i=0; i<n; i++ )
    {
        tmpDst[i] = 2 *(cos ((i+1)*PI/(n+1)) -1);
    }
    checkCudaErrors( cudaMemcpyAsync( lambda, tmpDst, n*sizeof(float), cudaMemcpyHostToDevice, mStream) );
    delete[] tmpDst;
}

#if SC_Test
void SeamlessClone::resetLambdaMatrix( float* lambdaMatrix, int n )
{
    float* tmpDst = new float[n*n];
    memset( tmpDst, 0, sizeof(float)*n*n );
    for( int i=0; i<n; i++ )
    {
        tmpDst[i*n+i] = 2 *(cos ((i+1)*PI/(n+1)) -1);
    }
    checkCudaErrors( cudaMemcpyAsync( lambdaMatrix, tmpDst, n*n*sizeof(float), cudaMemcpyHostToDevice, mStream) );
    delete[] tmpDst;
}

void SeamlessClone::testDST( SCImage& g, SCImage& u )
{
    _print( lambda_matrix_n2_1, mStream );
    ImageMultiplyPerSlice( Tmp_n2_1, lambda_matrix_n2_1, Vn2_1 );
    ImageMultiplyPerSlice( lambda_matrix_n2_1, Vn2_1, Tmp_n2_1 );

    _print( lambda_matrix_n2_1, mStream );
}

#endif

void SeamlessClone::initDSTMatrix( float* Vn1_1, float* Vn2_1, float* lambda_n1_1, float* lambda_n2_1, float* lambda_matrix_n2_1, int nx, int ny )
{
	resetDSTMatrix( Vn1_1, ny );
	resetDSTMatrix( Vn2_1, nx );
	resetLambda( lambda_n1_1, ny );
	resetLambda( lambda_n2_1, nx );
#if SC_Test
	resetLambdaMatrix( lambda_matrix_n2_1, nx );
#endif
}

#if true
#define MY_PRECISION_SCALE 1.0
__global__ void initDSTMatrix_kernel( float* Vn1_1, float* Vn2_1, float* lambda_n1_1, float* lambda_n2_1,
                                       float* lambda_matrix_n2_1,
                                       float* filter_X, float* filter_Y,
                                       int nx, int ny )
{
    int x = threadIdx.x + blockDim.x*blockIdx.x;
    int y = threadIdx.y + blockDim.y*blockIdx.y;

    int n = ny;
    if( x<n && y<n ) //resetDSTMatrix( Vn1_1, ny )
    {
        int idx = y*n + x;
		Vn1_1[idx] = sin((x+1.0)*(y+1.0)*PI/(n+1.0)) * sqrt(2.0/(n+1.0)) * MY_PRECISION_SCALE;

		if( x==y ) //resetLambda( lambda_n1_1, ny )
		{
		    lambda_n1_1[x] = 2.0 *(cos ((x+1.0)*PI/(n+1.0)) -1.0) * MY_PRECISION_SCALE;
            filter_Y[x] = 2.0 * cos(PI/(n+1.0) * (x + 1.0));
        }
    }

    n = nx;
    if( x<n && y<n ) //resetDSTMatrix( Vn2_1, nx )
    {
        int idx = y*n + x;
		Vn2_1[idx] = sin((x+1.0)*(y+1.0)*PI/(n+1.0)) * sqrt(2.0/(n+1.0)) * MY_PRECISION_SCALE;

		if( x==y ) //resetLambda( lambda_n2_1, nx )
		{
		    lambda_n2_1[x] = 2.0 *(cos((x+1.0)*PI/(n+1.0)) -1.0) * MY_PRECISION_SCALE;
            filter_X[x] = 2.0 * cos(PI/(n+1.0) * (x + 1.0));
		}

    }
}

#else
#define MY_PRECISION_SCALE 1.0
__global__ void initDSTMatrix_kernel( float* Vn1_1, float* Vn2_1, float* lambda_n1_1, float* lambda_n2_1,
                                        float* lambda_matrix_n2_1,
                                        float* filter_X, float* filter_Y,
                                        int nx, int ny )
{
    int x = threadIdx.x + blockDim.x*blockIdx.x;
    int y = threadIdx.y + blockDim.y*blockIdx.y;

    int n = ny;
    if( x<n && y<n ) //resetDSTMatrix( Vn1_1, ny )
    {
        int idx = y*n + x;
		Vn1_1[idx] = sinf((x+1.0f)*(y+1.0f)*PI/(n+1.0f)) * sqrtf(2.0f/(n+1.0f));

		if( x==y ) //resetLambda( lambda_n1_1, ny )
		{
		    lambda_n1_1[x] = 2.0f *(cosf ((x+1.0f)*PI/(n+1.0f)) -1.0f);
            filter_Y[x] = 2.0 * cos(PI/(n+1.0) * (x + 1.0));
        }
    }

    n = nx;
    if( x<n && y<n ) //resetDSTMatrix( Vn2_1, nx )
    {
        int idx = y*n + x;
		Vn2_1[idx] = sinf((x+1.0f)*(y+1.0f)*PI/(n+1.0f)) * sqrtf(2.0f/(n+1.0f));

		if( x==y ) //resetLambda( lambda_n2_1, nx )
		{
		    lambda_n2_1[x] = 2.0f *(cosf((x+1.0f)*PI/(n+1.0f)) -1.0f);
            filter_X[x] = 2.0 * cos(PI/(n+1.0) * (x + 1.0));
        }
    }
}

#endif

void SeamlessClone::initDSTMatrix_resize( float* Vn1_1, float* Vn2_1, float* lambda_n1_1, float* lambda_n2_1,
                                            float* lambda_matrix_n2_1,
                                            float* filter_X, float* filter_Y,
                                            int nx, int ny )
{
    dim3 threads(16, 16);
    int maxXY=nx>ny?nx:ny;
    dim3 blocks(iDivUp(maxXY, threads.x), iDivUp(maxXY, threads.y));

    initDSTMatrix_kernel<<<blocks, threads, 0, mStream>>>( Vn1_1, Vn2_1, lambda_n1_1, lambda_n2_1, lambda_matrix_n2_1,
                                                            filter_X, filter_Y,
                                                            nx, ny );

	/*resetDSTMatrix( Vn1_1, ny );
	resetDSTMatrix( Vn2_1, nx );
	resetLambda( lambda_n1_1, ny );
	resetLambda( lambda_n2_1, nx );
#if SC_Test
	resetLambdaMatrix( lambda_matrix_n2_1, nx );
#endif
    */
}

#if SC_Enable_Cooperative_Group
__global__ void myErode_cg( unsigned char* dst, unsigned char* src, int width, int height, int loops, int scale,
                            int mskWidth, int mskHeight, int mskOffsetX, int mskOffsetY )
{
        int X = threadIdx.x + blockDim.x*blockIdx.x;
        int Y = threadIdx.y + blockDim.y*blockIdx.y;
	cg::grid_group grid = cg::this_grid();

	    int src_sliceSz = width*height;
	    int dst_sliceSz = mskWidth*mskHeight;
	    for( int x=scale*X; x<scale*(X+1); x++ )
	    {
            for( int y=scale*Y; y<scale*(Y+1); y++ )
            {
                // src channel = 3
                if( x<width && y<height )
                {
                    src[0*src_sliceSz + y*width+x] = dst[0*dst_sliceSz + (y+mskOffsetY)*mskWidth + (x+mskOffsetX)];
                    src[1*src_sliceSz + y*width+x] = dst[1*dst_sliceSz + (y+mskOffsetY)*mskWidth + (x+mskOffsetX)];
                    src[2*src_sliceSz + y*width+x] = dst[2*dst_sliceSz + (y+mskOffsetY)*mskWidth + (x+mskOffsetX)];
                }
            }
        }
        grid.sync();

        for( int loop=0; loop<loops; loop++ )
        {
		for( int x=scale*X; x<scale*(X+1); x++ )
                for( int y=scale*Y; y<scale*(Y+1); y++ )
                if( x<width && y<height )
                {
                        int idx = y*width+x;
                        if( x==0 || y==0 || x==(width-1) || y==(height-1) )
                        {
                                dst[idx] = 0;
                        }
                        else
                        {
                                int sum = (int)src[idx] +
                                          (int)src[idx-1] +
                                          (int)src[idx+1] +
                                          (int)src[idx+width] +
                                          (int)src[idx-1+width] +
                                          (int)src[idx+1+width] +
                                          (int)src[idx-width] +
                                          (int)src[idx-1-width] +
                                          (int)src[idx+1-width];
                                if( sum!=(255*9) )
                                        dst[idx] = 0;
				else
                                        dst[idx] = 255;
                        }
                }
		unsigned char* tmp = src;
                src = dst;
                dst = tmp;
		grid.sync();
        }
}

__global__ void calBoundingBox_cg( int* bb, unsigned char* mask, int width, int height, int scale )
{
	int X = threadIdx.x + blockDim.x*blockIdx.x;
    int Y = threadIdx.y + blockDim.y*blockIdx.y;
	cg::thread_block block = cg::this_thread_block();
	cg::grid_group grid = cg::this_grid();

    if( X==0 && Y==0 )
    {
        bb[0] = width-1;
		bb[1] = 0;
		bb[2] = height-1;
		bb[3] = 0;
    }
    grid.sync();

	__shared__ int rect[4];
	if( threadIdx.x==0 && threadIdx.y==0 )
	{
		rect[0] = width-1;
		rect[1] = 0;
		rect[2] = height-1;
		rect[3] = 0;
	}
	//__syncthreads();
	block.sync();

    for( int x=X*scale; x<(X+1)*scale; x++ )
    for( int y=Y*scale; y<(Y+1)*scale; y++ )
	if( x<width && y<height )
	{
		if( mask[y*width+x]!=0 )
		{
			atomicMin( &rect[0], x );
			atomicMax( &rect[1], x );
			atomicMin( &rect[2], y );
			atomicMax( &rect[3], y );
		}
	}

	//__syncthreads();
	block.sync();

	if( threadIdx.x==0 && threadIdx.y==0 )
    {
        atomicMin( &bb[0], rect[0] );
        atomicMax( &bb[1], rect[1] );
        atomicMin( &bb[2], rect[2] );
        atomicMax( &bb[3], rect[3] );
    }
}

__global__ void pre_process_kernel_cg( float* LAPXY, float* G,
                unsigned char* BKGRD, int bkgrdWidth, int bkgrdHeight, int bkgrdPitch,
                float* GDX, float* GDY,
                unsigned char* PATCH, int patchWidth, int patchHeight, int patch_offsetX, int patch_offsetY,
                unsigned char* MASK,
                int width, int height, int channel,
                int ltX, int ltY,
	        int scale )
{
        int X = threadIdx.x + blockDim.x*blockIdx.x;
        int Y = threadIdx.y + blockDim.y*blockIdx.y;
	cg::grid_group grid = cg::this_grid();

	{
		unsigned char* bkgrd = BKGRD;
		float* gdX = GDX;
		float* gdY = GDY;
		unsigned char* patch = PATCH;
		unsigned char* mask = MASK;
		for( int ch=0; ch<channel; ch++ )
		{
			for( int x=scale*X; x<scale*(X+1); x++ )
			for( int y=scale*Y; y<scale*(Y+1); y++ )
			if(x<width && y<height)
			{
				unsigned char* ptr0 = bkgrd + (y+ltY)*bkgrdWidth + x+ltX;
				float np;
				if((x)<(width-1)) np = (float)(ptr0[1]); else np = (float)ptr0[-1]; // BORDER_DEFAULT
				float gdX_bk = (np-(float)ptr0[0]);

				if((y)<(height-1)) np = (float)(ptr0[bkgrdPitch]); else np = (float)ptr0[-bkgrdPitch]; // BORDER_DEFAULT
				float gdY_bk = (np-(float)ptr0[0]);

				unsigned char* ptr = patch + (y+patch_offsetY)*patchWidth + (x+patch_offsetX);
				if( x<width-1 ) np = (float)(ptr[1]); else np = (float)ptr[-1]; // BORDER_DEFAULT
				float gdX_pt = (np-(float)ptr[0]);

				if( y<height-1 ) np = (float)(ptr[patchWidth]); else np = (float)ptr[-patchWidth]; // BORDER_DEFAULT
				float gdY_pt = (np-(float)ptr[0]);

				float msk = (float)mask[y*width+x] * (1.0f/255.0f);
				float _gdX, _gdY;
				_gdX = (1.0f-msk)*gdX_bk + msk*gdX_pt;
				_gdY = (1.0f-msk)*gdY_bk + msk*gdY_pt;
				gdX[y*width+x] = _gdX;
				gdY[y*width+x] = _gdY;

			}
			bkgrd += bkgrdPitch*bkgrdHeight;
			patch += patchWidth*patchHeight;
			gdX   += width*height;
			gdY   += width*height;
		}
	}

	grid.sync();

	{
		float* lapXY = LAPXY;
		unsigned char* bkgrd = BKGRD;
		float* gdX = GDX;
		float* gdY = GDY;
		float* g = G;

		for( int ch=0; ch<channel; ch++ )
		{
			for( int x=scale*X; x<scale*(X+1); x++ )
                        for( int y=scale*Y; y<scale*(Y+1); y++ )
			if( x<width && y<height )
			{
				float lap = 0.0f;
				//lapXY[y*width+x] = 0.0f;
				if(x>=1 && x<=(width-2) && y>=1 && y<=(height-2))
				{
					float* gdX_ptr = gdX + y*width + x;
					float* gdY_ptr = gdY + y*width + x;
					float _gdX = gdX_ptr[0]-gdX_ptr[-1];
					float _gdY = gdY_ptr[0]-gdY_ptr[-width];
					lap += _gdX + _gdY;

					unsigned char* ptr = bkgrd + (y+ltY)*bkgrdWidth + x+ltX;
					if( x==1 )
					{
						lap -= ((float)ptr[-1]);
					}
					if( y==1 )
					{
						lap -= ((float)ptr[-bkgrdWidth]);
					}
					if( x==(width-2) )
					{
						lap -= ((float)ptr[1]);
					}
					if( y==(height-2) )
					{
						lap -= ((float)ptr[bkgrdWidth]);
					}
					g[(y-1)*(width-2)+x-1] = lap;
					lapXY[y*width+x] = lap;
				}

			}
			bkgrd += bkgrdPitch*bkgrdHeight;
			lapXY += width*height;
			gdX   += width*height;
			gdY   += width*height;
			g     += (width-2)*(height-2);
		}
	}
}

#else

__global__ void myErode( unsigned char* dst, unsigned char* src, int width, int height, int loops )
{
	int x = threadIdx.x + blockDim.x*blockIdx.x;
        int y = threadIdx.y + blockDim.y*blockIdx.y;

	//for( int loop=0; loop<loops; loop++ )
	{
		if( x<width && y<height )
		{
			int idx = y*width+x;
			if( x==0 || y==0 || x==(width-1) || y==(height-1) )
			{
				dst[idx] = 0;
			}
			else
			{
				int sum = (int)src[idx] +
					  (int)src[idx-1] +
					  (int)src[idx+1] +
					  (int)src[idx+width] +
					  (int)src[idx-1+width] +
					  (int)src[idx+1+width] +
					  (int)src[idx-width] +
					  (int)src[idx-1-width] +
					  (int)src[idx+1-width];
				if( sum!=(255*9) )
					dst[idx] = 0;
				else
					dst[idx] = 255;
			}
		}

	}
}

__global__ void calBoundingBox( int* bb, unsigned char* mask, int width, int height )
{
	int x = threadIdx.x + blockDim.x*blockIdx.x;
        int y = threadIdx.y + blockDim.y*blockIdx.y;

	__shared__ int rect[4];

	if( threadIdx.x==0 && threadIdx.y==0 )
	{
		rect[0] = width-1;
		rect[1] = 0;
		rect[2] = height-1;
		rect[3] = 0;
	}
	__syncthreads();

	if( x<width && y<height )
	{
		if( mask[y*width+x]!=0 )
		{
			atomicMin( &rect[0], x );
			atomicMax( &rect[1], x );
			atomicMin( &rect[2], y );
			atomicMax( &rect[3], y );
		}
	}

	__syncthreads();

	if( threadIdx.x==0 && threadIdx.y==0 )
        {
                atomicMin( &bb[0], rect[0] );
                atomicMax( &bb[1], rect[1] );
                atomicMin( &bb[2], rect[2] );
                atomicMax( &bb[3], rect[3] );
        }
}

#endif

__global__ void setMaskBoundaryToConstant( unsigned char* mask, int width, int height, int left, int right, int top, int bottom, unsigned char value )
{
	int x = threadIdx.x + blockDim.x*blockIdx.x;
    int y = threadIdx.y + blockDim.y*blockIdx.y;
	if( x<width && y<height )
	{
		if( x<left || x>=width-right || y<top || y>=height-bottom )
			mask[ y*width+x ] = value;
	}
}

void SeamlessClone::initMask( )
{
	int loops = 4;
    //int kernel[9] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
	int kx=3, ky=3;
    assert( loops==4 && kx==ky && kx==3);

	_print( ucMask, mStream );
	dim3 threads(16, 16);
    dim3 blocks(iDivUp(ucMask.mWidth, threads.x), iDivUp(ucMask.mHeight, threads.y));

	setMaskBoundaryToConstant<<<blocks, threads, 0, mStream>>>( ucMask.mData, ucMask.mWidth, ucMask.mHeight, 1,1,1,1, 0 );
	_print( ucMask, mStream );

	mRect.resize(4, 1, 1, SCImageDataType_Int32, SCImageOrder_Row);
	int sharedMemSz = sizeof(int)*4;
#if SC_Enable_Cooperative_Group
    int scale = 1;
	blocks = calCooperativeGroupBlocks( (void*)calBoundingBox_cg, sharedMemSz, scale, threads, ucMask.mWidth, ucMask.mHeight, mSM );
    void *kernelArgs[] = {
				(void*)&(mRect.mData),
				(void*)&(ucMask.mData),
				(void*)&ucMask.mWidth,
				(void*)&ucMask.mHeight,
				(void*)&scale,
			};
	checkCudaErrors( cudaLaunchCooperativeKernel( (void*)calBoundingBox_cg, blocks, threads, kernelArgs, sharedMemSz, mStream ) );
#else
    int rectBuf[4] = { ucMask.mWidth-1, 0, ucMask.mHeight-1, 0 }; // left, right, top, bottom, left<right, top<bottom;
	checkCudaErrors( cudaMemcpyAsync( mRect.mData, rectBuf, mRect.sz(), cudaMemcpyHostToDevice, mStream ));
	calBoundingBox<<<blocks, threads, sharedMemSz, mStream>>>( (int*)mRect.mData, ucMask.mData, ucMask.mWidth, ucMask.mHeight );
#endif
	_print(mRect, mStream);

	int* hostRect = mRect.hostPtr<int>( mStream );
	assert( (hostRect[1]-hostRect[0])>0 && (hostRect[3]-hostRect[2])>0 );
	mPatchOffset.x = hostRect[0];
	mPatchOffset.y = hostRect[2];
	ucMaskDup.resize( hostRect[1]-hostRect[0]+1, hostRect[3]-hostRect[2]+1, ucMask.mChannel, SCImageDataType_UC, SCImageOrder_Row );
	{
#if SC_Enable_Cooperative_Group
		{
		    /*nppUtil.copyROI(  ucMaskDup, 0, 0,
                            ucMask, hostRect[0], hostRect[2],
                            ucMaskDup.mWidth, ucMaskDup.mHeight,
                            mStream );
            ucMask.resize( ucMaskDup.mWidth, ucMaskDup.mHeight, ucMaskDup.mChannel, SCImageDataType_UC, SCImageOrder_Row );
            */

			dim3 threads(16, 16);
			int scale = 1;
			dim3 blocks = calCooperativeGroupBlocks( (void*)myErode_cg, 0, scale, threads, ucMaskDup.mWidth, ucMaskDup.mHeight, mSM );
			printf( "grid{%d,%d,%d}, scale=%d\n", blocks.x, blocks.y, blocks.z, scale );
			_print( ucMaskDup, mStream );
			//printf("before myErode_merge\n");

			int loops = 3;
			void *kernelArgs[] = {
				(void*)&(ucMask.mData),
				(void*)&(ucMaskDup.mData),
				(void*)&ucMaskDup.mWidth,
				(void*)&ucMaskDup.mHeight,
				(void*)&loops,
				(void*)&scale,
				(void*)&ucMask.mWidth, (void*)&ucMask.mHeight, (void*)&hostRect[0], (void*)&hostRect[2],
			};
			checkCudaErrors( cudaLaunchCooperativeKernel( (void*)myErode_cg, blocks, threads, kernelArgs, 0, mStream ) );
			// this resize just set width/etc parameters, no buffer reallocated, and mData pointer is the same
			ucMask.resize( ucMaskDup.mWidth, ucMaskDup.mHeight, ucMaskDup.mChannel, SCImageDataType_UC, SCImageOrder_Row );
			//printf("after  myErode_cg\n");
			_print( ucMask, mStream );
		}
#else
		{
		    nppUtil.copyROI( ucMaskDup, 0, 0,
		    ucMask, hostRect[0], hostRect[2],
		    ucMaskDup.mWidth, ucMaskDup.mHeight,
		    mStream );
		    ucMask.resize( ucMaskDup.mWidth, ucMaskDup.mHeight, ucMaskDup.mChannel, SCImageDataType_UC, SCImageOrder_Row );

			dim3 threads(16, 16);
			dim3 blocks(iDivUp(ucMaskDup.mWidth, threads.x), iDivUp(ucMaskDup.mHeight, threads.y));
			myErode<<<blocks, threads, 0, mStream>>>( ucMask.mData, ucMaskDup.mData, ucMaskDup.mWidth, ucMaskDup.mHeight, loops );
			myErode<<<blocks, threads, 0, mStream>>>( ucMaskDup.mData, ucMask.mData, ucMaskDup.mWidth, ucMaskDup.mHeight, loops );
			myErode<<<blocks, threads, 0, mStream>>>( ucMask.mData, ucMaskDup.mData, ucMaskDup.mWidth, ucMaskDup.mHeight, loops );
		}
#endif
	}
	leftTop = Point2D( centerPt.x-(ucMaskDup.mWidth>>1), centerPt.y-(ucMaskDup.mHeight>>1) );

    delete[] hostRect;
	//printf("endof %s\n", __FUNCTION__);
	_print( ucMask, mStream );
}

void SeamlessClone::init_resize()
{
    initMask( );

	gdX.resize( ucMask.mWidth, ucMask.mHeight, ucRGB_Patch.mChannel, SCImageDataType_Float, SCImageOrder_Row );
	gdY.resize( ucMask.mWidth, ucMask.mHeight, ucRGB_Patch.mChannel, SCImageDataType_Float, SCImageOrder_Row );
	lapXY.resize( ucMask.mWidth, ucMask.mHeight, ucRGB_Patch.mChannel, SCImageDataType_Float, SCImageOrder_Row );
	g.resize( lapXY.mWidth-2, lapXY.mHeight-2, lapXY.mChannel, SCImageDataType_Float, SCImageOrder_Row );
	u.resize( g.mWidth, g.mHeight, g.mChannel, SCImageDataType_Float, SCImageOrder_Row );
	u_.resize( g.mWidth, g.mHeight, g.mChannel, SCImageDataType_UC, SCImageOrder_Row );
	Vn1_1.resize( g.mHeight, g.mHeight, 1, SCImageDataType_Float, SCImageOrder_Row );
	Vn2_1.resize( g.mWidth, g.mWidth, 1, SCImageDataType_Float, SCImageOrder_Row );
	lambda_n1_1.resize( g.mHeight, 1, 1, SCImageDataType_Float, SCImageOrder_Row );
    lambda_n2_1.resize( g.mWidth,1, 1, SCImageDataType_Float, SCImageOrder_Row );
    filter_X.resize( g.mWidth, 1, 1, SCImageDataType_Float, SCImageOrder_Row );
    filter_Y.resize( g.mHeight,1, 1, SCImageDataType_Float, SCImageOrder_Row );
    tempComplex0.resize( (2 * g.mWidth + 2)*2, g.mHeight, 1, SCImageDataType_Float, SCImageOrder_Row );
    tempComplex1.resize( (2 * g.mHeight + 2)*2, g.mWidth, 1, SCImageDataType_Float, SCImageOrder_Row );
	//ucRGB_Output.resize( ucRGB_Bkgrd.mWidth, ucRGB_Bkgrd.mHeight, ucRGB_Bkgrd.mChannel, SCImageDataType_UC, SCImageOrder_Row );

#if SC_Test
	lambda_matrix_n2_1.resize( g.mWidth, g.mWidth, 1, SCImageDataType_Float, SCImageOrder_Row );
#endif

	initDSTMatrix_resize( (float*)Vn1_1.mData, (float*)Vn2_1.mData,
			(float*)lambda_n1_1.mData, (float*)lambda_n2_1.mData,
			(float*)lambda_matrix_n2_1.mData,
			(float*)filter_X.mData, (float*)filter_Y.mData,
			g.mWidth, g.mHeight);
#if SC_Test
	_print( lambda_matrix_n2_1, mStream );
#endif
	//ucRGB_Output.copyFrom( ucRGB_Bkgrd.mData, mStream );

	SCImage images[12] = {  u, g, Vn2_1,
                            g, Vn1_1, u,
                            u, g, Vn2_1,
                            g, Vn1_1, u};
#if true
    initBlas_resize( images, 12 );
#else
    initBlas( images, 12 );
#endif
}

void SeamlessClone::initBlas( SCImage* images, int imageNum )
{
	float* pptr[4*9];

	for( int blasIdx=0; blasIdx<4; blasIdx++ )
	{
		SCImage C = images[blasIdx*3+0];
		SCImage A = images[blasIdx*3+1];
		SCImage B = images[blasIdx*3+2];
		float** pptrA = pptr + blasIdx*9 + 3*0;
		float** pptrB = pptr + blasIdx*9 + 3*1;
		float** pptrC = pptr + blasIdx*9 + 3*2;

		for( int ch = 0; ch<C.mChannel; ch++ )
		{
			pptrA[ch] = (A.mChannel==3)?(float*)A.ptr(0, 0, ch):(float*)A.ptr(0, 0, 0);
			pptrB[ch] = (B.mChannel==3)?(float*)B.ptr(0, 0, ch):(float*)B.ptr(0, 0, 0);
			pptrC[ch] = (float*)C.ptr(0, 0, ch);
		}
        if( blasParamsA[blasIdx]==NULL ||
            blasParamsB[blasIdx]==NULL ||
            blasParamsC[blasIdx]==NULL)
        {
            assert( blasParamsA[blasIdx]==NULL &&
                    blasParamsB[blasIdx]==NULL &&
                    blasParamsC[blasIdx]==NULL );
            checkCudaErrors(cudaMalloc((void **)&blasParamsA[blasIdx], C.mChannel * sizeof(*blasParamsA[0])));
            checkCudaErrors(cudaMalloc((void **)&blasParamsB[blasIdx], C.mChannel * sizeof(*blasParamsB[0])));
            checkCudaErrors(cudaMalloc((void **)&blasParamsC[blasIdx], C.mChannel * sizeof(*blasParamsC[0])));
        }

		checkCudaErrors( cudaMemcpy(blasParamsA[blasIdx], pptrA, sizeof(*pptrA)*C.mChannel, cudaMemcpyHostToDevice) );
                checkCudaErrors( cudaMemcpy(blasParamsB[blasIdx], pptrB, sizeof(*pptrB)*C.mChannel, cudaMemcpyHostToDevice) );
                checkCudaErrors( cudaMemcpy(blasParamsC[blasIdx], pptrC, sizeof(*pptrC)*C.mChannel, cudaMemcpyHostToDevice) );
	}
}

__global__ void initBlas_kernel( unsigned char* pptr, int step,
                                    float* u, int u_sliceSz, int u_channel,
                                    float* g, int g_sliceSz, int g_channel,
                                    float* Vn2_1, int Vn2_1_sliceSz, int Vn2_1_channel,
                                    float* Vn1_1, int Vn1_1_sliceSz, int Vn1_1_channel)
{
    int x = threadIdx.x + blockDim.x*blockIdx.x;
    int y = threadIdx.y + blockDim.y*blockIdx.y;

    if( x==0 && y==0 )
    {
        {
            float** pptrA = (float**)(pptr+step*0);
            float** pptrB = (float**)(pptr+step*1);
            float** pptrC = (float**)(pptr+step*2);
            //u, g, Vn2_1<--
            //g, Vn1_1, u
            //u, g, Vn2_1
            //g, Vn1_1, u
            float* C = u;       /*int C_channel = u_channel;*/      int C_sliceSz = u_sliceSz;
            float* A = g;       int A_channel = g_channel;      int A_sliceSz = g_sliceSz;
            float* B = Vn2_1;   int B_channel = Vn2_1_channel;  int B_sliceSz = Vn2_1_sliceSz;
            for( int ch=0; ch<u_channel; ch++ )
            {
                pptrA[ch] = (A_channel==3)?(A+ch*A_sliceSz):A;
                pptrB[ch] = (B_channel==3)?(B+ch*B_sliceSz):B;
                pptrC[ch] =                (C+ch*C_sliceSz);
            }
        }

        {
            float** pptrA = (float**)(pptr+step*3);
            float** pptrB = (float**)(pptr+step*4);
            float** pptrC = (float**)(pptr+step*5);
            //u, g, Vn2_1
            //g, Vn1_1, u<--
            //u, g, Vn2_1
            //g, Vn1_1, u
            float* C = g;       /*int C_channel = g_channel;*/      int C_sliceSz = g_sliceSz;
            float* A = Vn1_1;   int A_channel = Vn1_1_channel;  int A_sliceSz = Vn1_1_sliceSz;
            float* B = u;       int B_channel = u_channel;      int B_sliceSz = u_sliceSz;
            for( int ch=0; ch<u_channel; ch++ )
            {
                pptrA[ch] = (A_channel==3)?(A+ch*A_sliceSz):A;
                pptrB[ch] = (B_channel==3)?(B+ch*B_sliceSz):B;
                pptrC[ch] =                (C+ch*C_sliceSz);
            }
        }

        {
            float** pptrA = (float**)(pptr+step*6);
            float** pptrB = (float**)(pptr+step*7);
            float** pptrC = (float**)(pptr+step*8);
            //u, g, Vn2_1
            //g, Vn1_1, u
            //u, g, Vn2_1<--
            //g, Vn1_1, u
            float* C = u;       /*int C_channel = u_channel;*/      int C_sliceSz = u_sliceSz;
            float* A = g;       int A_channel = g_channel;      int A_sliceSz = g_sliceSz;
            float* B = Vn2_1;   int B_channel = Vn2_1_channel;  int B_sliceSz = Vn2_1_sliceSz;
            for( int ch=0; ch<u_channel; ch++ )
            {
                pptrA[ch] = (A_channel==3)?(A+ch*A_sliceSz):A;
                pptrB[ch] = (B_channel==3)?(B+ch*B_sliceSz):B;
                pptrC[ch] =                (C+ch*C_sliceSz);
            }
        }

        {
            float** pptrA = (float**)(pptr+step*9);
            float** pptrB = (float**)(pptr+step*10);
            float** pptrC = (float**)(pptr+step*11);
            //u, g, Vn2_1
            //g, Vn1_1, u
            //u, g, Vn2_1
            //g, Vn1_1, u<--
            float* C = g;       /*int C_channel = g_channel;*/      int C_sliceSz = g_sliceSz;
            float* A = Vn1_1;   int A_channel = Vn1_1_channel;  int A_sliceSz = Vn1_1_sliceSz;
            float* B = u;       int B_channel = u_channel;      int B_sliceSz = u_sliceSz;
            for( int ch=0; ch<u_channel; ch++ )
            {
                pptrA[ch] = (A_channel==3)?(A+ch*A_sliceSz):A;
                pptrB[ch] = (B_channel==3)?(B+ch*B_sliceSz):B;
                pptrC[ch] =                (C+ch*C_sliceSz);
            }
        }
    }
}

void SeamlessClone::initBlas_resize( SCImage* images, int imageNum )
{
    const int STEP = 256;
    blasParams.resize( STEP*12, 1, 1, SCImageDataType_UC, SCImageOrder_Row );

    dim3 threads(1);
    dim3 blocks(1);
    initBlas_kernel<<<blocks, threads, 0, mStream>>>( blasParams.mData, STEP,
                        (float*)images[0].mData, images[0].mWidth*images[0].mHeight, images[0].mChannel, //u, g, Vn2_1, Vn1_1,
                        (float*)images[1].mData, images[1].mWidth*images[1].mHeight, images[1].mChannel,
                        (float*)images[2].mData, images[2].mWidth*images[2].mHeight, images[2].mChannel,
                        (float*)images[4].mData, images[4].mWidth*images[4].mHeight, images[4].mChannel);

    for( int blasIdx=0; blasIdx<4; blasIdx++ )
    {
        blasParamsA[blasIdx] = (float**)(blasParams.mData+STEP*3*blasIdx + STEP*0);
        blasParamsB[blasIdx] = (float**)(blasParams.mData+STEP*3*blasIdx + STEP*1);
        blasParamsC[blasIdx] = (float**)(blasParams.mData+STEP*3*blasIdx + STEP*2);
    }
}

// C = A*B
void SeamlessClone::ImageMultiplyPerSlice( SCImage& C, SCImage& A, SCImage& B, int blasIdx, cublasOperation_t transa )
{
	assert( A.mDType==SCImageDataType_Float &&
	     B.mDType==SCImageDataType_Float &&
	     C.mDType==SCImageDataType_Float );
	assert( A.mHeight==C.mHeight && B.mWidth==C.mWidth && A.mWidth==B.mHeight );
#if SC_Test
	assert( 3==C.mChannel || 1==C.mChannel );
#else
	assert( 3==C.mChannel );
#endif
	int M = A.mHeight, N = B.mWidth, K = B.mHeight;
	float alpha = 1.0f, beta = 0.0f;
#if true // as a batch of size 3

	cublasSgemmBatched( mCublasHandle, transa, CUBLAS_OP_N, M, N, K, &alpha, blasParamsA[blasIdx],
                       transa==CUBLAS_OP_T?K:M, blasParamsB[blasIdx], K, &beta, blasParamsC[blasIdx], M,
		       C.mChannel );
#else // slice by slice
	for( int ch = 0; ch<C.mChannel; ch++ )
	{
		float* ptrA = (A.mChannel==3)?(float*)A.ptr(0, 0, ch):(float*)A.ptr(0, 0, 0);
		float* ptrB = (B.mChannel==3)?(float*)B.ptr(0, 0, ch):(float*)B.ptr(0, 0, 0);
		float* ptrC = (float*)C.ptr(0, 0, ch);
		cublasSgemm( mCublasHandle, transa, CUBLAS_OP_N, M, N, K, &alpha, ptrA,
                       transa==CUBLAS_OP_T?K:M, ptrB, K, &beta, ptrC, M );
	}
#endif
	C.mOrder = SCImageOrder_Column;
}

__global__ void updateUij_kernel( float* srcDst, float* lambda1, float* lambda2, int width, int height, int channel )
{
	int x = threadIdx.x + blockDim.x*blockIdx.x;
	int y = threadIdx.y + blockDim.y*blockIdx.y;
	for( int ch=0; ch<channel; ch++ )
	{
		if( x<width && y<height )
		{
			int idx = x*height+y;
			srcDst[idx] /= (lambda1[y]+lambda2[x]);

			srcDst += width*height;
		}
	}
}

void updateUij( SCImage& g, SCImage& lambda1, SCImage& lambda2, cudaStream_t stream )
{
	dim3 threads(16, 16);
    	dim3 blocks(iDivUp(g.mWidth, threads.x), iDivUp(g.mHeight, threads.y));
	updateUij_kernel<<<blocks, threads, 0, stream>>>( (float*)g.mData,
			(float*)lambda1.mData, (float*)lambda2.mData,
			g.mWidth, g.mHeight, g.mChannel );
}

void SeamlessClone::poissonSolver2D( SCImage& g, SCImage& u )
{
	ImageMultiplyPerSlice( u, g, Vn2_1, 0, CUBLAS_OP_T );
	//_print( u, mStream );
	ImageMultiplyPerSlice( g, Vn1_1, u, 1 );
	//_print( g, mStream );
	updateUij( g, lambda_n1_1, lambda_n2_1, mStream );
	//_print( g, mStream );
	ImageMultiplyPerSlice( u, g, Vn2_1, 2 );
	//_print( u, mStream );
	ImageMultiplyPerSlice( g, Vn1_1, u, 3 );
    //_print( g, mStream );
}

template<int SCALE>
__global__ void dft_kernel_0_template( SCComplex* dst, float* src,
                            int dstWidth, int dstHeight,
                            int srcWidth, int srcHeight,
                            int scale )
{
    /*src.copyTo(temp(Rect(1,0, src.cols, src.rows)));
    for(int j = 0 ; j < src.rows ; ++j)
    {
        float * tempLinePtr = temp.ptr<float>(j);
        const float * srcLinePtr = src.ptr<float>(j);
        for(int i = 0 ; i < src.cols ; ++i)
        {
            tempLinePtr[src.cols + 2 + i] = - srcLinePtr[src.cols - 1 - i];
        }
    }*/

    int X = threadIdx.x + blockDim.x*blockIdx.x;
    int Y = threadIdx.y + blockDim.y*blockIdx.y;

    for( int ys=0; ys<SCALE; ys++ )
    for( int xs=0; xs<SCALE; xs++ )
    {
        int x = blockDim.x*gridDim.x*xs + X;
        int y = blockDim.y*gridDim.y*ys + Y;
        if( x<dstWidth && y<dstHeight )
        {
            int idx = y*dstWidth+x;
            dst[idx].r = 0.0f;
            dst[idx].c = 0.0f;
            if( x>=1 && x<=srcWidth )
            {
                dst[idx].r = src[y*srcWidth+(x-1)];
            }
            else if( x>(srcWidth+1) )
            {
                dst[idx].r = -src[y*srcWidth+(2*srcWidth+2-1-x)];
            }
        }
    }
}

__global__ void dft_kernel_0( SCComplex* dst, float* src,
                            int dstWidth, int dstHeight,
                            int srcWidth, int srcHeight,
                            int scale )
{
    /*src.copyTo(temp(Rect(1,0, src.cols, src.rows)));
    for(int j = 0 ; j < src.rows ; ++j)
    {
        float * tempLinePtr = temp.ptr<float>(j);
        const float * srcLinePtr = src.ptr<float>(j);
        for(int i = 0 ; i < src.cols ; ++i)
        {
            tempLinePtr[src.cols + 2 + i] = - srcLinePtr[src.cols - 1 - i];
        }
    }*/

    int X = threadIdx.x + blockDim.x*blockIdx.x;
    int Y = threadIdx.y + blockDim.y*blockIdx.y;

    int y = Y;
    for( int ys=0; ys<scale; ys++ )
    {
        int x = X;
        for( int xs=0; xs<scale; xs++ )
        {
            if( x<dstWidth && y<dstHeight )
            {
                int idx = y*dstWidth+x;
                dst[idx].r = 0.0f;
                dst[idx].c = 0.0f;
                if( x>=1 && x<=srcWidth )
                {
                    dst[idx].r = src[y*srcWidth+(x-1)];
                }
                else if( x>(srcWidth+1) )
                {
                    dst[idx].r = -src[y*srcWidth+(2*srcWidth+2-1-x)];
                }
            }
            x += blockDim.x*gridDim.x;
        }
        y += blockDim.y*gridDim.y;
    }
}

void dft_kernel_0_func( SCComplex* dst, float* src,
                            int dstWidth, int dstHeight,
                            int srcWidth, int srcHeight,
                            int scale,
                            dim3& blocks, dim3& threads, int sharedMemSz, cudaStream_t stream)
{
    void *kernelArgs[] = {
                (void*)&(dst),
                (void*)&(src),
                (void*)&dstWidth,
                (void*)&dstHeight,
                (void*)&srcWidth,
                (void*)&srcHeight,
                (void*)&scale,
    };
    switch( scale )
    {
        case 1:
            checkCudaErrors( cudaLaunchKernel( (void*)dft_kernel_0_template<1>, blocks, threads, kernelArgs, sharedMemSz, stream ) );
            break;
        case 2:
            checkCudaErrors( cudaLaunchKernel( (void*)dft_kernel_0_template<2>, blocks, threads, kernelArgs, sharedMemSz, stream ) );
            break;
        case 4:
            checkCudaErrors( cudaLaunchKernel( (void*)dft_kernel_0_template<4>, blocks, threads, kernelArgs, sharedMemSz, stream ) );
            break;
        case 8:
            checkCudaErrors( cudaLaunchKernel( (void*)dft_kernel_0_template<8>, blocks, threads, kernelArgs, sharedMemSz, stream ) );
            break;
        default:
            checkCudaErrors( cudaLaunchKernel( (void*)dft_kernel_0            , blocks, threads, kernelArgs, sharedMemSz, stream ) );
            break;
    }
}

template <int SCALE>
__global__ void dft_kernel_1_template( SCComplex* dst, SCComplex* src,
                            int dstWidth, int dstHeight,
                            int srcWidth, int srcHeight,
                            int Width, int Height,
                            int scale )
{
    /*
    split(complex, planes);
    temp = Mat::zeros(src.cols, 2 * src.rows + 2, CV_32F);
    for(int j = 0 ; j < src.cols ; ++j)
    {
        float * tempLinePtr = temp.ptr<float>(j);
        for(int i = 0 ; i < src.rows ; ++i)
        {
            float val = planes[1].ptr<float>(i)[j + 1];
            tempLinePtr[i + 1] = val;
            tempLinePtr[temp.cols - 1 - i] = - val;
        }
    }
    */
    int X = threadIdx.x + blockDim.x*blockIdx.x;
    int Y = threadIdx.y + blockDim.y*blockIdx.y;

    for( int ys=0; ys<SCALE; ys++ )
    for( int xs=0; xs<SCALE; xs++ )
    {
        int x = blockDim.x*gridDim.x*xs + X;
        int y = blockDim.y*gridDim.y*ys + Y;
        if( x<dstWidth && y<dstHeight )
        {
            int idx = y*dstWidth+x;
            if( x>0 && x<=Height )// Height = (dstWidth-2)/2
            {
                float val = src[(x-1)*srcWidth + (y+1)].c;
                dst[idx].r = val;

            }
            else if( x>=Height+2 && x<dstWidth )
            {
                float val = src[(dstWidth-1-x)*srcWidth + (y+1)].c;
                dst[idx].r = -val;
            }
            else
            {
                dst[idx].r = 0.0f;
            }
            dst[idx].c = 0;
        }
    }
}

__global__ void dft_kernel_1( SCComplex* dst, SCComplex* src,
                            int dstWidth, int dstHeight,
                            int srcWidth, int srcHeight,
                            int Width, int Height,
                            int scale )
{
    /*
    split(complex, planes);
    temp = Mat::zeros(src.cols, 2 * src.rows + 2, CV_32F);
    for(int j = 0 ; j < src.cols ; ++j)
    {
        float * tempLinePtr = temp.ptr<float>(j);
        for(int i = 0 ; i < src.rows ; ++i)
        {
            float val = planes[1].ptr<float>(i)[j + 1];
            tempLinePtr[i + 1] = val;
            tempLinePtr[temp.cols - 1 - i] = - val;
        }
    }
    */
    int X = threadIdx.x + blockDim.x*blockIdx.x;
    int Y = threadIdx.y + blockDim.y*blockIdx.y;

    int y = Y;
    for( int ys=0; ys<scale; ys++ )
    {
        int x = X;
        for( int xs=0; xs<scale; xs++ )
        {
            if( x<dstWidth && y<dstHeight )
            {
                int idx = y*dstWidth+x;
                if( x>0 && x<=Height )// Height = (dstWidth-2)/2
                {
                    float val = src[(x-1)*srcWidth + (y+1)].c;
                    dst[idx].r = val;

                }
                else if( x>=Height+2 && x<dstWidth )
                {
                    float val = src[(dstWidth-1-x)*srcWidth + (y+1)].c;
                    dst[idx].r = -val;
                }
                else
                {
                    dst[idx].r = 0.0f;
                }
                dst[idx].c = 0;
            }
            x += blockDim.x*gridDim.x;
        }
        y += blockDim.y*gridDim.y;
    }
}

void dft_kernel_1_func( SCComplex* dst, SCComplex* src,
                            int dstWidth, int dstHeight,
                            int srcWidth, int srcHeight,
                            int Width, int Height,
                            int scale,
                            dim3& blocks, dim3& threads, int sharedMemSz, cudaStream_t stream)
{
    void *kernelArgs[] = {
                (void*)&(dst),
                (void*)&(src),
                (void*)&dstWidth,
                (void*)&dstHeight,
                (void*)&srcWidth,
                (void*)&srcHeight,
                (void*)&Width,
                (void*)&Height,
                (void*)&scale,
    };
    switch( scale )
    {
        case 1:
            checkCudaErrors( cudaLaunchKernel( (void*)dft_kernel_1_template<1>, blocks, threads, kernelArgs, sharedMemSz, stream ) );
            break;
        case 2:
            checkCudaErrors( cudaLaunchKernel( (void*)dft_kernel_1_template<2>, blocks, threads, kernelArgs, sharedMemSz, stream ) );
            break;
        case 4:
            checkCudaErrors( cudaLaunchKernel( (void*)dft_kernel_1_template<4>, blocks, threads, kernelArgs, sharedMemSz, stream ) );
            break;
        case 8:
            checkCudaErrors( cudaLaunchKernel( (void*)dft_kernel_1_template<8>, blocks, threads, kernelArgs, sharedMemSz, stream ) );
            break;
        default:
            checkCudaErrors( cudaLaunchKernel( (void*)dft_kernel_1            , blocks, threads, kernelArgs, sharedMemSz, stream ) );
            break;
    }
}

__global__ void fft_copy_real_to_complex( SCComplex* complex, float* real,
                                        int realWidth, int realHeight,
                                        int scale)
{
    int X = threadIdx.x + blockDim.x*blockIdx.x;
    int Y = threadIdx.y + blockDim.y*blockIdx.y;

    for( int ys=0; ys<scale; ys++ )
    for( int xs=0; xs<scale; xs++ )
    {
        int x = blockDim.x*gridDim.x*xs + X;
        int y = blockDim.y*gridDim.y*ys + Y;
        if( x<realWidth && y<realHeight )
        {
            complex[y*realWidth+x].r = real[y*realWidth+x];
            complex[y*realWidth+x].c = 0;
        }
    }
}

__global__ void dft_copy_kernel_transpose( float* dst, int dstWidth, int dstHeight,
                                       float* src, int srcWidth, int srcHeight,
                                       int scale)
{
    int X = threadIdx.x + blockDim.x*blockIdx.x;
    int Y = threadIdx.y + blockDim.y*blockIdx.y;

    for( int ys=0; ys<scale; ys++ )
    for( int xs=0; xs<scale; xs++ )
    {
        int x = blockDim.x*gridDim.x*xs + X;
        int y = blockDim.y*gridDim.y*ys + Y;
        if( x<dstWidth && y<dstHeight )
        {
            dst[y*dstWidth+x] = src[x*srcWidth+(y+1)*2+1];
        }
    }
}

__global__ void updateUij_kernel_fft( float* dst, int dstWidth, int dstHeight,
		                              float* filter_X, float* filter_Y,
		                              int scale )
{
    /*
    for(int j = 0 ; j < h-2; j++)
    {
        float * resLinePtr = res.ptr<float>(j);
        for(int i = 0 ; i < w-2; i++)
        {
            resLinePtr[i] /= (filter_X[i] + filter_Y[j] - 4);
        }
    }
    */
    int X = threadIdx.x + blockDim.x*blockIdx.x;
    int Y = threadIdx.y + blockDim.y*blockIdx.y;

    for( int ys=0; ys<scale; ys++ )
    for( int xs=0; xs<scale; xs++ )
    {
        int x = blockDim.x*gridDim.x*xs + X;
        int y = blockDim.y*gridDim.y*ys + Y;
        if( x<dstWidth && y<dstHeight )
        {
            dst[y*dstWidth+x] /= (filter_X[x] + filter_Y[y] - 4);
        }
    }
}


__global__ void scale_kernel_fft_transpose( float* dst, float* src, int Width, int Height,
                                  float ratio,
		                          int scale )
{
    int X = threadIdx.x + blockDim.x*blockIdx.x;
    int Y = threadIdx.y + blockDim.y*blockIdx.y;

    for( int ys=0; ys<scale; ys++ )
    for( int xs=0; xs<scale; xs++ )
    {
        int x = blockDim.x*gridDim.x*xs + X;
        int y = blockDim.y*gridDim.y*ys + Y;
        if( x<Width && y<Height )
        {
            // scale and transpose to column-order
            float t = src[y*Width+x] * ratio;
            dst[x*Height+y] = t;
        }
    }
}

//void SeamlessClone::dst(const Mat& src, Mat& dest, bool invert)
void SeamlessClone::dst(SCImage& src, SCImage& dest, bool invert/*=false*/, int channelIdx)
{
    //Mat temp = Mat::zeros(src.rows, 2 * src.cols + 2, CV_32F);

    tempComplex0.resize( (2 * src.mWidth + 2)*2, src.mHeight, 1, SCImageDataType_Float, SCImageOrder_Row );

    //int flag = invert ? DFT_ROWS + DFT_SCALE + DFT_INVERSE: DFT_ROWS;
    int flag = invert ? CUFFT_INVERSE:CUFFT_FORWARD;

    /*src.copyTo(temp(Rect(1,0, src.cols, src.rows)));
    for(int j = 0 ; j < src.rows ; ++j)
    {
        float * tempLinePtr = temp.ptr<float>(j);
        const float * srcLinePtr = src.ptr<float>(j);
        for(int i = 0 ; i < src.cols ; ++i)
        {
            tempLinePtr[src.cols + 2 + i] = - srcLinePtr[src.cols - 1 - i];
        }
    }*/
    {
        dim3 threads(16, 16);
        int scale = 1;
        dim3 blocks = calCooperativeGroupBlocks( (void*)dft_kernel_0, 0, scale, threads, tempComplex0.mWidth>>1, tempComplex0.mHeight, mSM );
        /*dft_kernel_0<<<blocks, threads, 0, mStream>>>( (SCComplex*)tempComplex0.mData, (float*)src.mData,
                    tempComplex0.mWidth>>1, tempComplex0.mHeight,
                    src.mWidth, src.mHeight,
                    scale );*/
        dft_kernel_0_func( (SCComplex*)tempComplex0.mData, (float*)src.mData,
                    tempComplex0.mWidth>>1, tempComplex0.mHeight,
                    src.mWidth, src.mHeight,
                    scale,
                    blocks, threads, 0, mStream);
    }

    /*Mat planes[] = {temp, Mat::zeros(temp.size(), CV_32F)};
    Mat complex;
    merge(planes, 2, complex);
    dft(complex, complex, flag);*/

    {
        int nx = tempComplex0.mHeight, ny=(tempComplex0.mWidth>>1);
        if( !fftParams0.isEqualTo(ny, CUFFT_C2C, nx) )
        {
            cufftDestroy(fftParams0.plan);
            cufftHandle plan;
            cufftPlan1d(&plan, ny, CUFFT_C2C, nx);
            cufftSetStream(plan, mStream);
            fftParams0 = FFTParams( ny, CUFFT_C2C, nx, plan );
        }
        cufftExecC2C(fftParams0.plan, (cufftComplex *)tempComplex0.mData, (cufftComplex *)tempComplex0.mData, flag);
    }

    /*split(complex, planes);
    temp = Mat::zeros(src.cols, 2 * src.rows + 2, CV_32F);
    for(int j = 0 ; j < src.cols ; ++j)
    {
        float * tempLinePtr = temp.ptr<float>(j);
        for(int i = 0 ; i < src.rows ; ++i)
        {
            float val = planes[1].ptr<float>(i)[j + 1];
            tempLinePtr[i + 1] = val;
            tempLinePtr[temp.cols - 1 - i] = - val;
        }
    }
    */

    tempComplex1.resize( (2 * src.mHeight + 2)*2, src.mWidth, 1, SCImageDataType_Float, SCImageOrder_Row );
    {
        dim3 threads(16, 16);
        int scale = 1;
        dim3 blocks = calCooperativeGroupBlocks( (void*)dft_kernel_1, 0, scale, threads, tempComplex1.mWidth>>1, tempComplex1.mHeight, mSM );
        /*dft_kernel_1<<<blocks, threads, 0, mStream>>>( (SCComplex*)tempComplex1.mData, (SCComplex*)tempComplex0.mData,
                    tempComplex1.mWidth>>1, tempComplex1.mHeight,
                    tempComplex0.mWidth>>1, tempComplex0.mHeight,
                    src.mWidth, src.mHeight,
                    scale);*/
        dft_kernel_1_func( (SCComplex*)tempComplex1.mData, (SCComplex*)tempComplex0.mData,
                    tempComplex1.mWidth>>1, tempComplex1.mHeight,
                    tempComplex0.mWidth>>1, tempComplex0.mHeight,
                    src.mWidth, src.mHeight,
                    scale,
                    blocks, threads, 0, mStream);
    }
    /*
    Mat planes2[] = {temp, Mat::zeros(temp.size(), CV_32F)};
    merge(planes2, 2, complex);
    dft(complex, complex, flag);
    */

    SWAP<SCImage>( tempComplex1, tempComplex0 );
    {
        int nx = tempComplex0.mHeight, ny=(tempComplex0.mWidth>>1);
        if( !fftParams1.isEqualTo(ny, CUFFT_C2C, nx) )
        {
            cufftDestroy(fftParams1.plan);
            cufftHandle plan;
            cufftPlan1d(&plan, ny, CUFFT_C2C, nx);
            cufftSetStream(plan, mStream);
            fftParams1 = FFTParams( ny, CUFFT_C2C, nx, plan );
        }
        cufftExecC2C(fftParams1.plan, (cufftComplex *)tempComplex0.mData, (cufftComplex *)tempComplex0.mData, flag);
    }

    /*
    split(complex, planes2);
    temp = planes2[1].t();
    temp(Rect( 0, 1, src.cols, src.rows)).copyTo(dest);
    */

    {
        dim3 threads(16, 16);
        int scale = 1;
        dim3 blocks = calCooperativeGroupBlocks( (void*)dft_copy_kernel_transpose, 0, scale, threads, dest.mWidth, dest.mHeight, mSM );
        dft_copy_kernel_transpose<<<blocks, threads, 0, mStream>>>( (float*)dest.mData, dest.mWidth, dest.mHeight,
                           (float*)tempComplex0.mData, tempComplex0.mWidth, tempComplex0.mHeight,
                           scale);
    }
}

//void SeamlessClone::solve(const Mat &img, Mat& mod_diff, Mat &result)
void SeamlessClone::solve(SCImage& g, SCImage& u, int channelIdx )
{
    /*const int w = img.cols;
    const int h = img.rows;

    Mat res;
    dst(mod_diff, res);
    */
    dst(g, u, false, channelIdx);

    /*
    for(int j = 0 ; j < h-2; j++)
    {
        float * resLinePtr = res.ptr<float>(j);
        for(int i = 0 ; i < w-2; i++)
        {
            resLinePtr[i] /= (filter_X[i] + filter_Y[j] - 4);
        }
    }
    */
    {
        dim3 threads(16, 16);
		int scale = 1;
		dim3 blocks = calCooperativeGroupBlocks( (void*)updateUij_kernel_fft, 0, scale, threads, u.mWidth, u.mHeight, mSM );
		updateUij_kernel_fft<<<blocks, threads, 0, mStream>>>( (float*)u.mData, u.mWidth, u.mHeight,
		                                                        (float*)filter_X.mData, (float*)filter_Y.mData,
		                                                         scale);
    }

    //dst(res, mod_diff, true);
    dst(u, g, true, channelIdx);

    /*
    unsigned char *  resLinePtr = result.ptr<unsigned char>(0);
    const unsigned char * imgLinePtr = img.ptr<unsigned char>(0);
    const float * interpLinePtr = NULL;

     //first col
    for(int i = 0 ; i < w ; ++i)
        result.ptr<unsigned char>(0)[i] = img.ptr<unsigned char>(0)[i];

    for(int j = 1 ; j < h-1 ; ++j)
    {
        resLinePtr = result.ptr<unsigned char>(j);
        imgLinePtr  = img.ptr<unsigned char>(j);
        interpLinePtr = mod_diff.ptr<float>(j-1);

        //first row
        resLinePtr[0] = imgLinePtr[0];

        for(int i = 1 ; i < w-1 ; ++i)
        {
            //saturate cast is not used here, because it behaves differently from the previous implementation
            //most notable, saturate_cast rounds before truncating, here it's the opposite.
            float value = interpLinePtr[i-1];
            if(value < 0.)
                resLinePtr[i] = 0;
            else if (value > 255.0)
                resLinePtr[i] = 255;
            else
                resLinePtr[i] = static_cast<unsigned char>(value);
        }

        //last row
        resLinePtr[w-1] = imgLinePtr[w-1];
    }

    //last col
    resLinePtr = result.ptr<unsigned char>(h-1);
    imgLinePtr = img.ptr<unsigned char>(h-1);
    for(int i = 0 ; i < w ; ++i)
        resLinePtr[i] = imgLinePtr[i];
    */
    //SWAP<SCImage>( g, u );
    {
        dim3 threads(16, 16);
		int scale = 1;
		dim3 blocks = calCooperativeGroupBlocks( (void*)scale_kernel_fft_transpose, 0, scale, threads, g.mWidth, g.mHeight, mSM );
		scale_kernel_fft_transpose<<<blocks, threads, 0, mStream>>>( (float*)u.mData, (float*)g.mData, g.mWidth, g.mHeight,
                                                             1.0f/((g.mWidth*2+2)*(g.mHeight*2+2)), // two dft forward and backward,
                                                             scale);
    }
}

void SeamlessClone::poissonSolver2D_FFT( SCImage& g, SCImage& u )
{
    for( int i=0; i<g.mChannel; i++ )
    {
        SCImage gslice, uslice;
        {
            int sliceSz = g.mWidth*g.mHeight*g.elmtSz();
            gslice = SCImage( g.mData + i*sliceSz, g.mWidth, g.mHeight, 1, g.mDType, g.mOrder, -1 );
        }
        {
            int sliceSz = u.mWidth*u.mHeight*u.elmtSz();
            uslice = SCImage( u.mData + i*sliceSz, u.mWidth, u.mHeight, 1, u.mDType, u.mOrder, -1 );
        }
        solve( gslice, uslice, i );
        _print(gslice, mStream);
    }
    // output is in u, so swap it with g.
    SWAP<SCImage>(g, u);
    // make it same column-order format as in poissonSolver2D
    g.mOrder = SCImageOrder_Column;
}

__global__ void pre_process_kernel_gradient( float* lapXY,
		unsigned char* bkgrd, int bkgrdWidth, int bkgrdHeight, int bkgrdPitch,
		float* gdX, float* gdY,
		unsigned char* patch, int patchWidth, int patchHeight, int patch_offsetX, int patch_offsetY,
		unsigned char* mask,
		int width, int height, int channel,
	        int ltX, int ltY )
{
	int x = threadIdx.x + blockDim.x*blockIdx.x;
    int y = threadIdx.y + blockDim.y*blockIdx.y;

	for( int ch=0; ch<channel; ch++ )
	{
		if(x<width && y<height)
		{
			unsigned char* ptr0 = bkgrd + (y+ltY)*bkgrdWidth + x+ltX;
			float np;
			if((x)<(width-1)) np = (float)(ptr0[1]); else np = (float)ptr0[-1]; // BORDER_DEFAULT
			float gdX_bk = (np-(float)ptr0[0]);

			if((y)<(height-1)) np = (float)(ptr0[bkgrdPitch]); else np = (float)ptr0[-bkgrdPitch]; // BORDER_DEFAULT
			float gdY_bk = (np-(float)ptr0[0]);

			unsigned char* ptr = patch + (y+patch_offsetY)*patchWidth + (x+patch_offsetX);
			if( x<width-1 ) np = (float)(ptr[1]); else np = (float)ptr[-1]; // BORDER_DEFAULT
                        float gdX_pt = (np-(float)ptr[0]);

			if( y<height-1 ) np = (float)(ptr[patchWidth]); else np = (float)ptr[-patchWidth]; // BORDER_DEFAULT
                        float gdY_pt = (np-(float)ptr[0]);

			float msk = (float)mask[y*width+x] * (1.0f/255.0f);
			float _gdX, _gdY;
			_gdX = (1.0f-msk)*gdX_bk + msk*gdX_pt;
			_gdY = (1.0f-msk)*gdY_bk + msk*gdY_pt;

			gdX[y*width+x] = _gdX;
			gdY[y*width+x] = _gdY;

			bkgrd += bkgrdPitch*bkgrdHeight;
                	patch += patchWidth*patchHeight;
                	gdX   += width*height;
                	gdY   += width*height;
		}
	}
}

__global__ void pre_process_kernel_lapXY( float* lapXY, float* g,
                unsigned char* bkgrd, int bkgrdWidth, int bkgrdHeight, int bkgrdPitch,
                float* gdX, float* gdY,
                unsigned char* patch,
                unsigned char* mask,
                int width, int height, int channel,
                int ltX, int ltY )
{
    int x = threadIdx.x + blockDim.x*blockIdx.x;
    int y = threadIdx.y + blockDim.y*blockIdx.y;

    for( int ch=0; ch<channel; ch++ )
    {
        if( x<width && y<height )
        {
            float lap = 0.0f;
            //lapXY[y*width+x] = 0.0f;
            if(x>=1 && x<=(width-2) && y>=1 && y<=(height-2))
            {
                float* gdX_ptr = gdX + y*width + x;
                float* gdY_ptr = gdY + y*width + x;
                float _gdX = gdX_ptr[0]-gdX_ptr[-1];
                float _gdY = gdY_ptr[0]-gdY_ptr[-width];
                unsigned char* ptr = bkgrd + (y+ltY)*bkgrdWidth + x+ltX;
                lap += _gdX + _gdY;

                if( x==1 )
                {
                    lap -= ((float)ptr[-1]);
                }
                if( y==1 )
                {
                    lap -= ((float)ptr[-bkgrdWidth]);
                }
                if( x==(width-2) )
                {
                    lap -= ((float)ptr[1]);
                }
                if( y==(height-2) )
                {
                    lap -= ((float)ptr[bkgrdWidth]);
                }
                g[(y-1)*(width-2)+x-1] = lap;
                lapXY[y*width+x] = lap;
            }
            bkgrd += bkgrdPitch*bkgrdHeight;
            lapXY += width*height;
            gdX   += width*height;
            gdY   += width*height;
            g     += (width-2)*(height-2);
        }
    }
}

dim3 calCooperativeGroupBlocks( void* func, int sharedMem, int& scale, dim3& threads, int width, int height, int SM )
{
	dim3 blocks(iDivUp(width, threads.x), iDivUp(height, threads.y));

	int maxBlocks = 0;
	checkCudaErrors(cudaOccupancyMaxActiveBlocksPerMultiprocessor ( &maxBlocks, func, threads.x*threads.y, sharedMem ));
	maxBlocks *= SM;
	while( maxBlocks<blocks.x*blocks.y )
	{
		blocks.x = ((blocks.x+1)>>1);
		blocks.y = ((blocks.y+1)>>1);
		scale *= 2;
	}

	return blocks;
}

void SeamlessClone::pre_process_v2()
{
#if SC_Enable_Cooperative_Group
    {
		dim3 threads(16, 16);
		int scale = 1;
		dim3 blocks = calCooperativeGroupBlocks( (void*)pre_process_kernel_cg, 0, scale, threads, lapXY.mWidth, lapXY.mHeight, mSM );
		pre_process_kernel_cg<<<blocks, threads, 0, mStream>>>(
				(float*)lapXY.mData, (float*)g.mData,
				ucRGB_Bkgrd.mData, ucRGB_Bkgrd.mWidth, ucRGB_Bkgrd.mHeight, ucRGB_Bkgrd.pitch(),
				(float*)gdX.mData, (float*)gdY.mData,
				ucRGB_Patch.mData, ucRGB_Patch.mWidth, ucRGB_Patch.mHeight, mPatchOffset.x, mPatchOffset.y,
				ucMask.mData,
				lapXY.mWidth, lapXY.mHeight, lapXY.mChannel,
				leftTop.x, leftTop.y,
				scale
				);
	}
#else
	{
		dim3 threads(16, 16);
        	dim3 blocks(iDivUp(lapXY.mWidth, threads.x), iDivUp(lapXY.mHeight, threads.y));
		pre_process_kernel_gradient<<<blocks, threads, 0, mStream>>>( (float*)lapXY.mData,
				ucRGB_Bkgrd.mData, ucRGB_Bkgrd.mWidth, ucRGB_Bkgrd.mHeight, ucRGB_Bkgrd.pitch(),
				(float*)gdX.mData, (float*)gdY.mData,
				ucRGB_Patch.mData, ucRGB_Patch.mWidth, ucRGB_Patch.mHeight, mPatchOffset.x, mPatchOffset.y,
				ucMask.mData,
				lapXY.mWidth, lapXY.mHeight, lapXY.mChannel,
				leftTop.x, leftTop.y);

		pre_process_kernel_lapXY<<<blocks, threads, 0, mStream>>>( (float*)lapXY.mData, (float*)g.mData,
				ucRGB_Bkgrd.mData, ucRGB_Bkgrd.mWidth, ucRGB_Bkgrd.mHeight, ucRGB_Bkgrd.pitch(),
				(float*)gdX.mData, (float*)gdY.mData,
				ucRGB_Patch.mData,
				ucMask.mData,
				lapXY.mWidth, lapXY.mHeight, lapXY.mChannel,
				leftTop.x, leftTop.y);
	}
#endif
}

__global__ void post_processing( unsigned char* ucRGB_Output, int offsetX, int offsetY,
				int oWidth, int oHeight, int oChannel,
				float* g, int gWidth, int gHeight,
				float scale,
				unsigned char* u_)
{
	int x = threadIdx.x + blockDim.x*blockIdx.x;
    int y = threadIdx.y + blockDim.y*blockIdx.y;

    for( int ch=0; ch<oChannel; ch++ )
    {
        if( x<gWidth && y<gHeight )
        {
            float d = g[x*gHeight+y]*scale;
            d = d>255.0f?255.0f:d;
            d = d<0.0f?0.0f:d;
            unsigned char uc = d;
            ucRGB_Output[(offsetY+y)*oWidth+(offsetX+x)] = uc;
            u_[(y*gWidth+x)*oChannel+oChannel-ch-1] = uc; // RGBI

            ucRGB_Output += oWidth*oHeight;
            g += gWidth*gHeight;
            //u_+= gWidth*gHeight;
        }
	}
}

void SeamlessClone::run()
{
    init_resize();

	//ucRGB_Output.copyFrom( ucRGB_Bkgrd.mData, mStream );
	ucMask.write2Yaml2("ucMask", mStream, true);
	pre_process_v2();
#if SC_Test
	testDST( g, u );
#endif
#if SCDEBUG
	g.write2Yaml2("g", mStream, true);
#endif

#if SC_FFT_ENABLE
	poissonSolver2D_FFT( g, u );
#else
	poissonSolver2D( g, u );
#endif
	_print( g, mStream );
	{
		dim3 threads(16, 16);
	        dim3 blocks(iDivUp(g.mWidth, threads.x), iDivUp(g.mHeight, threads.y));
		post_processing<<<blocks, threads, 0, mStream>>>( ucRGB_Output.mData, leftTop.x+1, leftTop.y+1,
				ucRGB_Output.mWidth, ucRGB_Output.mHeight, ucRGB_Output.mChannel,
			       	(float*)g.mData, g.mWidth, g.mHeight,
				    1.0f/(MY_PRECISION_SCALE*MY_PRECISION_SCALE*MY_PRECISION_SCALE),
				    u_.mData);
	}
	return;
}

#endif
