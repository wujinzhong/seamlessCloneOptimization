#ifndef __SEAMLESS_CLONE_COMMON_H__
#define __SEAMLESS_CLONE_COMMON_H__

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/photo.hpp"
#include <iostream>
#include <cublas_v2.h>
#include <vector>

#define SCDEBUG true
#define SC_Enable_Cooperative_Group false
#define SC_FFT_ENABLE true

#define PI 3.14159265358979323846f
//#define PI 3.1415926f
//#define PI 3.141593f
//#define M_PI 3.141592653589793238462643383279502884197169399375105820974944592308
//#define M_PI_F 3.14159265358979323846f

#define RELEASE_DEV_PTR( ptr ) 		\
{					\
	if(ptr!=NULL)			\
		cudaFree(ptr);		\
	ptr = NULL;			\
}

#define case_print(error) case error: printf("%s\n", #error); break;
#if SCDEBUG
	#define _print( d, stream ) { printf("\n\n--------------%s------------------\n", #d); d.print(stream); }
#else
	#define _print( d, stream ) 
#endif

template<class DType>
void SWAP( DType& a, DType& b )
{
	DType tmp = a;
        a = b;
        b = tmp;
}

using namespace cv;

#define write2Yaml_( mat ) write2Yaml( "./images/", #mat, mat )

#if SCDEBUG
void write2Yaml( const char* file_dir, const char* mat_name, const Mat& mat )
{
	std::string file_path(file_dir);
	file_path.append( mat_name );
	file_path.append( ".yml" );
	printf( "writing to file %s, rows x cols: %4d x %4d, type %4d.\n", file_path.c_str(), mat.rows, mat.cols, mat.type() );

	cv::FileStorage fs(file_path.c_str(), cv::FileStorage::WRITE);
	fs<<"mat_name"<<mat_name;
	fs<<"data"<<mat;
}
#else
void write2Yaml( const char* file_dir, const char* mat_name, const Mat& mat ){}
#endif

Mat readFromYaml( const char* file_path )
{
	Mat mat;
	printf("begin reading file: %s\n", file_path);

	cv::FileStorage fs( file_path, cv::FileStorage::READ);
	fs["data"]>>mat;

    printf("mat shape: %d, %d, %d\n", mat.cols, mat.rows, mat.channels());

	return mat;
}

bool loadImageFromYml( const char* file_path, unsigned char** bufs, int* pitchs, int* widths, int* heights, cudaStream_t stream )
{
	Mat mat = readFromYaml( file_path );
	assert( mat.type()==CV_8UC3 || mat.type()==CV_8UC1);
        assert( mat.channels()==3 || mat.channels()==1);
	for( int ch=0; ch<mat.channels(); ch++ )
	{
		widths[ch] = mat.cols;
		heights[ch] = mat.rows;
		pitchs[ch] = mat.cols;
		checkCudaErrors( cudaMalloc( (void**)&bufs[ch], sizeof(unsigned char)*widths[ch]*heights[ch] ) );
	}

	unsigned char* outData = new unsigned char[mat.cols*mat.rows*mat.channels()];

	int nr=mat.rows;
    	for(int k=0;k<nr;k++)
    	{
    	    const uchar* inData=mat.ptr<uchar>(k);

	    for( int col=0; col<mat.cols; col++ )
	    {
    	    	for(int ch=0; ch<mat.channels(); ch++)
    	    	{
    	        	outData[ch*widths[ch]*heights[ch]+k*pitchs[ch]+col] = inData[col*mat.channels()+mat.channels()-1-ch];
    	    	}
	    }
    	}

	for( int ch=0; ch<mat.channels(); ch++ )
	{
		checkCudaErrors( cudaMemcpyAsync( bufs[ch], outData+ch*widths[ch]*heights[ch], sizeof(unsigned char)*widths[ch]*heights[ch], cudaMemcpyHostToDevice, stream ) );
	}

	delete[] outData;
	return true;
}

typedef struct Point2D{
	Point2D():x(0), y(0){}
	Point2D(int x_, int y_):x(x_), y(y_){}
	int x, y;
}Point2D;

typedef enum SCImageDataType_enum{
	SCImageDataType_None=0,
	SCImageDataType_UC,
	SCImageDataType_Float,
	SCImageDataType_Int32
}SCImageDataType;

typedef enum SCImageOrder_enum{
	SCImageOrder_Row = 0,
	SCImageOrder_Column
}SCImageOrder;

#define SCImageCapacity_Constant 2
typedef enum SCImageCapacityType_enum{
    SCImageCapacityType_None = 0,
    SCImageCapacityType_x1,
    SCImageCapacityType_xN // N==SCImageCapacity_Constant
}SCImageCapacityType;

class SCImage
{
public:
	SCImage():
		mData(NULL), mWidth(0), mHeight(0), mChannel(0), mDType(SCImageDataType_None), mOrder(SCImageOrder_Row)
		, mCapacity(0)
        {
        }

	SCImage( void* d, int w, int h, int c, SCImageDataType dt, SCImageOrder od/*=SCImageOrder_Row*/, int capacity/*=0*/ ):
		mData((unsigned char*)d), mWidth(w), mHeight(h), mChannel(c), mDType(dt), mOrder(od)
		, mCapacity(capacity)
	{
	    if( mCapacity==-1 )
	    {
	        mCapacity = w*h*c*elmtSz();
	    }
	}

private:
	static SCImage createImage( int w, int h, int c, SCImageDataType dt,
	SCImageOrder od/*=SCImageOrder_Row*/, SCImageCapacityType capacityType, bool bAddToOccupy=true )
	{
		unsigned char* ptr = NULL;
		int sz = h*c*w*elmtSz_(dt);
		switch( capacityType )
		{
		    case SCImageCapacityType_x1:
		        break;
		    case SCImageCapacityType_xN:
		        sz *= SCImageCapacity_Constant;
		        break;
		    default:
		        assert(false);
		        break;
		}

		checkCudaErrors( cudaMalloc((void**)&ptr, sz) );
		SCImage img((void*)ptr, w, h, c, dt, od, sz);
		//img.print();
		if( bAddToOccupy ) mOccupy += sz;
		return img;
	}

public:
	static int getTotalDeviceMemoryOccupy(){ return mOccupy; }
	static void resetOccupy(){ mOccupy=0; }

    void resize( int w, int h, int c, SCImageDataType dt, SCImageOrder od=SCImageOrder_Row )
    {
        int new_sz = h*c*w*elmtSz_(dt);
        if( new_sz>mCapacity )
        {
            destroy();
            *this = createImage( w, h, c, dt, SCImageOrder_Row, SCImageCapacityType_xN, true );
        }
        else
        {
            *this = SCImage( mData, w, h, c, dt, od, mCapacity );
        }
    }

	void destroy()
	{
		if( mData!=NULL )
		{
			checkCudaErrors( cudaFree(mData) );
			*this = SCImage();
		}
	}

	static int elmtSz_( SCImageDataType dt )
	{
		switch( dt )
                {
                        case SCImageDataType_UC:
                                return 1;
                                break;
                        case SCImageDataType_Float:
			case SCImageDataType_Int32:
                                return 4;
                                break;
                        default:
                                assert(false);
                                break;
                }
		return -1;
	}
	int elmtSz()
	{
		return elmtSz_(mDType);
	}

	int sz()
	{
		return mHeight*mWidth*elmtSz()*mChannel;
	}

    int sliceSz()
    {
        return mWidth*mHeight*elmtSz();
    }

	int pitch()
	{
		int pitch = 0;
		switch( mOrder )
		{
			case SCImageOrder_Row:
				pitch = mWidth*elmtSz();	
				break;
			case SCImageOrder_Column:
                                pitch = mHeight*elmtSz();
                                break;
		}
		return pitch;
	}

	void copyFrom( unsigned char* src, cudaStream_t stream )
	{
		checkCudaErrors( cudaMemcpyAsync( mData, src, sz(), cudaMemcpyDeviceToDevice,
				       stream	));
	}

	void splitTo( std::vector<SCImage>& imgs, cudaStream_t stream )
	{
	    imgs.resize( mChannel );
	    for( int i=0; i<mChannel; i++ )
	    {
	        imgs[i].resize( mWidth, mHeight, 1, mDType, mOrder );
	        int sliceSz = mWidth*mHeight*elmtSz();
	        checkCudaErrors( cudaMemcpyAsync( imgs[i].mData, mData+i*sliceSz, sliceSz,
	                    cudaMemcpyDeviceToDevice, stream ));
	    }
	}

	void mergeFrom( std::vector<SCImage> imgs, cudaStream_t stream )
	{
	    assert( mChannel==imgs.size() );
	    for( int i=0; i<mChannel; i++ )
	    {
	        assert( imgs[i].mChannel==1 );
	        assert( imgs[i].mWidth==mWidth &&
                    imgs[i].mHeight==mHeight &&
                    imgs[i].mDType==mDType &&
                    imgs[i].mOrder==mOrder);
            int sliceSz = mWidth*mHeight*elmtSz();
            checkCudaErrors( cudaMemcpyAsync( mData+i*sliceSz, imgs[i].mData, sliceSz,
	                        cudaMemcpyDeviceToDevice, stream ));
	    }
	}

	template<typename DType>
	DType AT(unsigned char* data, int w, int h, int c){ 
		int pos = 0;
		
		if( mOrder==SCImageOrder_Row ) 		    pos = sliceSz()*c+h*mWidth *elmtSz()+w*elmtSz();
		else if( mOrder==SCImageOrder_Column ) 	pos = sliceSz()*c+w*mHeight*elmtSz()+h*elmtSz();
		else assert( false );
		return *((DType*)(data+pos));
	}
private:
	unsigned char* ptr( unsigned char* data, int w, int h, int c )
	{
	    assert( mOrder==SCImageOrder_Row );
		int pos = sliceSz()*c+h*pitch()+w*elmtSz();
        return data+pos;
	}
public:
	unsigned char* ptr( int w, int h, int c )
        {
                assert( mOrder==SCImageOrder_Row );
                int pos = sliceSz()*c+h*pitch()+w*elmtSz();
                return mData+pos;
        }
	unsigned char* copyD2H( cudaStream_t stream )
	{
		unsigned char *tmpImg = NULL;
		int size = sz();
		tmpImg = new unsigned char[size];
		checkCudaErrors( cudaMemcpyAsync(tmpImg, mData, size, cudaMemcpyDeviceToHost, stream) );
		checkCudaErrors( cudaStreamSynchronize(stream) );
		return tmpImg;
	}
	void setConstant( int value, cudaStream_t stream )
	{
		checkCudaErrors( cudaMemsetAsync(mData, value, sz(), stream) );
	}

	void write2Yaml2( const char* name, cudaStream_t stream, bool bSlice )
	{
#if SCDEBUG
		assert( bSlice==true );
		unsigned char* tmpImg = copyD2H( stream );
		
		for( int ch=0; ch<mChannel; ch++ )
		{
			if( mDType==SCImageDataType_UC )
			{
				unsigned char* ptr = tmpImg + mWidth*mHeight*ch;
				Mat mat = Mat::zeros( mHeight, mWidth, CV_8UC1 );
				for( int row=0; row<mHeight; row++ )
				{
					unsigned char* matPtr = mat.ptr<unsigned char>(row);
					for( int x=0; x<mWidth; x++ )
					{
						matPtr[x] = ptr[row*mWidth+x];
					}
				}
				std::stringstream mat_name; mat_name<<name<<ch;
				
				write2Yaml( "./output/", mat_name.str().c_str(), mat );
			}
			else if( mDType==SCImageDataType_Float )
                        {
				float* ptr = ((float*)tmpImg) + mWidth*mHeight*ch;
                                Mat mat = Mat::zeros( mHeight, mWidth, CV_32FC1 );
                                for( int row=0; row<mHeight; row++ )
                                {
                                        float* matPtr = mat.ptr<float>(row);
                                        for( int x=0; x<mWidth; x++ )
                                        {
                                                matPtr[x] = ptr[row*mWidth+x];
                                        }
                                }
                                std::stringstream mat_name; mat_name<<name<<ch;

                                write2Yaml( "./output/", mat_name.str().c_str(), mat );
                        }
			else if( mDType==SCImageDataType_Int32 )
                        {
                                int* ptr = ((int*)tmpImg) + mWidth*mHeight*ch;
                                Mat mat = Mat::zeros( mHeight, mWidth, CV_32SC1 );
                                for( int row=0; row<mHeight; row++ )
                                {
                                        int* matPtr = mat.ptr<int>(row);
                                        for( int x=0; x<mWidth; x++ )
                                        {
                                                matPtr[x] = ptr[row*mWidth+x];
                                        }
                                }
                                std::stringstream mat_name; mat_name<<name<<ch;

                                write2Yaml( "./output/", mat_name.str().c_str(), mat );
                        }
			else assert( false );
		}
		delete[] tmpImg;
#endif
	}

	char* getDTypeStr()
	{
		switch( mDType )
		{
			case SCImageDataType_None:
                                return "None";
				break;
			case SCImageDataType_UC:
				return "UC";
				break;
			case SCImageDataType_Float:
                                return "Float";
                                break;
			case SCImageDataType_Int32:
                                return "Int32";
                                break;
			default: 
				assert(false);
				break;
		}
		return "";
	}

	// returned pointer deleted outside
	template<class DType>
	DType* hostPtr( cudaStream_t stream )
	{
		unsigned char* tmpImg = copyD2H( stream );
		return (DType*)tmpImg;
	}

	void print( cudaStream_t stream )
	{
		printf( "data(%lx), width(%d), height(%d), channel(%d), pitch(%d), dtype(%s), order(%s)\n", 
				(unsigned long)(mData), mWidth, mHeight, mChannel, pitch(),
				getDTypeStr(),
		     		mOrder==SCImageOrder_Row?"Row":"Column" );
		
		unsigned char* tmpImg = copyD2H( stream );
		const int maxW = 5;
		const int maxH = 10;
		int W = maxW<mWidth?maxW:mWidth;
		int H = maxH<mHeight?maxH:mHeight;
		for( int h=0; h<H; h++ )
		{
			printf("line %4d: ", h);
			for( int w=0; w<W; w++ )
			{
				printf("(");
				for( int c=0; c<mChannel; c++ )
				{
					switch( mDType )
					{
						case SCImageDataType_UC:
							printf("%3u,", AT<unsigned char>(tmpImg,w,h,c));
							break;
						case SCImageDataType_Float:
                                                        printf("%3.2f,", AT<float>(tmpImg,w,h,c));
							break;
						case SCImageDataType_Int32:
                                                        printf("%d,", AT<int>(tmpImg,w,h,c));
                                                        break;
						default:
							printf("error,");
							break;
					}
				}
				printf(")");
			}
			printf("\n");
		}
		delete[] tmpImg;
	}
public:
	unsigned char* mData;
	int mWidth, mHeight, mChannel, mCapacity; // capacity in bytes;
	SCImageDataType mDType; 
	SCImageOrder mOrder;
	static int mOccupy;
};

int SCImage::mOccupy = 0;

void checkCublasErrors( cublasStatus_t error )
{
  if( error==CUBLAS_STATUS_SUCCESS ) return;
  switch( error )
  {
    case_print(CUBLAS_STATUS_SUCCESS)
    case_print(CUBLAS_STATUS_NOT_INITIALIZED)
    case_print(CUBLAS_STATUS_ALLOC_FAILED)
    case_print(CUBLAS_STATUS_INVALID_VALUE)
    case_print(CUBLAS_STATUS_ARCH_MISMATCH)
    case_print(CUBLAS_STATUS_MAPPING_ERROR)
    case_print(CUBLAS_STATUS_EXECUTION_FAILED)
    case_print(CUBLAS_STATUS_INTERNAL_ERROR)
    case_print(CUBLAS_STATUS_NOT_SUPPORTED)
    case_print(CUBLAS_STATUS_LICENSE_ERROR)
    default:
      assert("checkCublasErrors() failed!"==0);
      break;
  }
}
#endif
