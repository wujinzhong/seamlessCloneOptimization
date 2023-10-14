#include "seamlessClone_imp.cu"

//static Mat readFromYaml( const char* file_path )
//{
//	Mat mat;
//	printf("begin reading file: %s\n", file_path);
//
//	cv::FileStorage fs( file_path, cv::FileStorage::READ);
//	fs["data"]>>mat;
//
//    printf("mat shape: %d, %d, %d\n", mat.cols, mat.rows, mat.channels());
//
//	return mat;
//}

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

int main(int argc, const char *argv[])
{
  printf("argc: %d\n", argc);
	for(int i=0; i<argc; i++)
		printf("argv[%d]: %s\n", i, argv[i]);
	assert( argc==7 );
  char* src_image_path = (char*)argv[1];
	char* dst_image_path = (char*)argv[2];
  char* mask_image_path = (char*)argv[3];
	int centerX = atoi(argv[4]);
	int centerY = atoi(argv[5]);
  int gpu = atoi(argv[6]);

  Mat destMat, patchMat, maskMat;
  //
  if( load_inputs(destMat, patchMat, maskMat,
                  dst_image_path,
                  src_image_path,
                  mask_image_path))
        return EXIT_FAILURE;

  Mat retMat = seamlessClone_imp((void*)&patchMat, (void*)&destMat, (void*)&maskMat, centerX, centerY, gpu);
}