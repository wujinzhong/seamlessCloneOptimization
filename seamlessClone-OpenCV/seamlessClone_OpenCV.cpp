#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/photo.hpp"
#include <iostream>
#include <chrono>
#include <cassert>
using namespace std::chrono;
#include <iostream>
using namespace std;

using namespace cv;

#define write2Yaml_( mat ) write2Yaml( "./output/", #mat, mat )

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

Mat readFromYaml( char* file_path )
{
	Mat mat;

	cv::FileStorage fs( file_path, cv::FileStorage::READ);
	fs["data"]>>mat;

	return mat;	
}

#define USE_POLYGON_MASK false

void seamlessClone_sample( int argc, char** argv )
{
	printf("argc: %d\n", argc);
	for(int i=0; i<argc; i++)
		printf("argv[%d]: %s\n", i, argv[i]);
	assert( argc==5 );
	char* src_image_path = (char*)argv[1];
	char* dst_image_path = (char*)argv[2];
	int centerX = atoi(argv[3]);
	int centerY = atoi(argv[4]);

    // Read images : src image will be cloned into dst
#if true // load images from jpg
    Mat src = imread(src_image_path);
	Mat dst = imread(dst_image_path);
#else // just for testing, load images from yml, pre-convert jpg to yml via jpg2yml
	Mat src = readFromYaml(src_image_path);
    Mat dst = readFromYaml(dst_image_path);
#endif
    // Create a rough mask around the airplane.
    Mat src_mask = Mat::zeros(src.rows, src.cols, src.depth());
    
    // Define the mask as a closed polygon
#if USE_POLYGON_MASK
    {
	    Point poly[1][7];
	    poly[0][0] = Point(4, 80);
	    poly[0][1] = Point(30, 54);
	    poly[0][2] = Point(151,63);
	    poly[0][3] = Point(254,37);
	    poly[0][4] = Point(298,90);
	    poly[0][5] = Point(272,134);
	    poly[0][6] = Point(43,122);

	    const Point* polygons[1] = { poly[0] };
	    int num_points[] = { 7 };
	    // Create mask by filling the polygon
	    fillPoly(src_mask, polygons, num_points, 1, Scalar(255,255,255));
    }
#else
    {
	    Point poly[1][4];
            poly[0][0] = Point(0, 0);
            poly[0][1] = Point(src.cols-1, 0);
            poly[0][2] = Point(src.cols-1,src.rows-1);
            poly[0][3] = Point(0,src.rows-1);

	    const Point* polygons[1] = { poly[0] };
            int num_points[] = { 4 };
	    // Create mask by filling the polygon
    	    fillPoly(src_mask, polygons, num_points, 1, Scalar(255,255,255));
    }
#endif
	imwrite("./output/opencv-seamless-cloning-mask.jpg", src_mask);
	imwrite("./output/opencv-seamless-cloning-mask.bmp", src_mask);
	write2Yaml_( src_mask );

    
	// The location of the center of the src in the dst
	Point center(centerX, centerY);    
    
    Mat result;
	//warmup
	seamlessClone(src, dst, src_mask, center, result, NORMAL_CLONE);

	//profiling
	auto start = high_resolution_clock::now();
	{
		// Seamlessly clone src into dst and put the results in result
    	seamlessClone(src, dst, src_mask, center, result, NORMAL_CLONE);
	}
	auto stop = high_resolution_clock::now();
	auto duration = duration_cast<microseconds>(stop - start);
	cout << endl << "*************************************************************" <<endl<<endl;
	cout << "opencv seamlessClone() executing time for patch " << src.cols <<"x"<<src.rows <<": " << duration.count()/1000.0f << " ms."<<endl;
	cout << endl << "*************************************************************" <<endl<<endl;

    // Save result
    imwrite("./output/opencv-seamless-cloning-result.jpg", result);
	imwrite("./output/opencv-seamless-cloning-result.bmp", result);
    write2Yaml_( src );
	write2Yaml_( dst );
	write2Yaml_( result );
}

//clear && make && clear && make && ./seamlessClone_OpenCV ./images/airplane.jpg ./images/sky.jpg 800 150
int main( int argc, char** argv )
{
	seamlessClone_sample( argc, argv );
	return 0;
}

