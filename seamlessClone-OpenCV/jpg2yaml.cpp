#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/photo.hpp"
#include <iostream>

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

Mat readFromYaml( const char* file_path )
{
	Mat mat;

	cv::FileStorage fs( file_path, cv::FileStorage::READ);
	fs["data"]>>mat;

	return mat;	
}

#define USE_POLYGON_MASK false

void seamlessClone_jpg2yml(int argc, char** argv)
{
	printf("argc: %d\n", argc);
	for(int i=0; i<argc; i++)
		printf("argv[%d]: %s\n", i, argv[i]);
	assert( argc==3 );
	char* src_image_path = (char*)argv[1];
	char* dst_image_path = (char*)argv[2];

	Mat src = imread(src_image_path);
	Mat dst = imread(dst_image_path);

    // Create a rough mask around the airplane.
    Mat src_mask = Mat::zeros(src.rows, src.cols, src.depth());
    
    // Define the mask as a closed polygon
#if False //USE_POLYGON_MASK
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
    
    write2Yaml_( src_mask );
    write2Yaml_( src );
	write2Yaml_( dst );
}

int main(int argc, char** argv)
{
	seamlessClone_jpg2yml( argc, argv );
	return 0;
}

