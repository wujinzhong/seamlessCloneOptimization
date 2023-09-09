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

void seamlessClone_sample()
{
    // Read images : src image will be cloned into dst
#if false
    //Mat src = imread("./images/airplane2.jpg");
	//Mat src = imread("./images/airplane592x592.jpg");
	Mat src = imread("./images/airplane154x100.jpg");
	Mat dst = imread("./images/sky.jpg");
#else
    //Mat src = readFromYaml("./images/src.yml");
    //Mat dst = readFromYaml("./images/dst.yml");
    Mat src = readFromYaml("./images/src_2400x1552.yml");
    Mat dst = readFromYaml("./images/dst_4800x2694.yml");
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
    // The location of the center of the src in the dst
    //Point center(800,100);
    Point center(2400,1347);
    write2Yaml_( src_mask );
    
    // Seamlessly clone src into dst and put the results in output
    Mat output;
    seamlessClone(src, dst, src_mask, center, output, NORMAL_CLONE);
    
    // Save result
    //imwrite("./images/opencv-seamless-cloning-example.jpg", output);
#if USE_POLYGON_MASK
    imwrite("./output/opencv-seamless-cloning-example-polygon.bmp", output);
#else
    //imwrite("./output/opencv-seamless-cloning-example-rect-all-255.bmp", output);
    imwrite("./output/opencv-seamless-cloning-example-rect-all-255-2400x1552.bmp", output);
#endif
    //Mat src_154x100 = src;
    //write2Yaml_( src );
	//write2Yaml_( dst );
	//write2Yaml_( output );
}

int main()
{
	seamlessClone_sample(  );
	return 0;
}

