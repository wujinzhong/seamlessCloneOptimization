# seamlessCloneOptimization
This is my CUDA optimization of OpenCV seamlessClone API at NORMAL_CLONE mode.

# Overview
Seamless clone is an advanced image editing feature published in this paper, “Poisson Image Editing”, OpenCV has this feature too. This project re-implements seamless clone feature for NVIDIA GPU platform, using CUDA programming.

Seamless clone is an advanced image editing feature, not like direct cloning image patch from source image to destination image, which has a drawback of inconsistent boundary colors. Seamless clone uses a guidance of vector in differential field and solve a Poisson equation with boundary pixels from destination image. This tech beautifully solves the problem of inconsistent drawback between source image and destination image. Details please refer to the original paper.

![image](https://github.com/wujinzhong/seamlessCloneOptimization/assets/52945455/95437287-3f6f-44bd-8411-378f681ef442)

(a)

![image](https://github.com/wujinzhong/seamlessCloneOptimization/assets/52945455/8ae8d7c9-1f26-4fef-a0eb-60d0f75ac7a1)

(b)

My test result:

Fig. 1 Seamless clone concepts.

![image](https://github.com/wujinzhong/seamlessCloneOptimization/assets/52945455/d8889fc4-b57b-42e2-acee-f08c4295d3f6)

+

![image](https://github.com/wujinzhong/seamlessCloneOptimization/assets/52945455/780841ca-f238-4436-a462-739010744beb)

=

![image](https://github.com/wujinzhong/seamlessCloneOptimization/assets/52945455/d8cbcad6-67cf-40a9-a882-11cfa949b139)


# How to use this source code

> I use docker environment for this developing.
> 
> “docker run --gpus all -it -v /home/usr_name/:/usr_name --network host nvcr.io/nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04 bash”
> 
> In docker, download CUDA samples from here, https://github.com/NVIDIA/cuda-samples.
> 
> Install OpenCV, I use OpenCV3.4.5 from source code, as to print some log and output some data while verifying CUDA re-implementation.
> 
> git clone https://github.com/wujinzhong/seamlessCloneOptimization.git
> 
> download data from here, https://drive.google.com/drive/folders/1k4pJLi0T0u3stawnQ76NkLchA7tTmgO7?usp=sharing, and put to target folder repectively.
> 
> Copy “seamlessClone-CUDA” dir to “/your_cuda_samples_repo/Samples/4_CUDA_Libraries/”. Goto “./seamlessClone-CUDA” and run “./make2.sh” and the output image is in “./seamlessClone-CUDA/output/ucRGB_Output.bmp”, this is the seamless cloned image from source patch “./images/src.yml” and destination image “./images/dst.yml”. We use yml format for source/destination images to make sure the input is always exactly the same for our CUDA re-implementation and OpenCV. Open “./seamlessClone-CUDA/output/ucRGB_Output.bmp” to see if the output is correct.
> 
> Copy “seamlessClone-OpenCV” dir to “/your_opencv_dir/OpenCV-3.4.5/opencv-3.4.5/samples/cpp/” and goto “./seamlessClone-OpenCV”. “make” it and run “./seamlessClone-OpenCV”. Check the OpenCV output here “./output/opencv-seamless-cloning-example-rect-all-255.bmp”.
> 
> goto “/seamlessCloneOptimization/compare”, open file “./vs.py” find and modify output file path “xxxxx/opencv-seamless-cloning-example-rect-all-255.bmp”, ”xxxx/ucRGB_Output.bmp”, “xxxxx/g*.yml”, “xxxxx/mod_diff*.yml”, to reflect your file path.
> 
> 9.	Run “python vs.py” and the output look like is in diff.yml

##Install OpenCV from source code
>apt-get update && apt-get install -y build-essential pkg-config software-properties-common cmake git libgtk2.0-dev libavcodec-dev libavformat-dev libswscale-dev && add-apt-repository "deb http://security.ubuntu.com/ubuntu xenial-security main" && apt install -y libjasper1 libjasper-dev python-dev python-numpy libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libjasper-dev libdc1394-22-dev libavcodec-dev libavformat-dev libswscale-dev libv4l-dev liblapacke-dev libxvidcore-dev libx264-dev libatlas-base-dev gfortran
>
>apt-get install -y ffmpeg
>
>cd /thor/OpenCV3.4.5
>
>wget https://github.com/opencv/opencv/archive/3.4.5.zip
>
>mv 3.4.5.zip opencv-3.4.5.zip
>
>wget https://github.com/opencv/opencv_contrib/archive/3.4.5.zip
>
>mv 3.4.5.zip opencv_contrib-3.4.5.zip
>
>apt-get install unzip
>
>unzip opencv-3.4.5.zip
>
>unzip opencv_contrib-3.4.5.zip
>
>cd /thor/OpenCV3.4.5/opencv-3.4.5/
>
>mkdir build && cd build
>
>python -c "import sys; print sys.prefix"会输出/usr, 替换掉下面的‘/usr’
>
>cmake -D CMAKE_BUILD_TYPE=RELEASE     -D CMAKE_INSTALL_PREFIX=/usr     -D INSTALL_C_EXAMPLES=ON     -D OPENCV_EXTRA_MODULES_PATH=/thor/opencv/opencv_contrib-3.4.5/modules     -D WITH_TBB=ON     -D WITH_V4L=ON     -D WITH_GTK=ON     -D WITH_OPENGL=ON     -D BUILD_EXAMPLES=ON -D WITH_CUDNN=ON -D OPENCV_DNN_CUDA=ON ..
>
>make -j16
>
>make install

# Tech background & implementation details 
Check document in this repo

# Optimization Tips

## Initialization
We group all the device memory allocation operations in initialization step, and free all the allocated device memory just before application exit. So there is no need to break the GPU computation pipeline for pre-processing and Poisson solver.

## Minimal Memory Copy
OpenCV implementation has many tiny operators while data pre-processing, like sub image crop, int to float type conversion, etc.; which are much a memory bound problem, slow down the overall performance. In our first version implementation, we use NPP for data pre-processing, which has the same problem. At last, while performance optimization, we remove all these memory copy operations into one kernel, avoid unnecessary read/write of device memory.

## CUDA Kernel Merging
Small operations are all grouped into one CUDA kernel as much as possible, which avoid unnecessary read/write of device memory and kernel launch operations.

## CUDA Stream
We use one CUDA stream for all the data pre-processing and Poisson equation solver, which make it possible for potential performance improvement using multiple threads. This strategy avoids unnecessary CUDA stream synchronize operations.

## Warm Up and Multiple Run
Warm up and multiple run is necessary for true performance timing, like this pseudo code shows:
> Func();
> 
> Timing(begin);
> 
> For( int loop=0; loop<Loops; loop++ ){
> 
> 	Func();
> 
> 	Synchronize(CUDA stream);
> 
> }
> 
> Timing(end);
> 
> Print_time(end-start);

# Profiling
I original use nvprof while profiling and performance optimizing, but NSight System is new and much better than nvprof, very worth to have a try.

Please check the document for details of profiling step by step.
