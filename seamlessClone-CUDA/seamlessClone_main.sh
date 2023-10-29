rm *.o *.so seamlessClone_main

####################### build *.o file ##############################################

#seamlessClone_main.o
/usr/local/cuda/bin/nvcc -ccbin g++ -I../../common/inc -I../../common/  -I../../../common/inc -I../../../common/ -I../../Common/inc -I../../Common/  -I../../../Common/inc -I../../../Common/ -m64    -dc -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_75,code=compute_75 -o seamlessClone_main.o -c seamlessClone_main.cu

####################### build *.so file for later python wrapping ##############################################

#seamlessClone_main.so
/usr/local/cuda/bin/nvcc -Xcompiler=-fPIC -ccbin g++ -I../../common/inc -I../../common/  -I../../../common/inc -I../../../common/ -I../../Common/inc -I../../Common/  -I../../../Common/inc -I../../../Common/ -m64 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_75,code=compute_75 -I/usr/include/python2.7 -I/usr/local/lib/python2.7/dist-packages/numpy/core/include/ -shared `pkg-config --cflags opencv`  `pkg-config --libs opencv` --shared -o seamlessClone_main.so seamlessClone_main.cu -lcudadevrt -lcublas -lnppisu -lnppist -lnppicc -lnppif -lnppc -lnpps -lnppial -lnppidei -lnppim -lopencv_stitching -lopencv_superres -lopencv_videostab -lopencv_aruco -lopencv_bgsegm -lopencv_bioinspired -lopencv_ccalib -lopencv_dnn_objdetect -lopencv_dpm -lopencv_face -lopencv_freetype -lopencv_fuzzy -lopencv_hfs -lopencv_img_hash -lopencv_line_descriptor -lopencv_optflow -lopencv_reg -lopencv_rgbd -lopencv_saliency -lopencv_stereo -lopencv_structured_light -lopencv_phase_unwrapping -lopencv_surface_matching -lopencv_tracking -lopencv_datasets -lopencv_text -lopencv_dnn -lopencv_plot -lopencv_xfeatures2d -lopencv_shape -lopencv_video -lopencv_ml -lopencv_ximgproc -lopencv_xobjdetect -lopencv_objdetect -lopencv_calib3d -lopencv_features2d -lopencv_highgui -lopencv_videoio -lopencv_imgcodecs -lopencv_flann -lopencv_xphoto -lopencv_photo -lopencv_imgproc -lopencv_core -ldl -lm -lpthread -lrt -lnppig -lcufft -lnvjpeg

cp ./seamlessClone_main.so /usr/lib/x86_64-linux-gnu/

####################### seamlessClone_main.o can be replaced by seamlessClone_main.so ##############################################
/usr/local/cuda/bin/nvcc -ccbin g++   -m64      -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_75,code=sm_75 -o seamlessClone_main seamlessClone_main.so  -lcudadevrt -lcublas -lnppisu -lnppist -lnppicc -lnppif -lnppc -lnpps -lnppial -lnppidei -lnppim -lopencv_stitching -lopencv_superres -lopencv_videostab -lopencv_aruco -lopencv_bgsegm -lopencv_bioinspired -lopencv_ccalib -lopencv_dnn_objdetect -lopencv_dpm -lopencv_face -lopencv_freetype -lopencv_fuzzy -lopencv_hfs -lopencv_img_hash -lopencv_line_descriptor -lopencv_optflow -lopencv_reg -lopencv_rgbd -lopencv_saliency -lopencv_stereo -lopencv_structured_light -lopencv_phase_unwrapping -lopencv_surface_matching -lopencv_tracking -lopencv_datasets -lopencv_text -lopencv_dnn -lopencv_plot -lopencv_xfeatures2d -lopencv_shape -lopencv_video -lopencv_ml -lopencv_ximgproc -lopencv_xobjdetect -lopencv_objdetect -lopencv_calib3d -lopencv_features2d -lopencv_highgui -lopencv_videoio -lopencv_imgcodecs -lopencv_flann -lopencv_xphoto -lopencv_photo -lopencv_imgproc -lopencv_core -ldl -lm -lpthread -lrt -lnppig -lcufft -lnvjpeg

./seamlessClone_main ./images/src.yml ./images/dst.yml ./images/src_mask.yml 800 150 1
#./seamlessClone_main ./images/src4_109x164.yml ./images/dst4_494x875.yml ./images/src4_mask_109x164.yml 255 479 1
#./seamlessClone_main ./images/src3_181x153.yml ./images/dst3_1920x1080.yml ./images/src3_mask_181x153.yml 923 236 1
#./seamlessClone_main ./images/src2_356x376.yml ./images/dst2_1920x1080.yml ./images/src2_mask_356x376.yml 936 188 1
#./seamlessClone_main ./images/src_494x528.yml ./images/dst_1920x1080.yml ./images/src_mask_494x528.yml 1114 376 1
#./seamlessClone_main ./images/src_2400x1552.yml ./images/dst_4800x2694.yml ./images/src_mask_2400x1552.yml 2400 1347 1

make
cd ./seamlessClone-python-binding && make && python SeamlessClone_test.py && cd .. 