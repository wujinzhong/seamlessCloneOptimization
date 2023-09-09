rm seamlessClone .o seamlessClone

/usr/local/cuda/bin/nvcc -ccbin g++ -I../../common/inc -I../../common/  -I../../../common/inc -I../../../common/ -I../../Common/inc -I../../Common/  -I../../../Common/inc -I../../../Common/ -m64    -dc -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_75,code=compute_75 -o seamlessClone.o -c seamlessClone.cu

/usr/local/cuda/bin/nvcc -ccbin g++   -m64      -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_75,code=sm_75 -o seamlessClone seamlessClone.o  -lcudadevrt -lcublas -lnppisu -lnppist -lnppicc -lnppif -lnppc -lnpps -lnppial -lnppidei -lnppim -lopencv_stitching -lopencv_superres -lopencv_videostab -lopencv_aruco -lopencv_bgsegm -lopencv_bioinspired -lopencv_ccalib -lopencv_dnn_objdetect -lopencv_dpm -lopencv_face -lopencv_freetype -lopencv_fuzzy -lopencv_hfs -lopencv_img_hash -lopencv_line_descriptor -lopencv_optflow -lopencv_reg -lopencv_rgbd -lopencv_saliency -lopencv_stereo -lopencv_structured_light -lopencv_phase_unwrapping -lopencv_surface_matching -lopencv_tracking -lopencv_datasets -lopencv_text -lopencv_dnn -lopencv_plot -lopencv_xfeatures2d -lopencv_shape -lopencv_video -lopencv_ml -lopencv_ximgproc -lopencv_xobjdetect -lopencv_objdetect -lopencv_calib3d -lopencv_features2d -lopencv_highgui -lopencv_videoio -lopencv_imgcodecs -lopencv_flann -lopencv_xphoto -lopencv_photo -lopencv_imgproc -lopencv_core -ldl -lm -lpthread -lrt -lnppig -lcufft -lnvjpeg

./seamlessClone device=0

