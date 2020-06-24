#-----------------------------------------------------------------------

MK_OPENCV_DIR           = MK_OPENCV_VERS          = 4.3.0_5
MK_OPENCV_DOT_PC        = $(MK_OPENCV_DIR)/$(MK_OPENCV_VERS)/lib/pkgconfig/opencv4.pc

#-----------------------------------------------------------------------

# To determine include flags, library flags and libraries, type this
# command at the Linux prompt in an XTerm window:
#     pkg-config --cflags --libs <the value of $/usr/local/Cellar/opencv

MK_I_FLAG               =
MK_I_FLAG              += -I/usr/local/Cellar/opencv/4.3.0_5/include/opencv4/opencv
MK_I_FLAG              += -I/usr/local/Cellar/opencv/4.3.0_5/include/opencv4

MK_L_FLAG               =
MK_L_FLAG              += -L/usr/local/Cellar/opencv/4.3.0_5/lib

MK_LIB                  =
MK_LIB                 += -lopencv_gapi
MK_LIB                 += -lopencv_stitching
MK_LIB                 += -lopencv_alphamat
MK_LIB                 += -lopencv_aruco
MK_LIB                 += -lopencv_bgsegm
MK_LIB                 += -lopencv_bioinspired
MK_LIB                 += -lopencv_ccalib
MK_LIB                 += -lopencv_dnn_objdetect
MK_LIB                 += -lopencv_dnn_superres
MK_LIB                 += -lopencv_dpm
MK_LIB                 += -lopencv_highgui
MK_LIB                 += -lopencv_face
MK_LIB                 += -lopencv_freetype
MK_LIB                 += -lopencv_fuzzy
MK_LIB                 += -lopencv_hfs
MK_LIB                 += -lopencv_img_hash
MK_LIB                 += -lopencv_intensity_transform
MK_LIB                 += -lopencv_line_descriptor
MK_LIB                 += -lopencv_quality
MK_LIB                 += -lopencv_rapid
MK_LIB                 += -lopencv_reg
MK_LIB                 += -lopencv_rgbd
MK_LIB                 += -lopencv_saliency
MK_LIB                 += -lopencv_sfm
MK_LIB                 += -lopencv_stereo
MK_LIB                 += -lopencv_structured_light
MK_LIB                 += -lopencv_phase_unwrapping
MK_LIB                 += -lopencv_superres
MK_LIB                 += -lopencv_optflow
MK_LIB                 += -lopencv_surface_matching
MK_LIB                 += -lopencv_tracking
MK_LIB                 += -lopencv_datasets
MK_LIB                 += -lopencv_text
MK_LIB                 += -lopencv_dnn
MK_LIB                 += -lopencv_plot
MK_LIB                 += -lopencv_videostab
MK_LIB                 += -lopencv_videoio
MK_LIB                 += -lopencv_xfeatures2d
MK_LIB                 += -lopencv_shape
MK_LIB                 += -lopencv_ml
MK_LIB                 += -lopencv_ximgproc
MK_LIB                 += -lopencv_video
MK_LIB                 += -lopencv_xobjdetect
MK_LIB                 += -lopencv_objdetect
MK_LIB                 += -lopencv_calib3d
MK_LIB                 += -lopencv_imgcodecs
MK_LIB                 += -lopencv_features2d
MK_LIB                 += -lopencv_flann
MK_LIB                 += -lopencv_xphoto
MK_LIB                 += -lopencv_photo
MK_LIB                 += -lopencv_imgproc
MK_LIB                 += -lopencv_core

#-----------------------------------------------------------------------

a.out:
	g++ \
            -std=c++11 \
            $(MK_I_FLAG) \
            $(MK_L_FLAG) \
            $(MK_LIB) \
            hello_world.cpp

#-----------------------------------------------------------------------

