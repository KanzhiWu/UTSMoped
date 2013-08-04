LIBNAME     := libmoped

# USING SIFTFAST
INCLUDES    += -Ilibs/libsiftfast-1.1-src
SLIBS       += libs/libsiftfast-1.1-src/libsiftfast.a

# USING SIFTGPU
INCLUDES    += -Ilibs/SiftGPU/
SLIBS       += libs/SiftGPU/libsiftgpu.a

# USING LEVMAR
INCLUDES    += -Ilibs/levmar-2.4
SLIBS       += libs/levmar-2.4/liblevmar.a

# USING LBFGS
INCLUDES    += -Ilibs/liblbfgs
SLIBS       += libs/liblbfgs/liblbfgs.a

# USING ANN
INCLUDES    += -Ilibs/ann_1.1.1/ANN
SLIBS       += libs/ann_1.1.1/libann.a 

# USING ICP
INCLUDES    += -Ilibs/libicp
SLIBS       += libs/libicp/libicp.a

# USING OPENCV
#INCLUDES    += `pkg-config --cflags opencv`
CV_INC      := $(shell rospack cflags-only-I -q opencv2 | sed 's/ / -I/g;')
CV_LIB      := $(shell rospack libs-only-l -q opencv2 | sed 's/ / -l/g;')
INCLUDES    += -I$(CV_INC)

# USING PCL
#INCLUDES    += `pkg-config --cflags pcl`
#PCL_INC      := $(shell rospack cflags-only-I -q pcl | sed 's/ / -I/g;')
#PCL_LIB      := $(shell rospack libs-only-l -q pcl | sed 's/ / -l/g;')
#INCLUDES    += -I$(PCL_INC)

#PCL_INC      := $(-I/opt/ros/groovy/include/pcl-1.6 -I/opt/ros/groovy/include -I/usr/include/eigen3)
PCL_LIB      := $(pcl_keypoints -lpcl_registration -lpcl_features -lpcl_segmentation -lpcl_surface -lpcl_tracking -lpcl_filters -lpcl_sample_consensus -lpcl_search -lpcl_visualization -lpcl_io -lpcl_kdtree -lpcl_octree -lflann_cpp -lflann_cpp-gd -lpcl_common)
INCLUDES    += -I/opt/ros/fuerte/include/pcl-1.5
INCLUDES    += -I/opt/ros/fuerte/include
INCLUDES    += -I/usr/include/eigen3

INCLUDES    += -I/usr/include/vtk-5.8

# Sources
SOURCES.hpp := $(shell cd src;     find -name "*.hpp" | cut -c 3- )
SOURCES.ihpp:= $(shell cd include; find -name "*.hpp" | cut -c 3- )
SOURCES.hpp := $(SOURCES.hpp:%=src/%)
SOURCES.hpp += $(SOURCES.ihpp:%=include/%)

SOURCES.o   := $(shell cd src;     find -name "*.cpp" | cut -c 3- )
SOURCES.o   := $(SOURCES.o:%.cpp=tmp/%.o)

# Compiler flags 
CXX = g++
CXXFLAGS += -O3 -Wall -march=native -mfpmath=sse -Isrc -Iinclude -funroll-loops -fopenmp
CXXFLAGS += -ffast-math -fno-trapping-math 

CXXFLAGS += -g
#CXXFLAGS += -DNDEBUG -fsingle-precision-constant


# Targets 
.PHONY: clean all libs $(LIBNAME)
all : $(LIBNAME)

test : test.cpp lib/$(LIBNAME).a
	$(CXX) $(CXXFLAGS) $(INCLUDES) -Wl,--start-group -lgomp -lGLEW -lglut -lIL -l$(CV_LIB) -l$(PCL_LIB) -lvtkCommon -lvtkFiltering -lvtkRendering -lcv -lhighgui -lcvaux lib/$(LIBNAME).a $< -o $@
	./test



$(LIBNAME) : lib/$(LIBNAME).a

$(SLIBS) : libs

libs : 
	@cd libs; $(MAKE) --no-print-directory

lib/$(LIBNAME).a : $(SOURCES.o) $(SLIBS)
	-@mkdir -p lib
	-@mkdir -p tmp/libs
	@echo [ Creating $(@F) ]
	@cd tmp/libs/; for f in $(SLIBS); do ar x ../../$$f; done
	@ar rcs $@ $(SOURCES.o) `dir tmp/libs/*.o`

tmp/%.o : src/%.cpp $(SOURCES.hpp) $(SLIBS)
	-@mkdir -p $(@D)
	@echo [ Compiling $(@F) ]
	@$(CXX) -fPIC $(CXXFLAGS) $(INCLUDES) -c $< -o $@

clean:
	$(RM) -r tmp
	$(RM) -r lib
	cd libs; $(MAKE) clean --no-print-directory

