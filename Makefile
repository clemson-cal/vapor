CXXFLAGS = -std=c++17 -Ofast -Wall
NVCFLAGS = -std=c++17 --extended-lambda
NVCC = nvcc
MPICC = mpicc
VAPOR_INCLUDES = $(wildcard include/vapor/*.hpp)
HDF5_INCLUDES = $(wildcard include/hdf5/*.hpp)

default: \
 examples/array_demo \
 examples/config_demo \
 examples/euler1d \
 examples/hdf5_demo \
 examples/sim_demo \
 examples/srhd1d

examples/array_demo: examples/array_demo.cpp $(VAPOR_INCLUDES)
	$(CXX) $(CXXFLAGS) -o $@ $< -I include -D VAPOR_ARRAY_BOUNDS_CHECK=1 -D VAPOR_VEC_BOUNDS_CHECK=1

examples/config_demo: examples/config_demo.cpp $(VAPOR_INCLUDES)
	$(CXX) $(CXXFLAGS) -o $@ $< -I include

examples/euler1d: examples/euler1d.cpp $(VAPOR_INCLUDES)
	$(CXX) $(CXXFLAGS) -o $@ $< -I include

examples/hdf5_demo: examples/hdf5_demo.cpp $(VAPOR_INCLUDES) $(HDF5_INCLUDES)
	$(CXX) $(CXXFLAGS) -o $@ $< -I include -lhdf5

examples/sim_demo: examples/sim_demo.cpp $(VAPOR_INCLUDES) $(HDF5_INCLUDES)
	$(CXX) $(CXXFLAGS) -o $@ $< -I include -lhdf5

examples/srhd1d: examples/srhd1d.cpp $(VAPOR_INCLUDES)
	$(CXX) $(CXXFLAGS) -o $@ $< -I include

gpu: \
 examples/demo_gpu \
 examples/euler1d_gpu

examples/euler1d_gpu: examples/euler1d.cpp $(VAPOR_INCLUDES)
	$(NVCC) $(NVCFLAGS) -x cu $< -o $@ -I include

examples/array_demo_gpu: examples/array_demo.cpp $(VAPOR_INCLUDES)
	$(NVCC) $(NVCFLAGS) -x cu $< -o $@ -I include

mpi: \
 examples/mpi_demo
 
examples/mpi_demo: examples/mpi_demo.cpp $(VAPOR_INCLUDES)
	$(MPICC) $(CXXFLAGS) -o $@ $< -I include -lc++ -D VAPOR_ARRAY_BOUNDS_CHECK=1
