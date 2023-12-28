CXXFLAGS = -std=c++17 -Ofast -Wall
NVCFLAGS = -std=c++17 --extended-lambda
NVCC = nvcc

default: \
 examples/array_demo \
 examples/euler1d \
 examples/srhd1d \
 examples/hdf5_demo \
 examples/config_demo \
 examples/sim_demo

examples/array_demo: examples/array_demo.cpp include/*.hpp
	$(CXX) $(CXXFLAGS) -o $@ $< -I include

examples/euler1d: examples/euler1d.cpp include/*.hpp
	$(CXX) $(CXXFLAGS) -o $@ $< -I include

examples/srhd1d: examples/srhd1d.cpp include/*.hpp
	$(CXX) $(CXXFLAGS) -o $@ $< -I include

examples/hdf5_demo: examples/hdf5_demo.cpp include/*.hpp
	$(CXX) $(CXXFLAGS) -o $@ $< -I include -lhdf5

examples/config_demo: examples/config_demo.cpp include/*.hpp
	$(CXX) $(CXXFLAGS) -o $@ $< -I include

examples/sim_demo: examples/sim_demo.cpp include/*.hpp
	$(CXX) $(CXXFLAGS) -o $@ $< -I include -lhdf5

gpu: \
 examples/demo_gpu \
 examples/euler1d_gpu

examples/euler1d_gpu: examples/euler1d.cpp include/*.hpp
	$(NVCC) $(NVCFLAGS) -x cu $< -o $@ -I include

examples/array_demo_gpu: examples/array_demo.cpp include/*.hpp
	$(NVCC) $(NVCFLAGS) -x cu $< -o $@ -I include
