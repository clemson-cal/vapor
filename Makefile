CXXFLAGS = -Wall -Ofast
CUDAFLAGS = -std=c++17 --extended-lambda
NVCC = nvcc

default: examples/array_demo examples/hdf5_demo examples/euler1d
gpu: examples/demo_gpu examples/euler1d_gpu

examples/array_demo: examples/array_demo.cpp include/*.hpp
	$(CXX) $(CXXFLAGS) -std=c++17 -o $@ $< -I include

examples/array_demo_gpu: examples/array_demo.cpp include/*.hpp
	$(NVCC) $(CUDAFLAGS) -x cu $< -o $@ -I include

examples/euler1d: examples/euler1d.cpp include/*.hpp
	$(CXX) $(CXXFLAGS) -std=c++17 -o $@ $< -I include

examples/euler1d_gpu: examples/euler1d.cpp include/*.hpp
	$(NVCC) $(CUDAFLAGS) -x cu $< -o $@ -I include

examples/hdf5_demo: examples/hdf5_demo.cpp include/*.hpp
	$(CXX) $(CXXFLAGS) -std=c++17 -o $@ $< -I include -lhdf5
