VAPOR_HOME = .
CXX_DBG = c++
CXX_CPU = c++
CXX_OMP = c++
CXX_GPU = nvcc

FLAG_ANY = -Wall -std=c++17 -MMD -MP -I$(VAPOR_HOME)/include
FLAG_DBG = -O0 -g -D VAPOR_DEBUG
FLAG_CPU = -Ofast
FLAG_OMP = -Ofast -Xpreprocessor -fopenmp
FLAG_GPU = -O3 -x cu --extended-lambda

LIBS_ANY = -lhdf5
LIBS_DBG =
LIBS_CPU =
LIBS_OMP = -lomp
LIBS_GPU =

EXE_DBG += bin/array_demo_dbg
EXE_DBG += bin/config_demo_dbg
EXE_DBG += bin/euler1d_dbg
EXE_DBG += bin/hdf5_demo_dbg
EXE_DBG += bin/sim_demo_dbg
EXE_DBG += bin/srhd1d_dbg
EXE_CPU += bin/array_demo_cpu
EXE_CPU += bin/config_demo_cpu
EXE_CPU += bin/euler1d_cpu
EXE_CPU += bin/hdf5_demo_cpu
EXE_CPU += bin/sim_demo_cpu
EXE_CPU += bin/srhd1d_cpu
EXE_OMP += bin/array_demo_omp
EXE_OMP += bin/config_demo_omp
EXE_OMP += bin/euler1d_omp
EXE_OMP += bin/hdf5_demo_omp
EXE_OMP += bin/sim_demo_omp
EXE_OMP += bin/srhd1d_omp
OBJ_DBG += build/array_demo_dbg.o
OBJ_DBG += build/config_demo_dbg.o
OBJ_DBG += build/euler1d_dbg.o
OBJ_DBG += build/hdf5_demo_dbg.o
OBJ_DBG += build/sim_demo_dbg.o
OBJ_DBG += build/srhd1d_dbg.o
OBJ_CPU += build/array_demo_cpu.o
OBJ_CPU += build/config_demo_cpu.o
OBJ_CPU += build/euler1d_cpu.o
OBJ_CPU += build/hdf5_demo_cpu.o
OBJ_CPU += build/sim_demo_cpu.o
OBJ_CPU += build/srhd1d_cpu.o
OBJ_OMP += build/array_demo_omp.o
OBJ_OMP += build/config_demo_omp.o
OBJ_OMP += build/euler1d_omp.o
OBJ_OMP += build/hdf5_demo_omp.o
OBJ_OMP += build/sim_demo_omp.o
OBJ_OMP += build/srhd1d_omp.o
DEP_DBG += build/array_demo_dbg.d
DEP_DBG += build/config_demo_dbg.d
DEP_DBG += build/euler1d_dbg.d
DEP_DBG += build/hdf5_demo_dbg.d
DEP_DBG += build/sim_demo_dbg.d
DEP_DBG += build/srhd1d_dbg.d
DEP_CPU += build/array_demo_cpu.d
DEP_CPU += build/config_demo_cpu.d
DEP_CPU += build/euler1d_cpu.d
DEP_CPU += build/hdf5_demo_cpu.d
DEP_CPU += build/sim_demo_cpu.d
DEP_CPU += build/srhd1d_cpu.d
DEP_OMP += build/array_demo_omp.d
DEP_OMP += build/config_demo_omp.d
DEP_OMP += build/euler1d_omp.d
DEP_OMP += build/hdf5_demo_omp.d
DEP_OMP += build/sim_demo_omp.d
DEP_OMP += build/srhd1d_omp.d

EXE += $(EXE_DBG)
EXE += $(EXE_CPU)
EXE += $(EXE_OMP)
OBJ += $(OBJ_DBG)
OBJ += $(OBJ_CPU)
OBJ += $(OBJ_OMP)
DEP += $(DEP_DBG)
DEP += $(DEP_CPU)
DEP += $(DEP_OMP)

dbg: $(EXE_DBG)
cpu: $(EXE_CPU)
omp: $(EXE_OMP)
all: dbg cpu omp

array_demo: bin/array_demo_dbg bin/array_demo_cpu bin/array_demo_omp
config_demo: bin/config_demo_dbg bin/config_demo_cpu bin/config_demo_omp
euler1d: bin/euler1d_dbg bin/euler1d_cpu bin/euler1d_omp
hdf5_demo: bin/hdf5_demo_dbg bin/hdf5_demo_cpu bin/hdf5_demo_omp
sim_demo: bin/sim_demo_dbg bin/sim_demo_cpu bin/sim_demo_omp
srhd1d: bin/srhd1d_dbg bin/srhd1d_cpu bin/srhd1d_omp

bin/array_demo_dbg: build/array_demo_dbg.o
	$(CXX_DBG) -o $@ $< $(LIBS_ANY) $(LIBS_DBG)
build/array_demo_dbg.o: examples/array_demo.cpp
	$(CXX_DBG) -o $@ $< $(FLAG_ANY) $(FLAG_DBG) -c
bin/array_demo_cpu: build/array_demo_cpu.o
	$(CXX_CPU) -o $@ $< $(LIBS_ANY) $(LIBS_CPU)
build/array_demo_cpu.o: examples/array_demo.cpp
	$(CXX_CPU) -o $@ $< $(FLAG_ANY) $(FLAG_CPU) -c
bin/array_demo_omp: build/array_demo_omp.o
	$(CXX_OMP) -o $@ $< $(LIBS_ANY) $(LIBS_OMP)
build/array_demo_omp.o: examples/array_demo.cpp
	$(CXX_OMP) -o $@ $< $(FLAG_ANY) $(FLAG_OMP) -c

bin/config_demo_dbg: build/config_demo_dbg.o
	$(CXX_DBG) -o $@ $< $(LIBS_ANY) $(LIBS_DBG)
build/config_demo_dbg.o: examples/config_demo.cpp
	$(CXX_DBG) -o $@ $< $(FLAG_ANY) $(FLAG_DBG) -c
bin/config_demo_cpu: build/config_demo_cpu.o
	$(CXX_CPU) -o $@ $< $(LIBS_ANY) $(LIBS_CPU)
build/config_demo_cpu.o: examples/config_demo.cpp
	$(CXX_CPU) -o $@ $< $(FLAG_ANY) $(FLAG_CPU) -c
bin/config_demo_omp: build/config_demo_omp.o
	$(CXX_OMP) -o $@ $< $(LIBS_ANY) $(LIBS_OMP)
build/config_demo_omp.o: examples/config_demo.cpp
	$(CXX_OMP) -o $@ $< $(FLAG_ANY) $(FLAG_OMP) -c

bin/euler1d_dbg: build/euler1d_dbg.o
	$(CXX_DBG) -o $@ $< $(LIBS_ANY) $(LIBS_DBG)
build/euler1d_dbg.o: examples/euler1d.cpp
	$(CXX_DBG) -o $@ $< $(FLAG_ANY) $(FLAG_DBG) -c
bin/euler1d_cpu: build/euler1d_cpu.o
	$(CXX_CPU) -o $@ $< $(LIBS_ANY) $(LIBS_CPU)
build/euler1d_cpu.o: examples/euler1d.cpp
	$(CXX_CPU) -o $@ $< $(FLAG_ANY) $(FLAG_CPU) -c
bin/euler1d_omp: build/euler1d_omp.o
	$(CXX_OMP) -o $@ $< $(LIBS_ANY) $(LIBS_OMP)
build/euler1d_omp.o: examples/euler1d.cpp
	$(CXX_OMP) -o $@ $< $(FLAG_ANY) $(FLAG_OMP) -c

bin/hdf5_demo_dbg: build/hdf5_demo_dbg.o
	$(CXX_DBG) -o $@ $< $(LIBS_ANY) $(LIBS_DBG)
build/hdf5_demo_dbg.o: examples/hdf5_demo.cpp
	$(CXX_DBG) -o $@ $< $(FLAG_ANY) $(FLAG_DBG) -c
bin/hdf5_demo_cpu: build/hdf5_demo_cpu.o
	$(CXX_CPU) -o $@ $< $(LIBS_ANY) $(LIBS_CPU)
build/hdf5_demo_cpu.o: examples/hdf5_demo.cpp
	$(CXX_CPU) -o $@ $< $(FLAG_ANY) $(FLAG_CPU) -c
bin/hdf5_demo_omp: build/hdf5_demo_omp.o
	$(CXX_OMP) -o $@ $< $(LIBS_ANY) $(LIBS_OMP)
build/hdf5_demo_omp.o: examples/hdf5_demo.cpp
	$(CXX_OMP) -o $@ $< $(FLAG_ANY) $(FLAG_OMP) -c

bin/sim_demo_dbg: build/sim_demo_dbg.o
	$(CXX_DBG) -o $@ $< $(LIBS_ANY) $(LIBS_DBG)
build/sim_demo_dbg.o: examples/sim_demo.cpp
	$(CXX_DBG) -o $@ $< $(FLAG_ANY) $(FLAG_DBG) -c
bin/sim_demo_cpu: build/sim_demo_cpu.o
	$(CXX_CPU) -o $@ $< $(LIBS_ANY) $(LIBS_CPU)
build/sim_demo_cpu.o: examples/sim_demo.cpp
	$(CXX_CPU) -o $@ $< $(FLAG_ANY) $(FLAG_CPU) -c
bin/sim_demo_omp: build/sim_demo_omp.o
	$(CXX_OMP) -o $@ $< $(LIBS_ANY) $(LIBS_OMP)
build/sim_demo_omp.o: examples/sim_demo.cpp
	$(CXX_OMP) -o $@ $< $(FLAG_ANY) $(FLAG_OMP) -c

bin/srhd1d_dbg: build/srhd1d_dbg.o
	$(CXX_DBG) -o $@ $< $(LIBS_ANY) $(LIBS_DBG)
build/srhd1d_dbg.o: examples/srhd1d.cpp
	$(CXX_DBG) -o $@ $< $(FLAG_ANY) $(FLAG_DBG) -c
bin/srhd1d_cpu: build/srhd1d_cpu.o
	$(CXX_CPU) -o $@ $< $(LIBS_ANY) $(LIBS_CPU)
build/srhd1d_cpu.o: examples/srhd1d.cpp
	$(CXX_CPU) -o $@ $< $(FLAG_ANY) $(FLAG_CPU) -c
bin/srhd1d_omp: build/srhd1d_omp.o
	$(CXX_OMP) -o $@ $< $(LIBS_ANY) $(LIBS_OMP)
build/srhd1d_omp.o: examples/srhd1d.cpp
	$(CXX_OMP) -o $@ $< $(FLAG_ANY) $(FLAG_OMP) -c


$(OBJ): | build
$(EXE): | bin

build:
	@mkdir -p $@
bin:
	@mkdir -p $@
clean:
	$(RM) $(DEP) $(OBJ) $(EXE)

-include $(DEP)
