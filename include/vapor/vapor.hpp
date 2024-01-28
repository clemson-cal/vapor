/**
================================================================================
Copyright 2023 - 2024, Jonathan Zrake

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
================================================================================
*/




#pragma once
#define VAPOR_STD_MAP
#define VAPOR_STD_STRING
#define VAPOR_STD_VECTOR
#include "hdf5/hdf5_array.hpp"
#include "hdf5/hdf5_map.hpp"
#include "hdf5/hdf5_native.hpp"
#include "hdf5/hdf5_repr.hpp"
#include "hdf5/hdf5_string.hpp"
#include "hdf5/hdf5_vector.hpp"
#include "vapor/array.hpp"
#include "vapor/comm.hpp"
#include "vapor/compat.hpp"
#include "vapor/executor.hpp"
#include "vapor/functional.hpp"
#include "vapor/future.hpp"
#include "vapor/index_space.hpp"
#include "vapor/mat.hpp"
#include "vapor/memory.hpp"
#include "vapor/optional.hpp"
#include "vapor/parse.hpp"
#include "vapor/print.hpp"
#include "vapor/runtime.hpp"
#include "vapor/sim.hpp"
#include "vapor/vec.hpp"
#include "visit_struct/visit_struct.hpp"
