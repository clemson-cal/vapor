#pragma once

#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif

namespace vapor {
	using uint = unsigned int;
}

#define VAPOR_ARRAY_BOUNDS_CHECK 0
