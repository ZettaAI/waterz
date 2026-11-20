#pragma once
// The thread count waterz's parallel regions run with: set per process (waterz.set_num_threads), 0 meaning
// OpenMP's own default. Every result is the same at any count; only the time changes.
#ifdef _OPENMP
#include <omp.h>
#endif

inline int& wz_thread_setting() {
	static int n = 0;
	return n;
}

inline int wz_threads() {
#ifdef _OPENMP
	const int n = wz_thread_setting();
	return n > 0 ? n : omp_get_max_threads();
#else
	return 1;
#endif
}
