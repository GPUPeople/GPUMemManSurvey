/* Copyright (c) 2010 The Board of Trustees of the University of Illinois.
 * All rights reserved.
 */

#ifndef _XMALLOC_
#define _XMALLOC_

// Volatile memory access.
#define XMC_VLD_U32(x) (*(volatile uint32_t *)&(x))
#define XMC_VLD_U64(x) (*(volatile uint64_t *)&(x))

// Load data using atomic instructions instead of volatile loads.
//#define XMC_VLD_U32(x) ((uint32_t)atomicOr((unsigned int *)&(x), 0U))

// The CAS value is a randomly generated number that is unlikely to occur.
// This makes the CAS likely to fail.  A failed CAS is slightly faster
// than a successful CAS.
//#define XMC_VLD_U64(x) ((uint64_t)atomicCAS((unsigned long long *)&(x), \
//  0xaa0ca36c6102d25fULL, 0xaa0ca36c6102d25fULL))


// Initialize XMalloc.
// Returns zero on error, nonzero on success.
int xmcInit(size_t size);

// Read and print performance statistics.  This will do nothing unless
// XMC_PERFCOUNT is defined.
void xmcPrintStatistics(void);

#ifdef XMC_PERFCOUNT
// Used by xmcPrintStatistics
unsigned long long xmcCounterRead(const char *counter_name);
#endif

// Allocate memory
__device__ void *xmcMalloc(unsigned int size);

// Deallocate memory
__device__ void xmcFree(void *);

#endif
