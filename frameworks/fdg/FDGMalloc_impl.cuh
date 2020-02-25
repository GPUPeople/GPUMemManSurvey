#ifndef __FDGMALLOC_CU
#define __FDGMALLOC_CU

/*!	\file		FDGMalloc.cu
 *	\brief		Includes the implementation of FDGMalloc.
 *	\author		Sven Widmer <sven.widmer@gris.informatik.tu-darmstadt.de>
 *	\author		Dominik Wodniok <dominik.wodniok@gris.informatik.tu-darmstadt.de>
 *	\author		Nicolas Weber <nicolas.weber@gris.informatik.tu-darmstadt.de>
 *	\author		Michael Goesele <michael.goesele@gris.informatik.tu-darmstadt.de>
 *	\version	1.0
 *	\date		06-12-2012 
 */

namespace FDG {
	#include "src/SuperBlock_impl.cuh"
	#include "src/List_impl.cuh"
	#include "src/Warp_impl.cuh"
}

#endif