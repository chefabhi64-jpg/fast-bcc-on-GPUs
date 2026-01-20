#ifndef BCC_H
#define BCC_H

#include <vector>

#include "cuda_bcc/bcc_memory_utils.cuh"

void cuda_bcc(gpu_bcc& g_bcc_ds, bool last_bcc = false);

#endif // BCC_H