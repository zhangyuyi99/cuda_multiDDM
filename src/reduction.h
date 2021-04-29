#ifndef _REDUCTION_H_
#define _REDUCTION_H_

template <class T>

void maskReduce(int size, int threads, int blocks, T *d_idata, bool *d_mask, T *d_odata);

#endif
