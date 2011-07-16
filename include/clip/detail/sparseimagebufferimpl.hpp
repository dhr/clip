#pragma once
#ifndef CLIP_DETAIL_SPARSEIMAGEBUFFERIMPL_H
#define CLIP_DETAIL_SPARSEIMAGEBUFFERIMPL_H

#include "clip/basictypes.hpp"
#include "clip/sparseimagedata.hpp"

namespace clip {

class SparseImageBuffer;

namespace detail {

void sparseFloatToHalf(const cl::Buffer& floats, i32 n, cl::Buffer& halfs) {
  static CachedKernel cache("sparsefloat2half");
  cl::Kernel& kernel = cache.get();
  kernel.setArg(0, floats);
  kernel.setArg(1, halfs);
  CurrentQueue().enqueueNDRangeKernel(kernel, cl::NullRange,
                                      cl::NDRange(n, 1), cl::NullRange);
}

void sparseHalfToFloat(const cl::Buffer& halfs, i32 n, cl::Buffer& floats) {
  static CachedKernel cache("sparsehalf2float");
  cl::Kernel& kernel = cache.get();
  kernel.setArg(0, halfs);
  kernel.setArg(1, floats);
  CurrentQueue().enqueueNDRangeKernel(kernel, cl::NullRange,
                                      cl::NDRange(n, 1), cl::NullRange);
}

i32 sparseValueSize(ValueType valType) {
  return 2 + 2 + SizeofValueType(valType);
}

class SparseImageBufferImpl {
 protected:
  cl::Buffer buffer_;
  ValueType valType_;
  i32 width_;
  i32 height_;
  i32 nelems_;
  
  friend class clip::SparseImageBuffer;
  
  SparseImageBufferImpl(const SparseImageData& data, ValueType valType,
                        cl_mem_flags flags)
  : valType_(valType),
    width_(data.width()), height_(data.height()),
    nelems_(data.numElems())
  {
    flags |= CL_MEM_COPY_HOST_PTR;
    flags &= ~CL_MEM_USE_HOST_PTR;
    
    if (nelems_ > 0) {
      buffer_ = cl::Buffer(CurrentContext(), flags,
                           nelems_*sizeof(SparseValue),
                           const_cast<SparseValue*>(&data.data()[0]));
      
      if (valType_ == Float16) {
        cl::Buffer temp(CurrentContext(), 0,
                        sparseValueSize(valType_)*nelems_);
        sparseFloatToHalf(buffer_, nelems_, temp);
        buffer_ = temp;
      }
    }
  }
  
  SparseImageBufferImpl(cl::Buffer data, i32 width, i32 height, i32 nelems,
                        ValueType valType)
  : buffer_(data), valType_(valType),
    width_(width), height_(height), nelems_(nelems) {}
  
 public:
  void fetchData(SparseValue* data) const {
    cl::Buffer sourceBuffer;
    if (valType_ == Float16) {
      sourceBuffer = cl::Buffer(CurrentContext(), 0,
                                sparseValueSize(valType_)*nelems_);
      sparseHalfToFloat(buffer_, nelems_, sourceBuffer);
    }
    else
      sourceBuffer = buffer_;
    
    CurrentQueue().enqueueReadBuffer(sourceBuffer, true, 0,
                                     nelems_*sizeof(SparseValue),
                                     data);
  }
  
  void sendData(const SparseValue* values) {
    i32 nBytes = nelems_*sizeof(SparseValue);
    
    if (valType_ == Float32) {
      CurrentQueue().enqueueWriteBuffer(buffer_, false, 0, nBytes, values);
      return;
    }
    
    cl::Buffer temp(CurrentContext(), CL_MEM_COPY_HOST_PTR,
                    sizeof(SparseValue)*nelems_,
                    const_cast<SparseValue*>(values));
    sparseFloatToHalf(temp, nelems_, buffer_);
  }
  
  inline const cl::Memory &mem() { return buffer_; }
  
  inline i32 width() const { return width_; }
  inline i32 height() const { return height_; }
  inline i32 numElems() const { return nelems_; }
  inline ValueType valType() const { return valType_; }
};

typedef std::tr1::shared_ptr<SparseImageBufferImpl> SparseImageBufferImplPtr;

}}

#endif
