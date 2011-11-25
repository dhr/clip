#pragma once
#ifndef CLIP_DETAIL_GLOBALBUFFERIMPL_H
#define CLIP_DETAIL_GLOBALBUFFERIMPL_H

#include "clip/basictypes.hpp"
#include "clip/detail/imagebufferimpl.hpp"

namespace clip {
namespace detail {

void floatToHalf(const cl::Buffer& floats, i32 n, cl::Buffer& halfs) {
  static CachedKernel cache("float2half");
  cl::Kernel& kernel = cache.get();
  kernel.setArg(0, floats);
  kernel.setArg(1, halfs);
  CurrentQueue().enqueueNDRangeKernel(kernel, cl::NullRange,
                                      cl::NDRange(n, 1), cl::NullRange);
}

void halfToFloat(const cl::Buffer& halfs, i32 n, cl::Buffer& floats) {
  static CachedKernel cache("half2float");
  cl::Kernel& kernel = cache.get();
  kernel.setArg(0, halfs);
  kernel.setArg(1, floats);
  CurrentQueue().enqueueNDRangeKernel(kernel, cl::NullRange,
                                      cl::NDRange(n, 1), cl::NullRange);
}

class GlobalBufferImpl : public ImageBufferImpl {
 protected:
  cl::Buffer buffer_;
  
  void initBufferFromFloatData(const f32 *data, cl_mem_flags flags) {
    flags |= CL_MEM_COPY_HOST_PTR;
    flags &= ~CL_MEM_USE_HOST_PTR;
    
    ImageData padded = makePaddedData(data);
    
    i32 n = paddedWidth_*paddedHeight_;
    buffer_ = cl::Buffer(CurrentContext(), flags, n*sizeof(f32),
                         &padded.data()[0]);
    
    if (valType_ == Float16) {
      cl::Buffer imvalBuffer(CurrentContext(), 0,
                             n*SizeofValueType(valType_));
      floatToHalf(buffer_, n, imvalBuffer);
      buffer_ = imvalBuffer;
    }
  }
  
 public:
  GlobalBufferImpl()
  : ImageBufferImpl(Global, Float32, &buffer_, 0, 0, 0, 0) {}
  
  GlobalBufferImpl(cl::Buffer data, i32 width, i32 height, ValueType valType,
                   i32 xAlign, i32 yAlign)
  : ImageBufferImpl(Global, CurrentImBufValueType(),
                    &buffer_, width, height, xAlign, yAlign),
    buffer_(data) {
    initPaddingValues();
    // FIXME: Ensure the data we've been passed conforms with the dimensions
    // Just make sure the amount of memory data references is >= what we need
  }
  
  GlobalBufferImpl(const f32 *data, i32 width, i32 height, ValueType valType,
                   i32 xAlign, i32 yAlign, cl_mem_flags flags)
  : ImageBufferImpl(Global, valType, &buffer_, width, height, xAlign, yAlign)
  {
    if (data != NULL)
      initBufferFromFloatData(data, flags);
    else {
      initPaddingValues();
      i32 nBytes = paddedWidth_*paddedHeight_*SizeofValueType(valType_);
      buffer_ = cl::Buffer(CurrentContext(), 0, nBytes);
    }
  }
  
  void fetchData(f32* data) const {
    i32 n = paddedWidth_*paddedHeight_;
    i32 nBytes = n*sizeof(f32);
    std::vector<f32> temp(n);
    
    cl::Buffer sourceBuf;
    if (valType_ == Float32)
      sourceBuf = buffer_;
    else if (valType_ == Float16) {
      sourceBuf = cl::Buffer(CurrentContext(), 0, nBytes);
      halfToFloat(buffer_, n, sourceBuf);
    }
    
    CurrentQueue().enqueueReadBuffer(sourceBuf, true, 0, nBytes, &temp[0]);
        
    unpadData(&temp[0], data);
  };
  
  void sendData(const f32 *data) {
    i32 n = paddedWidth_*paddedHeight_;
    i32 nBytes = n*SizeofValueType(valType_);
    
    ImageData padded = makePaddedData(data);
    f32* paddedPtr = &padded.data()[0];
    
    if (valType_ == Float32) {
      CurrentQueue().enqueueWriteBuffer(buffer_, true, 0, nBytes, paddedPtr);
      return;
    }
    
    cl::Buffer temp(CurrentContext(), CL_MEM_COPY_HOST_PTR,
                    n*sizeof(f32), paddedPtr);
    floatToHalf(temp, n, buffer_);
  }
  
  void copyInto(ImageBufferImpl *dest) const {
    assert(type_ == dest->type());
    assert(valType_ == dest->valType());
    assert(paddedWidth_ == dest->paddedWidth() &&
           paddedHeight_ == dest->paddedHeight());
    
    GlobalBufferImpl *realDest = static_cast<GlobalBufferImpl *>(dest);
    
    CurrentQueue().enqueueCopyBuffer(buffer_, realDest->buffer_, 0, 0,
                                     paddedWidth_*paddedHeight_*
                                       SizeofValueType(valType_));
  }
};

}}

#endif
