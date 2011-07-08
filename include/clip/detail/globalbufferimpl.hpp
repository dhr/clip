#pragma once
#ifndef CLIP_DETAIL_GLOBALBUFFERIMPL_H
#define CLIP_DETAIL_GLOBALBUFFERIMPL_H

#include "clip/basictypes.hpp"
#include "clip/detail/imagebufferimpl.hpp"

namespace clip {
namespace detail {

void float2imval(const cl::Buffer& floats, i32 n, cl::Buffer& imvals) {
  static CachedKernel cache("float2imval");
  cl::Kernel& kernel = cache.get();
  kernel.setArg(0, floats);
  kernel.setArg(1, imvals);
  CurrentQueue().enqueueNDRangeKernel(kernel, cl::NullRange,
                                      cl::NDRange(n, 1), cl::NullRange);
}

void imval2float(const cl::Buffer& imvals, i32 n, cl::Buffer& floats) {
  static CachedKernel cache("imval2float");
  cl::Kernel& kernel = cache.get();
  kernel.setArg(0, imvals);
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
    
    if (valType_ != Float32) {
      cl::Buffer imvalBuffer(CurrentContext(), 0,
                             n*SizeofValueType(valType_));
      float2imval(buffer_, n, imvalBuffer);
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
    itemRange_ = cl::NDRange(paddedWidth_, paddedHeight_);
  }
  
  void fetchData(f32* data) const {
    i32 n = paddedWidth_*paddedHeight_;
    i32 nBytes = n*sizeof(f32);
    std::vector<f32> temp(n);
    
    cl::Buffer sourceBuf;
    if (valType_ == Float32)
      sourceBuf = buffer_;
    else {
      sourceBuf = cl::Buffer(CurrentContext(), 0, nBytes);
      imval2float(buffer_, n, sourceBuf);
    }
    
    CurrentQueue().enqueueReadBuffer(sourceBuf, true, 0, nBytes, &temp[0]);
    
    i32 xPadding = paddedWidth_ - width_;
    i32 yPadding = paddedHeight_ - height_;
    
    i32 leftPad = xPadding/2 + xPadding%2;
    i32 rightPad = xPadding/2;
    i32 bottomPad = yPadding/2 + yPadding%2;
    i32 topPad = yPadding/2;
    
    PadData(&temp[0], paddedWidth_, paddedHeight_,
            -leftPad, -rightPad, -bottomPad, -topPad,
            0.f, data);
  };
  
  void sendData(const f32 *data) {
    i32 n = paddedWidth_*paddedHeight_;
    i32 nBytes = n*SizeofValueType(valType_);
    
    if (valType_ == Float32) {
      CurrentQueue().enqueueWriteBuffer(buffer_, false, 0, nBytes, data);
      return;
    }
    
    cl::Buffer temp(CurrentContext(), 0, n*sizeof(f32));
    CurrentQueue().enqueueWriteBuffer(temp, false, 0, n*sizeof(f32), data);
    float2imval(temp, n, buffer_);
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
