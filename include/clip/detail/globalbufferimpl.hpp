#pragma once
#ifndef CLIP_DETAIL_GLOBALBUFFERIMPL_H
#define CLIP_DETAIL_GLOBALBUFFERIMPL_H

#include "clip/basictypes.hpp"
#include "clip/detail/imagebufferimpl.hpp"

namespace clip {
namespace detail {

class GlobalBufferImpl : public ImageBufferImpl {
 protected:
  cl::Buffer buffer_;
  
  void initBufferFromFloatData(const f32 *data, cl_mem_flags flags) {
    flags |= CL_MEM_COPY_HOST_PTR;
    flags &= ~CL_MEM_USE_HOST_PTR;
    
    ImageData padded = makePaddedData(data);
    
    buffer_ = cl::Buffer(CurrentContext(), flags,
                         paddedWidth_*paddedHeight_*sizeof(f32),
                         &padded.data()[0]);
  }
  
 public:
  GlobalBufferImpl() : ImageBufferImpl(Global, &buffer_, 0, 0, 0, 0) {}
  
  GlobalBufferImpl(cl::Buffer data,
                   i32 width, i32 height,
                   i32 xAlign, i32 yAlign)
  : ImageBufferImpl(Global, &buffer_, width, height, xAlign, yAlign),
    buffer_(data) {
    initPaddingValues();
    // FIXME: Ensure the data we've been passed conforms with the dimensions
    // Just make sure the amount of memory data references is >= what we need
  }
  
  GlobalBufferImpl(const f32 *data,
                   i32 width, i32 height,
                   i32 xAlign, i32 yAlign,
                   cl_mem_flags flags)
  : ImageBufferImpl(Global, &buffer_, width, height, xAlign, yAlign)
  {
    if (data != NULL)
      initBufferFromFloatData(data, flags);
    else {
      initPaddingValues();
      i32 nBytes = paddedWidth_*paddedHeight_*sizeof(f32);
      buffer_ = cl::Buffer(CurrentContext(), flags, nBytes);
    }
    itemRange_ = cl::NDRange(paddedWidth_, paddedHeight_);
  }
  
  void fetchData(f32* data) const {
    std::vector<f32> temp(paddedWidth_*paddedHeight_);
    CurrentQueue().enqueueReadBuffer(buffer_, true, 0,
                                     paddedWidth_*paddedHeight_*sizeof(f32),
                                     &temp[0]);
    
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
    i32 nBytes = paddedWidth_*paddedHeight_*sizeof(f32);
    CurrentQueue().enqueueWriteBuffer(buffer_, false, 0, nBytes, data);
  }
  
  void copyInto(ImageBufferImpl *dest) const {
    assert(type_ == dest->type());
    assert(paddedWidth_ == dest->paddedWidth() &&
           paddedHeight_ == dest->paddedHeight());
    
    GlobalBufferImpl *realDest = static_cast<GlobalBufferImpl *>(dest);
    
    CurrentQueue().enqueueCopyBuffer(buffer_, realDest->buffer_, 0, 0,
                                     paddedWidth_*paddedHeight_*sizeof(f32));
  }
};

}}

#endif
