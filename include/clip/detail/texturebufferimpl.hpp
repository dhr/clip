#pragma once
#ifndef CLIP_DETAIL_TEXTUREBUFFERIMPL_H
#define CLIP_DETAIL_TEXTUREBUFFERIMPL_H

#include "clip/basictypes.hpp"
#include "clip/detail/imagebufferimpl.hpp"

namespace clip {
namespace detail {

class TextureBufferImpl : public ImageBufferImpl {
 protected:
  cl::Image2D buffer_;
  
  void initBufferFromFloatData(const f32 *data, cl_mem_flags flags) {
    flags |= CL_MEM_COPY_HOST_PTR;
    flags &= ~CL_MEM_USE_HOST_PTR;
    
    ImageData padded = makePaddedData(data);
    
    buffer_ = cl::Image2D(CurrentContext(), flags,
                          cl::ImageFormat(CL_RGBA, CL_FLOAT),
                          paddedWidth_/4, paddedHeight_,
                          paddedWidth_*sizeof(f32), &padded.data()[0]);
  }
  
 public:
  TextureBufferImpl()
  : ImageBufferImpl(Texture, Float32, &buffer_, 0, 0, 0, 0) {}
  
  TextureBufferImpl(cl::Image2D data, i32 width, i32 height, ValueType valType,
                    i32 xAlign, i32 yAlign)
  : ImageBufferImpl(Texture, valType, &buffer_, width, height, xAlign, yAlign),
    buffer_(data)
  {
    initPaddingValues();
    // FIXME: Ensure the data we've been passed conforms with the dimensions
  }
  
  TextureBufferImpl(const f32 *data, i32 width, i32 height, ValueType valType,
                    i32 xAlign, i32 yAlign, cl_mem_flags flags)
  : ImageBufferImpl(Texture, valType, &buffer_, width, height, xAlign, yAlign)
  {
    if (data != NULL)
      initBufferFromFloatData(data, flags);
    else {
      initPaddingValues();
      buffer_ = cl::Image2D(CurrentContext(), flags,
                            cl::ImageFormat(CL_RGBA, CL_FLOAT),
                            paddedWidth_/4, paddedHeight_, 0);
    }
    itemRange_ = cl::NDRange(paddedWidth_/4, paddedHeight_);
  }
  
  void fetchData(f32* data) const {
    std::vector<f32> temp(paddedWidth_*paddedHeight_);
  
    cl::size_t<3> origin;
    origin[0] = origin[1] = origin[2] = 0;
    cl::size_t<3> region;
    region[0] = paddedWidth_/4;
    region[1] = paddedHeight_;
    region[2] = 1;
    CurrentQueue().enqueueReadImage(buffer_, true, origin, region,
                                    paddedWidth_*4, 0, &temp[0]);
    unpadData(&temp[0], data);
  };
  
  void sendData(const f32 *data) {
    cl::size_t<3> orig;
    orig[0] = orig[1] = orig[2] = 0;
    cl::size_t<3> region;
    region[0] = paddedWidth_/4;
    region[1] = paddedHeight_;
    region[2] = 1;
    CurrentQueue().enqueueWriteImage(buffer_, false, orig, region, 0, 0,
                                     const_cast<f32 *>(data));
  }
  
  void copyInto(ImageBufferImpl *dest) const {
    assert(type_ == dest->type());
    assert(paddedWidth_ == dest->paddedWidth() &&
           paddedHeight_ == dest->paddedHeight());
    
    TextureBufferImpl *realDest = static_cast<TextureBufferImpl *>(dest);
    
    cl::size_t<3> origin;
    origin[0] = origin[1] = origin[2] = 0;
    cl::size_t<3> region;
    region[0] = paddedWidth_/4;
    region[1] = paddedHeight_;
    region[2] = 1;
    CurrentQueue().enqueueCopyImage(buffer_, realDest->buffer_,
                                    origin, origin, region);
  }
};

}}

#endif
