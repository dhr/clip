#pragma once
#ifndef CLIP_IMAGEBUFFER_H
#define CLIP_IMAGEBUFFER_H

#include <cassert>
#include <cstring>

#include "clip/basictypes.hpp"
#include "clip/clstate.hpp"
#include "clip/imagedata.hpp"
#include "clip/detail/buffercache.hpp"
#include "clip/detail/imagebufferimpl.hpp"
#include "clip/detail/texturebufferimpl.hpp"
#include "clip/detail/globalbufferimpl.hpp"
#include "clip/detail/memutil.hpp"

#define DefaultBufferType (USE_TEXTURES ? Texture : Global)
#define DefaultXAlign 64
#define DefaultYAlign 16

namespace clip {

void ClearBufferCache() {
  detail::BufferCache::instance().clear();
}

class ImageBuffer {  
 protected:
  detail::ImageBufferImplPtr impl_;
  bool cacheable_;
  
  void initialize(const f32 *data, i32 width, i32 height, ValueType valType,
                  i32 xAlign, i32 yAlign, ImageBufferType type) {
    using namespace detail;
    ImageBufferImplPtr impl =
      BufferCache::instance().retrieve(type, valType, width, height,
                                       xAlign, yAlign);
    
    if (impl.get()) {
      impl_ = impl;
      if (data) sendData(data);
    }
    else if (type == Global) {
      impl_ = ImageBufferImplPtr(
        new GlobalBufferImpl(data, width, height, valType, xAlign, yAlign, 0));
    }
    else {
      impl_ = ImageBufferImplPtr(
        new TextureBufferImpl(data, width, height, valType, xAlign, yAlign, 0));
    }
  }
  
 public:
  ImageBuffer() {}
  
  ImageBuffer(i32 width, i32 height)
  : cacheable_(true)
  {
    initialize(NULL, width, height, CurrentImBufValueType(),
               DefaultXAlign, DefaultYAlign, DefaultBufferType);
  }
  
  ImageBuffer(i32 width, i32 height, ValueType valType,
              i32 xAlign = DefaultXAlign, i32 yAlign = DefaultYAlign,
              ImageBufferType type = DefaultBufferType, bool cacheable = true)
  : cacheable_(cacheable)
  {
    initialize(NULL, width, height, valType, xAlign, yAlign, type);
  }
  
  explicit
  ImageBuffer(const ImageData &data)
  : cacheable_(true)
  {
    initialize(&data.data()[0], data.width(), data.height(),
               CurrentImBufValueType(),
               DefaultXAlign, DefaultYAlign, DefaultBufferType);
  }
  
  explicit
  ImageBuffer(const ImageData &data, ValueType valType,
              i32 xAlign = DefaultXAlign, i32 yAlign = DefaultYAlign,
              ImageBufferType type = DefaultBufferType, bool cacheable = true)
  : cacheable_(cacheable)
  {
    initialize(&data.data()[0], data.width(), data.height(), valType,
               xAlign, yAlign, type);
  }
  
  ImageBuffer(const f32* data, i32 width, i32 height)
  : cacheable_(true)
  {
    initialize(data, width, height, CurrentImBufValueType(),
               DefaultXAlign, DefaultYAlign, DefaultBufferType);
  }
  
  ImageBuffer(const f32* data, i32 width, i32 height, ValueType valType,
              i32 xAlign = DefaultXAlign, i32 yAlign = DefaultYAlign,
              ImageBufferType type = DefaultBufferType, bool cacheable = true)
  : cacheable_(cacheable)
  {
    initialize(data, width, height, valType, xAlign, yAlign, type);
  }
  
  ImageBuffer(cl::Buffer data, i32 width, i32 height, ValueType valType,
              i32 xAlign = DefaultXAlign, i32 yAlign = DefaultYAlign,
              bool cacheable = false)
  : cacheable_(cacheable)
  {
    impl_ = detail::ImageBufferImplPtr(
      new detail::GlobalBufferImpl(data, width, height, valType,
                                   xAlign, yAlign));
  }
  
  ImageBuffer(cl::Image2D data, i32 width, i32 height, ValueType valType,
              i32 xAlign = DefaultXAlign, i32 yAlign = DefaultYAlign,
              bool cacheable = false)
  : cacheable_(cacheable)
  {
    impl_ = detail::ImageBufferImplPtr(
      new detail::TextureBufferImpl(data, width, height, valType,
                                    xAlign, yAlign));
  }
  
  ~ImageBuffer() {
    if (impl_.use_count() == 1 && cacheable_)
      detail::BufferCache::instance().release(impl_);
  }
  
  void fetchData(f32 *data) const {
    return impl_->fetchData(data);
  }
  
  ImageData fetchData() const {
    ImageData data(impl_->width(), impl_->height());
    impl_->fetchData(&data.data()[0]);
    return data;
  }
  
  inline void sendData(const f32 *data) {
    impl_->sendData(data);
  }
  
  void sendData(const ImageData& data) {
    sendData(&data.data()[0]);
  }
  
  inline void copyInto(const ImageBuffer& dest) const {
    impl_->copyInto(dest.impl_.get());
  }
  
  inline ImageBuffer operator~() const {
    return ImageBuffer(impl_->width(), impl_->height(), impl_->valType(),
                       impl_->xAlign(), impl_->yAlign(), impl_->type());
  }
  
  inline ImageBuffer clone() const {
    ImageBuffer sizeClone = ~(*this);
    copyInto(sizeClone);
    return sizeClone;
  }
  
  inline const cl::Memory &mem() const { return impl_->mem(); }
  inline const cl::NDRange &offset() const { return impl_->offset(); }
  inline const cl::NDRange &itemRange() const { return impl_->itemRange(); }
  inline const cl::NDRange &groupRange() const { return impl_->groupRange(); }
  
  inline ImageBufferType type() const { return impl_->type(); }
  inline bool cacheable() { return cacheable_; }
  inline void setCacheable(bool cacheable) { cacheable_ = cacheable; }
  inline i32 width() const { return impl_->width(); }
  inline i32 height() const { return impl_->height(); }
  inline i32 xAlign() const { return impl_->xAlign(); }
  inline i32 yAlign() const { return impl_->yAlign(); }
  inline i32 paddedWidth() const { return impl_->paddedWidth(); }
  inline i32 paddedHeight() const { return impl_->paddedHeight(); }
  
  inline bool valid() const { return impl_.get() != NULL && mem()() != NULL; }
};

}

#endif
