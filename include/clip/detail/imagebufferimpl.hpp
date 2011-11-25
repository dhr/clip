#pragma once
#ifndef CLIP_DETAIL_IMAGEBUFFERIMPL_H
#define CLIP_DETAIL_IMAGEBUFFERIMPL_H

#include "clip/basictypes.hpp"
#include "clip/imagebuffertypes.hpp"

namespace clip {
namespace detail {

inline void CalcPaddedSizes(i32 width, i32 height,
                            i32 xAlign, i32 yAlign,
                            i32* paddedWidth, i32* paddedHeight) {
  i32 xRem = width%xAlign;
  i32 yRem = height%yAlign;
  *paddedWidth = width + (xAlign - xRem)*(xRem != 0);
  *paddedHeight = height + (yAlign - yRem)*(yRem != 0);
}

class ImageBufferImpl {
 protected:
  ImageBufferType type_;
  ValueType valType_;
  cl::Memory* memory_;
  i32 width_;
  i32 height_;
  i32 xAlign_;
  i32 yAlign_;
  i32 paddedWidth_;
  i32 paddedHeight_;
  
  ImageBufferImpl(ImageBufferType type,
                  ValueType valType,
                  cl::Memory* memory,
                  i32 width, i32 height,
                  i32 xAlign, i32 yAlign)
  : type_(type), valType_(valType), memory_(memory),
    width_(width), height_(height),
    xAlign_(xAlign), yAlign_(yAlign) {}
  
  void initPaddingValues(i32 *xPaddingRet = NULL,
                         i32 *yPaddingRet = NULL) {
    CalcPaddedSizes(width_, height_, xAlign_, yAlign_,
                    &paddedWidth_, &paddedHeight_);
    
    if (xPaddingRet != NULL)
      *xPaddingRet = paddedWidth_ - width_;
    if (yPaddingRet != NULL)
      *yPaddingRet = paddedHeight_ - height_;
  }
  
  ImageData makePaddedData(const f32 *data) {
    i32 xPadding, yPadding;
    initPaddingValues(&xPadding, &yPadding);
    
    i32 leftPad = xPadding/2 + xPadding%2;
    i32 rightPad = xPadding/2;
    i32 bottomPad = yPadding/2 + yPadding%2;
    i32 topPad = yPadding/2;
    
    return ImageData(data, width_, height_,
                     leftPad, rightPad, bottomPad, topPad);
  }
  
  void unpadData(const f32* padded, f32* unpadded) const {
    i32 xPadding = paddedWidth_ - width_;
    i32 yPadding = paddedHeight_ - height_;
    
    i32 leftPad = xPadding/2 + xPadding%2;
    i32 rightPad = xPadding/2;
    i32 bottomPad = yPadding/2 + yPadding%2;
    i32 topPad = yPadding/2;
    
    PadData(padded, paddedWidth_, paddedHeight_,
            -leftPad, -rightPad, -bottomPad, -topPad,
            0.f, unpadded);
  }
  
  friend class BufferCache;
  
 public:
  virtual ~ImageBufferImpl() {}
  
  virtual void fetchData(f32* data) const = 0;
  
  inline const cl::Memory& mem() { return *memory_; }
  
  virtual void sendData(const f32 *data) = 0;
  virtual void copyInto(ImageBufferImpl *dest) const = 0;
  
  inline ImageBufferType type() { return type_; }
  inline ValueType valType() { return valType_; }
  inline i32 width() const { return width_; }
  inline i32 height() const { return height_; }
  inline i32 paddedWidth() const { return paddedWidth_; }
  inline i32 paddedHeight() const { return paddedHeight_; }
  inline i32 xAlign() const { return xAlign_; }
  inline i32 yAlign() const { return yAlign_; }
};

typedef std::tr1::shared_ptr<ImageBufferImpl> ImageBufferImplPtr;

}}

#endif
