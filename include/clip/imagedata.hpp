#pragma once
#ifndef CLIP_IMAGEDATA_H
#define CLIP_IMAGEDATA_H

#include <cassert>
#include <cstring>

#include <valarray>

#include "clip/alignmentutil.hpp"
#include "clip/basictypes.hpp"
#include "clip/detail/memutil.hpp"

namespace clip {

typedef std::valarray<f32> ImageDataValues;
typedef std::tr1::shared_ptr<ImageDataValues> ImageDataValuesPtr;

class ImageData {
 protected:
  ImageDataValuesPtr data_;
  i32 width_;
  i32 height_;
  
  void initWithPadding(const f32 *data,
                       i32 leftPad, i32 rightPad,
                       i32 bottomPad, i32 topPad,
                       f32 padVal) {
    i32 paddedWidth = width_ + leftPad + rightPad;
    i32 paddedHeight = height_ + bottomPad + topPad;
    assert(paddedWidth >= 0 && paddedHeight >= 0);
    
    ImageDataValuesPtr padded
      (new ImageDataValues(padVal, paddedWidth*paddedHeight));
    f32 *paddedData = &(*padded)[0];
    
    PadData(data, width_, height_,
            leftPad, rightPad, bottomPad, topPad,
            padVal, paddedData);
    
    data_ = padded;
    width_ = paddedWidth;
    height_ = paddedHeight;
  }
  
 public:
  ImageData() {}
  
  ImageData(i32 width, i32 height)
  : data_(ImageDataValuesPtr(new ImageDataValues(width*height))),
    width_(width), height_(height) {}
  
	ImageData(const ImageData &other, bool deepCopy = false)
  : width_(other.width_), height_(other.height_)
  {
    if (deepCopy)
      data_ = ImageDataValuesPtr(new ImageDataValues(*other.data_));
    else
      this->data_ = other.data_;
  }
  
	ImageData(const ImageData &other,
            i32 leftPad, i32 rightPad,
            i32 bottomPad, i32 topPad,
            f32 padVal = 0.f)
  : width_(other.width_), height_(other.height_)
  {
    initWithPadding(&(*other.data_)[0],
                    leftPad, rightPad, bottomPad, topPad,
                    padVal);
  }
  
  ImageData(const f32 *data, i32 width, i32 height,
            i32 leftPad = 0, i32 rightPad = 0,
            i32 bottomPad = 0, i32 topPad = 0,
            f32 padVal = 0.f)
  : width_(width), height_(height)
  {
    initWithPadding(data, leftPad, rightPad, bottomPad, topPad, padVal);
  }
  
  ImageData clone() {
    return ImageData(*this, true);
  }
  
  ImageData& pad(i32 leftPad, i32 rightPad, i32 bottomPad, i32 topPad,
           f32 padVal = 0.f) {
    initWithPadding(&(*data_)[0], leftPad, rightPad, bottomPad, topPad, padVal);
    return *this;
  }
  
  ImageData& align(i32 xAlign, i32 yAlign, f32 padVal = 0.f) {
    i32 paddedWidth, paddedHeight;
    CalcAlignedSizes(width_, height_, xAlign, yAlign,
                     &paddedWidth, &paddedHeight);
    i32 xPadding = paddedWidth - width_;
    i32 yPadding = paddedHeight - height_;
    i32 leftPad = xPadding/2 + xPadding%2;
    i32 rightPad = xPadding/2;
    i32 bottomPad = yPadding/2 + yPadding%2;
    i32 topPad = yPadding/2;
    initWithPadding(&(*data_)[0], leftPad, rightPad, bottomPad, topPad, padVal);
    return *this;
  }
  
  inline i32 numElems() const { return i32((*data_).size()); }
  inline i32 width() const { return width_; }
  inline i32 height() const { return height_; }
  
  inline i32 index(i32 x, i32 y) const {
    return y*width_ + x;
  }
  
  ImageDataValues& data() { return *data_; }
  const ImageDataValues& data() const { return *data_; }
  
  inline f32& operator()(i32 x, i32 y) {
    return (*data_)[index(x, y)];
  }
  
  const inline f32& operator()(i32 x, i32 y) const {
    return (*data_)[index(x, y)];
  }
  
  inline f32& operator[](i32 i) { return (*data_)[i]; }
  const inline f32& operator[](i32 i) const { return (*data_)[i]; }
  
  ImageData& balance() {
    f32 posSum = 0;
    f32 negSum = 0;
    for (i32 i = 0; i < i32(data_->size()); i++) {
      if ((*data_)[i] > 0) posSum += (*data_)[i];
      else negSum -= (*data_)[i];
    }
    for (i32 i = 0; i < i32(data_->size()); i++)
      if ((*data_)[i] < 0) (*data_)[i] *= posSum/negSum;
    
    return *this;
  }
  
  ImageData& normalize() {
    f32 pos = 0;
    for (i32 i = 0; i < width_*height_; i++)
      if ((*data_)[i] > 0) pos += (*data_)[i];
    *data_ /= pos;
    
    return *this;
  }
  
  i32 numNonZeros(f32 eps = 0) const {
    i32 n = 0;
    for (i32 i = 0; i < i32(data_->size()); i++)
      n += std::abs((*data_)[i]) > eps;
    return n;
  }
};

}

#endif
