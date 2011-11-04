#ifndef CLIP_SPARSEIMAGEDATA_H
#define CLIP_SPARSEIMAGEDATA_H

#include <algorithm>
#include <iterator>

#include "clip/basictypes.hpp"
#include "clip/imagedata.hpp"

namespace clip {

struct SparseValue {
  i16 x;
  i16 y;
  f32 val;
};

typedef std::vector<SparseValue> SparseImageDataValues;
typedef std::tr1::shared_ptr<SparseImageDataValues> SparseImageDataValuesPtr;

class SparseImageData {
 protected:
  SparseImageDataValuesPtr data_;
  i32 width_;
  i32 height_;
  
  static inline bool sparseValueCompare(SparseValue a, SparseValue b) {
    return a.x == b.x ? (a.y == b.y ? false : a.y < b.y) : a.x < b.x;
  }
  
 public:
  SparseImageData() {}
  
  SparseImageData(i32 width, i32 height, i32 nelems = 0)
  : data_(SparseImageDataValuesPtr(new SparseImageDataValues(nelems))),
    width_(width), height_(height) {}
  
  SparseImageData(const SparseImageData &other, bool deepCopy = false)
  : width_(other.width_), height_(other.height_)
  {
    if (deepCopy) {
      data_ = SparseImageDataValuesPtr(new SparseImageDataValues(*other.data_));
    }
    else {
      data_ = other.data_;
    }
  }
  
	SparseImageData(const ImageData &imData, f32 zeroThresh = 0.f)
  : data_(SparseImageDataValuesPtr(new SparseImageDataValues())),
    width_(imData.width()), height_(imData.height())
  {
    i32 nnz = imData.numNonZeros(zeroThresh);
    data_->reserve(nnz);
    
    if (nnz > 0) {
      i32 i = 0;
      for (u16 y = 0; y < height_; y++) {
        for (u16 x = 0; x < width_; x++, i++) {
          f32 val = imData[i];
          if (std::abs(val) > zeroThresh) {
            SparseValue sv = {x, y, val};
            data_->push_back(sv);
          }
        }
      }
      
      std::sort(data_->begin(), data_->end(), sparseValueCompare);
    }
  }
  
  SparseImageData(const i16 *xs, const i16 *ys, const f32 *vals,
                  i32 n, i32 width, i32 height)
  : data_(new SparseImageDataValues()), width_(width), height_(height)
  {
    data_->reserve(n);
    for (i32 i = 0; i < n; i++) {
      SparseValue val = {xs[i], ys[i], vals[i]};
      (*data_)[i] = val;
    }
    
    std::sort(data_->begin(), data_->end(), sparseValueCompare);
  }
  
  SparseImageData clone() {
    return SparseImageData(*this, true);
  }
  
  inline i32 numElems() const { return i32((*data_).size()); }
  inline i32 width() const { return width_; }
  inline i32 height() const { return height_; }
  
  inline i32 findIndex(i16 x, i16 y) const {
    SparseValue target = {x, y, 0.f};
    SparseImageDataValues::iterator it =
      std::lower_bound(data_->begin(), data_->end(),
                       target, sparseValueCompare);
    if (it == data_->end() || it->x != x || it->y != y) return -1;
    return i32(std::distance(data_->begin(), it));
  }
  
  inline i32 findOrCreateIndex(i16 x, i16 y) {
    SparseValue target = {x, y, 0.f};
    SparseImageDataValues::iterator it =
      std::lower_bound(data_->begin(), data_->end(),
                       target, sparseValueCompare);
    if (it->x != x || it->y != y)
      it = data_->insert(it, target);
    return i32(std::distance(data_->begin(), it));
  }
  
  SparseImageDataValues &data() { return *data_; }
  const SparseImageDataValues &data() const { return *data_; }
  
  inline f32 &operator()(i16 x, i16 y) {
    return (*data_)[findOrCreateIndex(x, y)].val;
  }
  
  inline f32 operator()(i16 x, i16 y) const {
    i32 i = findIndex(x, y);
    if (i < 0) return 0.f;
    return (*data_)[i].val;
  }
  
  inline SparseValue &operator[](i32 i) { return (*data_)[i]; }
  const inline SparseValue &operator[](i32 i) const { return (*data_)[i]; }
  
  ImageData inflate() {
    ImageData imData(width_, height_);
    SparseImageDataValues::iterator it;
    for (it = data_->begin(); it != data_->end(); ++it) {
      SparseValue sv = *it;
      imData(sv.x, sv.y) = sv.val;
    }
    
    return imData;
  }
  
  SparseImageData& prune(f32 eps = 0.f) {
    SparseImageDataValues::iterator it;
    for (it = data_->begin(); it != data_->end(); ++it) {
      if (std::abs(it->val) <= eps)
        it = data_->erase(it);
    }
    
    return *this;
  }
  
  SparseImageData& balance() {
    f32 posSum = 0;
    f32 negSum = 0;
    for (i32 i = 0; i < i32(data_->size()); i++) {
      if ((*data_)[i].val > 0) posSum += (*data_)[i].val;
      else negSum -= (*data_)[i].val;
    }
    for (i32 i = 0; i < i32(data_->size()); i++)
      if ((*data_)[i].val < 0) (*data_)[i].val *= posSum/negSum;
    
    return *this;
  }
  
  SparseImageData &normalize() {
    f32 pos = 0;
    for (i32 i = 0; i < width_*height_; i++)
      if ((*data_)[i].val > 0) pos += (*data_)[i].val;
    for (i32 i = 0; i < width_*height_; i++)
      (*data_)[i].val /= pos;
    
    return *this;
  }
};

}

#endif
