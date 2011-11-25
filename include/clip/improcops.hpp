#pragma once
#ifndef CLIP_IMPROCOPS_H
#define CLIP_IMPROCOPS_H

#include <iterator>
#include <list>
#include <vector>

#include "clip/basictypes.hpp"
#include "clip/bufferops.hpp"
#include "clip/sparseimagebuffer.hpp"

namespace clip {

template<typename InIterator, typename OutIterator>
void LoadFilters(InIterator begin, InIterator end, OutIterator output) {
  i32 xAlign = (CurrentBufferType() == Texture)*3 + 1;
  for (InIterator i = begin; i != end; ++i) {
    *output++ = ImageBuffer(*i, CurrentFilterValueType(),
                            xAlign, 1, Global, false);
  }
}

template<typename InIterator, typename OutIterator>
void LoadSparseFilters(InIterator begin, InIterator end, OutIterator output) {
  for (InIterator i = begin; i != end; ++i)
    *output++ = SparseImageBuffer(*i, CurrentFilterValueType());
}

inline ImageBuffer Filter(const ImageBuffer& image,
                          const ImageBuffer& filter,
                          ImageBuffer o) {
  static CachedKernel cache("filter");
  cl::Kernel& kernel = cache.get();
  
  if (CurrentDeviceType() == CPU || image.type() == Texture) {
    kernel.setArg(0, image.mem());
    kernel.setArg(1, filter.mem());
    kernel.setArg(2, filter.paddedWidth());
    kernel.setArg(3, filter.paddedHeight());
    kernel.setArg(4, o.mem());
    Enqueue(kernel,
            cl::NullRange,
            cl::NDRange(image.paddedWidth()/((image.type() == Texture)*3 + 1),
                        image.paddedHeight()),
            cl::NullRange);
  }
  else {
    i32 kernWidth = filter.width();
    i32 halfKernWidth = kernWidth/2;
    i32 kernHeight = filter.height();
    i32 halfKernHeight = kernHeight/2;
    i32 apronRem = halfKernWidth%4;
    i32 apronWidth = halfKernWidth + (apronRem ? 4 - apronRem : 0);
    cl::LocalSpaceArg imCacheSize = 
      {(image.xAlign() + 2*apronWidth)*
       (image.yAlign() + 2*(halfKernHeight) - !(kernHeight%2))*
       sizeof(f32)};
    cl::LocalSpaceArg filtCacheSize = {kernWidth*kernHeight*sizeof(f32)};
    
    kernel.setArg(0, image.mem());
    kernel.setArg(1, imCacheSize);
    kernel.setArg(2, filter.mem());
    kernel.setArg(3, filtCacheSize);
    kernel.setArg(4, kernWidth);
    kernel.setArg(5, kernHeight);
    kernel.setArg(6, o.mem());
    
    Enqueue(kernel,
            cl::NullRange,
            cl::NDRange(image.paddedWidth()/4, image.paddedHeight()),
            cl::NDRange(image.xAlign()/4, image.yAlign()));
  }
  
  return o;
}

inline ImageBuffer Filter(const ImageBuffer& image,
                          const SparseImageBuffer& filter,
                          ImageBuffer o) {
  static CachedKernel cache("filter_sparse");
  
  if (CurrentDeviceType() == CPU || image.type() == Texture) {
    cl::Kernel& kernel = cache.get();
  
    kernel.setArg(0, image.mem());
    kernel.setArg(1, filter.mem());
    kernel.setArg(2, filter.width());
    kernel.setArg(3, filter.height());
    kernel.setArg(4, filter.numElems());
    kernel.setArg(5, o.mem());
    Enqueue(kernel, o);
  }
  else {
    if (filter.numElems() == 0)
      return Memset(o, 0);
  
    cl::Kernel& kernel = cache.get();
    
    i32 kernWidth = filter.width();
    i32 halfKernWidth = kernWidth/2;
    i32 kernHeight = filter.height();
    i32 halfKernHeight = kernHeight/2;
    i32 apronRem = halfKernWidth%4;
    i32 apronWidth = halfKernWidth + (apronRem ? 4 - apronRem : 0);
    cl::LocalSpaceArg imCacheSize = 
      {(image.xAlign() + 2*apronWidth)*
       (image.yAlign() + 2*(halfKernHeight) - !(kernHeight%2))*
       sizeof(f32)};
    cl::LocalSpaceArg filtCacheSize = {filter.numElems()*sizeof(SparseValue)};
    
    kernel.setArg(0, image.mem());
    kernel.setArg(1, imCacheSize);
    kernel.setArg(2, filter.mem());
    kernel.setArg(3, filtCacheSize);
    kernel.setArg(4, kernWidth);
    kernel.setArg(5, kernHeight);
    kernel.setArg(6, filter.numElems());
    kernel.setArg(7, o.mem());
    
    Enqueue(kernel,
            cl::NullRange,
            cl::NDRange(image.paddedWidth()/4, image.paddedHeight()),
            cl::NDRange(image.xAlign()/4, image.yAlign()));
  }
  
  return o;
}

inline ImageBuffer Filter(const ImageBuffer& image,
                          const ImageBuffer& filter) {
  return Filter(image, filter, ~image);
}

inline ImageBuffer Filter(const ImageBuffer& image,
                          const SparseImageBuffer& filter) {
  return Filter(image, filter, ~image);
}

namespace detail {

inline cl::Buffer& filterMemory() {
  static ContextID contextID = CurrentContextID();
  static cl::Buffer filterMemory(CurrentContext(), CL_MEM_READ_ONLY, 10000);
  
  if (contextID != CurrentContextID())
    filterMemory = cl::Buffer(CurrentContext(), CL_MEM_READ_ONLY, 10000);
  
  return filterMemory;
}

}

inline ImageBuffer Filter(const ImageBuffer& image, const ImageData& filter,
                          ImageBuffer o) {
  ImageBuffer filterBuffer(detail::filterMemory(),
                           filter.width(), filter.height(),
                           CurrentFilterValueType(),
                           (image.type() == Texture)*3 + 1, 1,
                           Global);
  filterBuffer.sendData(filter);
  return Filter(image, filterBuffer, o);
}

inline ImageBuffer Filter(const ImageBuffer& image, const ImageData& filter) {
  return Filter(image, filter, ~image);
}

inline ImageBuffer Filter(const ImageBuffer& image,
                          const SparseImageData& filter,
                          ImageBuffer o) {
  // This reserves space for a filter with maximum size 10000 bytes
  SparseImageBuffer filterBuffer(detail::filterMemory(),
                                 filter.width(), filter.height(),
                                 filter.numElems(),
                                 CurrentFilterValueType());
  filterBuffer.sendData(filter);
  return Filter(image, filterBuffer, o);
}

inline ImageBuffer Filter(const ImageBuffer& image,
                          const SparseImageData& filter) {
  return Filter(image, filter, ~image);
}

class GaussianBlurOp {
  ImageData gaussianX_;
  ImageData gaussianY_;
  
  f32 sigma_;
  
  void initGaussians() {
    f32 sigma2 = sigma_*sigma_;
    i32 center = gaussianX_.numElems()/2;
    for (i32 i = 0; i < gaussianX_.numElems(); ++i) {
      i32 x = i - center;
      gaussianX_[i] = exp(-x*x/(2*sigma2));
    }
    gaussianX_.normalize();
    gaussianY_ = ImageData(&gaussianX_.data()[0], 1, gaussianX_.width());
  }
  
 public:
  GaussianBlurOp(f32 sigma)
  : gaussianX_(std::max(4*i32(round(sigma)) + 1, 3), 1),
    sigma_(sigma)
  {
    initGaussians();
  }
  
  template<typename T>
  T operator()(const T& i1, T o) {
    Filter(i1, gaussianX_, o);
    return Filter(o, gaussianY_, o);
  }
  
  template<typename T>
  T operator()(const T& i1) { return operator()(i1, ~i1); }
};

class GradientOp {
  ImageData sobelX_;
  ImageData sobelY_;
  
  void initGradients() {
    f32 sobelX[] = {-1,  0,  1,
                    -1,  0,  1,
                    -1,  0,  1};
    
    // ImageData's (0, 0) starts at lower left corner, this gets flipped
    f32 sobelY[] = {-1, -1, -1,
                     0,  0,  0,
                     1,  1,  1};
    
    sobelX_ = ImageData(sobelX, 3, 3).normalize();
    sobelY_ = ImageData(sobelY, 3, 3).normalize();
  }
  
 public:
  GradientOp() {
    initGradients();
  }
  
  template<typename T>
  void operator()(const T& i1, T gx, T gy) {
    Filter(i1, sobelX_, gx);
    Filter(i1, sobelY_, gy);
  }
};

}

#endif
