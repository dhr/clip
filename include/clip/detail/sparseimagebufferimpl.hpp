#pragma once
#ifndef CLIP_DETAIL_SPARSEIMAGEBUFFERIMPL_H
#define CLIP_DETAIL_SPARSEIMAGEBUFFERIMPL_H

#include "clip/basictypes.hpp"
#include "clip/sparseimagedata.hpp"

namespace clip {

class SparseImageBuffer;

namespace detail {

class SparseImageBufferImpl {
 protected:
  cl::Buffer buffer_;
  i32 width_;
  i32 height_;
  i32 nelems_;
  
  friend class clip::SparseImageBuffer;
  
  SparseImageBufferImpl(const SparseImageData& data, cl_mem_flags flags)
  : width_(data.width()), height_(data.height()), nelems_(data.numElems())
  {
    flags |= CL_MEM_COPY_HOST_PTR;
    flags &= ~CL_MEM_USE_HOST_PTR;
    
    if (nelems_ > 0) {
      buffer_ = cl::Buffer(CurrentContext(), flags,
                           nelems_*sizeof(SparseValue),
                           const_cast<SparseValue*>(&data.data()[0]));
    }
  }
  
  SparseImageBufferImpl(cl::Buffer data, i32 width, i32 height, i32 nelems)
  : buffer_(data), width_(width), height_(height), nelems_(nelems) {}
  
 public:
  void fetchData(SparseValue* data) const {
    CurrentQueue().enqueueReadBuffer(buffer_, true, 0,
                                     nelems_*sizeof(SparseValue),
                                     data);
  }
  
  void sendData(const SparseValue* values) {
    i32 nBytes = nelems_*sizeof(SparseValue);
    CurrentQueue().enqueueWriteBuffer(buffer_, false, 0, nBytes, values);
  }
  
  inline const cl::Memory &mem() { return buffer_; }
  
  inline i32 width() const { return width_; }
  inline i32 height() const { return height_; }
  inline i32 numElems() const { return nelems_; }
};

typedef std::tr1::shared_ptr<SparseImageBufferImpl> SparseImageBufferImplPtr;

}}

#endif
