#ifndef CLIP_SPARSEIMAGEBUFFER_H
#define CLIP_SPARSEIMAGEBUFFER_H

#include "clip/basictypes.hpp"
#include "clip/sparseimagedata.hpp"
#include "clip/detail/sparseimagebufferimpl.hpp"

namespace clip {

class SparseImageBuffer {  
 protected:
  detail::SparseImageBufferImplPtr impl_;
  
 public:
  SparseImageBuffer() {}
  
  explicit
  SparseImageBuffer(const SparseImageData &data, ValueType valType,
                    cl_mem_flags flags = 0) {
    impl_ = detail::SparseImageBufferImplPtr(
      new detail::SparseImageBufferImpl(data, valType, flags));
  }
  
  explicit
  SparseImageBuffer(const ImageData &data, ValueType valType,
                    cl_mem_flags flags = 0) {
    SparseImageData sid(data);
    detail::SparseImageBufferImpl* p =
      new detail::SparseImageBufferImpl(sid, valType, flags);
    impl_ = detail::SparseImageBufferImplPtr(p);
  }
  
  SparseImageBuffer(cl::Buffer data, i32 width, i32 height, i32 nelems,
                    ValueType valType) {
    impl_ = detail::SparseImageBufferImplPtr(
      new detail::SparseImageBufferImpl(data, width, height, nelems, valType));
  }
  
  void fetchData(SparseValue* data) const {
    impl_->fetchData(data);
  }
  
  SparseImageData fetchData() const {
    SparseImageData data(impl_->width(), impl_->height(), impl_->numElems());
    impl_->fetchData(&data.data()[0]);
    return data;
  }
  
  void sendData(const SparseValue* data) {
    impl_->sendData(data);
  }
  
  void sendData(const SparseImageData& data) {
    assert(data.width() == width() && data.height() == height() &&
           data.numElems() == numElems() &&
           "Can't send SparseImageData with different number of elements");
    impl_->sendData(&data[0]);
  }
  
  inline const cl::Memory &mem() const { return impl_->mem(); }
  inline i32 width() const { return impl_->width(); }
  inline i32 height() const { return impl_->height(); }
  inline i32 numElems() const { return impl_->numElems(); }
  inline ValueType valType() const { return impl_->valType(); }
  
  inline bool valid() const { return impl_.get() != NULL && mem()() != NULL; }
};

}

#endif
