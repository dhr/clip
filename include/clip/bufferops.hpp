#ifndef CLIP_BUFFEROPS_H
#define CLIP_BUFFEROPS_H

#include <map>

#include "clip/basictypes.hpp"
#include "clip/clstate.hpp"
#include "clip/imagebuffer.hpp"
#include "clip/ioadaptors.hpp"

namespace clip {

inline void Enqueue(cl::Kernel& k, const ImageBuffer& b) {
  Enqueue(k, b.offset(), b.itemRange(), b.groupRange());
}

inline ImageBuffer DoBasicOp(cl::Kernel& kernel, const ImageBuffer& i1,
                             ImageBuffer o) {
  kernel.setArg(0, i1.mem());
  kernel.setArg(1, o.mem());
  Enqueue(kernel, o);
  return o;
}

inline ImageBuffer DoBasicOp(cl::Kernel& kernel,
                             const ImageBuffer& i1, const ImageBuffer& i2,
                             ImageBuffer o) {
  kernel.setArg(0, i1.mem());
  kernel.setArg(1, i2.mem());
  kernel.setArg(2, o.mem());
  Enqueue(kernel, o);
  return o;
}

inline ImageBuffer DoBasicOp(cl::Kernel& kernel,
                             const ImageBuffer& i1, const ImageBuffer& i2,
                             const ImageBuffer& i3, const ImageBuffer& i4,
                             ImageBuffer o) {
  kernel.setArg(0, i1.mem());
  kernel.setArg(1, i2.mem());
  kernel.setArg(2, i3.mem());
  kernel.setArg(3, i4.mem());
  kernel.setArg(4, o.mem());
  Enqueue(kernel, o);
  return o;
}

class BasicOp {
 protected:
  CachedKernel cache_;
 
 public:
  BasicOp(std::string name) : cache_(name) {}
};

struct UnaryOp : public BasicOp {
  typedef ImageBuffer result_type;
  
  UnaryOp(std::string name) : BasicOp(name) {}
  
  inline ImageBuffer operator()(const ImageBuffer& i1, ImageBuffer o) {
    return DoBasicOp(cache_.get(), i1, o);
  }
  
  inline ImageBuffer operator()(const ImageBuffer& i1) {
    return operator()(i1, ~i1);
  }
};

struct BinaryOp : public BasicOp {
  typedef ImageBuffer result_type;
  
  BinaryOp(std::string name) : BasicOp(name) {}
  
  inline ImageBuffer operator()(const ImageBuffer& i1,
                                const ImageBuffer& i2,
                                ImageBuffer o) {
    return DoBasicOp(cache_.get(), i1, i2, o);
  }
  
  inline ImageBuffer operator()(const ImageBuffer& i1,
                                const ImageBuffer& i2) {
    return operator()(i1, i2, ~i1);
  }
};

struct ScalarOp : public BasicOp {
  typedef ImageBuffer result_type;
  
  ScalarOp(std::string name) : BasicOp(name) {}
  
  inline ImageBuffer operator()(const ImageBuffer& i1, f32 amt,
                                ImageBuffer o) {
    cl::Kernel& kernel = cache_.get();
    kernel.setArg(0, i1.mem());
    kernel.setArg(1, amt);
    kernel.setArg(2, o.mem());
    Enqueue(kernel, o);
    return o;
  }
  
  inline ImageBuffer operator()(const ImageBuffer& i1, f32 amt) {
    return operator()(i1, amt, ~i1);
  }
};

class MergeOp {
 protected:
  CachedKernel cache2_;
  CachedKernel cache4_;
  
 public:
  typedef ImageBuffer result_type;
  
  MergeOp(std::string baseName)
  : cache2_(baseName + "2"), cache4_(baseName + "4") {}
 
  inline ImageBuffer operator()(const ImageBuffer& i1,
                                const ImageBuffer& i2,
                                ImageBuffer o) {
    return DoBasicOp(cache2_.get(), i1, i2, o);
  }
 
  inline ImageBuffer operator()(const ImageBuffer& i1,
                                const ImageBuffer& i2) {
    return operator()(i1, i2, ~i1);
  }

  inline ImageBuffer operator()(const ImageBuffer& i1,
                                const ImageBuffer& i2,
                                const ImageBuffer& i3,
                                const ImageBuffer& i4,
                                ImageBuffer o) {
    return DoBasicOp(cache4_.get(), i1, i2, i3, i4, o);
  }
 
  inline ImageBuffer operator()(const ImageBuffer& i1,
                                const ImageBuffer& i2,
                                const ImageBuffer& i3,
                                const ImageBuffer& i4) {
    return operator()(i1, i2, i3, i4, ~i1);
  }
};

template<typename Binary>
ImageBuffer Merge(Binary op, i32 n, InputAdaptor& input, ImageBuffer output) {
  if (n >= 2) {
    op(input.next(), input.next(), output);
    n -= 2;
  }
  else if (n == 1) {
    input.next().copyInto(output);
    return output;
  }
  else return output;
  
  while (n-- > 0)
    op(output, input.next(), output);
  
  return output;
}

inline ImageBuffer Merge(MergeOp op, i32 n, InputAdaptor& input,
                         ImageBuffer output) {
  if (n >= 4) {
    op(input.next(), input.next(), input.next(), input.next(), output);
    n -= 4;
  }
  else if (n >= 2) {
    op(input.next(), input.next(), output);
    n -= 2;
  }
  else if (n == 1) {
    input.next().copyInto(output);
    return output;
  }
  else return output;
  
  for (; n > 2; n -= 3)
    op(output, input.next(), input.next(), input.next(), output);
  
  while (n-- > 0)
    op(output, input.next(), output);
  
  return output;
}

inline ImageBuffer Merge(MergeOp op, i32 n, InputAdaptor& input) {
  return Merge(op, n, input, ~input.peek());
}

template<typename Unary>
void Map(Unary op, i32 n, InputAdaptor& input) {
  for (i32 i = 0; i < n; i++) {
    ImageBuffer image = input.next();
    op(image, image);
  }
}

template<typename Unary>
void Map(Unary op, i32 n, InputAdaptor& input, OutputAdaptor& output) {
  for (i32 i = 0; i < n; i++)
    output.output(op(input.next()));
}

struct MaxOp : public MergeOp { MaxOp() : MergeOp("max") {} };
static MaxOp Max;

struct MinOp : public MergeOp { MinOp() : MergeOp("min") {} };
static MinOp Min;

struct AddOp : public MergeOp, public ScalarOp {
  AddOp() : MergeOp("sum"), ScalarOp("addscalar") {}
  using MergeOp::operator();
  using ScalarOp::operator();
};
static AddOp Add;

struct AddAbsOp : public MergeOp { AddAbsOp() : MergeOp("abssum") {} };
static AddAbsOp AddAbs;

struct SubOp : public BinaryOp { SubOp() : BinaryOp("subtract") {} };
static SubOp Sub;

struct MulOp : public BinaryOp, public ScalarOp {
  typedef ImageBuffer result_type;
  MulOp() : BinaryOp("mul"), ScalarOp("scale") {}
  using BinaryOp::operator();
  using ScalarOp::operator();
};
static MulOp Mul;

struct DivOp : public BinaryOp { DivOp() : BinaryOp("div") {} };
static DivOp Div;

struct NegOp : public UnaryOp { NegOp() : UnaryOp("negate") {} };
static NegOp Negate;

struct BoundOp : public UnaryOp { BoundOp() : UnaryOp("bound") {} };
static BoundOp Bound;

struct HalfRectifyOp : public UnaryOp { HalfRectifyOp() : UnaryOp("bound") {} };
static HalfRectifyOp HalfRectify;

struct ThresholdOp : public ScalarOp {
  ThresholdOp() : ScalarOp("threshold") {}
};
static ThresholdOp Threshold;

struct PointwiseThresholdOp : public BinaryOp {
  PointwiseThresholdOp() : BinaryOp("pointwisethreshold") {}
};
static PointwiseThresholdOp PointwiseThreshold;

struct MemsetOp : public BasicOp {
  typedef ImageBuffer result_type;
  
  MemsetOp() : BasicOp("writeval") {}
  
  inline ImageBuffer operator()(const ImageBuffer& i1, f32 val) {
    cl::Kernel &kernel = cache_.get();
    kernel.setArg(0, i1.mem());
    kernel.setArg(1, val);
    Enqueue(kernel, i1);
    return i1;
  }
};
static MemsetOp Memset;

struct MulAddOp : public BasicOp {
  typedef ImageBuffer result_type;
  
  MulAddOp() : BasicOp("muladd") {}
  
  inline ImageBuffer operator()(const ImageBuffer& i1,
                                const ImageBuffer& i2,
                                f32 scale,
                                ImageBuffer o) {
    cl::Kernel &kernel = cache_.get();
    kernel.setArg(0, i1.mem());
    kernel.setArg(1, i2.mem());
    kernel.setArg(2, scale);
    kernel.setArg(3, o.mem());
    Enqueue(kernel, o);
    return o;
  }
  
  inline ImageBuffer operator()(const ImageBuffer& i1,
                                const ImageBuffer& i2,
                                f32 scale) {
    return operator()(i1, i2, scale, ~i1);
  }
};
static MulAddOp MulAdd;

struct PowerOp : public BasicOp {
  typedef ImageBuffer result_type;
  
  PowerOp() : BasicOp("power") {}
  
  inline ImageBuffer operator()(const ImageBuffer& i1,
                                f32 amt, bool abs,
                                ImageBuffer o) {
    cl::Kernel &kernel = cache_.get();
    kernel.setArg(0, i1.mem());
    kernel.setArg(1, amt);
    kernel.setArg(2, abs);
    kernel.setArg(3, o.mem());
    Enqueue(kernel, o);
    return o;
  }
  
  inline ImageBuffer operator()(const ImageBuffer& i1, f32 amt, bool abs) {
    return operator()(i1, amt, abs, ~i1);
  }
};
static PowerOp Power;

struct ReductionOp {
  
};

inline f32 MaxReduce(const ImageBuffer& a) {
  return a.fetchData().data().max();
}

inline f32 SumReduce(const ImageBuffer& a) {
  return a.fetchData().data().sum();
}

inline ImageBuffer operator+=(ImageBuffer& a, const ImageBuffer& b) {
  return Add(a, b, a);
}

inline ImageBuffer operator+=(ImageBuffer& a, f32 amt) {
  return Add(a, amt, a);
}

inline ImageBuffer operator-=(ImageBuffer& a, const ImageBuffer& b) {
  return Sub(a, b, a);
}

inline ImageBuffer operator-=(ImageBuffer& a, f32 amt) {
  return Add(a, -amt, a);
}

inline ImageBuffer operator*=(ImageBuffer& a, const ImageBuffer& b) {
  return Mul(a, b, a);
}

inline ImageBuffer operator*=(ImageBuffer& a, f32 amt) {
  return Mul(a, amt, a);
}

inline ImageBuffer operator/=(ImageBuffer& a, const ImageBuffer& b) {
  return Div(a, b, a);
}

inline ImageBuffer operator/=(ImageBuffer& a, f32 amt) {
  return Mul(a, 1/amt, a);
}

inline ImageBuffer operator^=(ImageBuffer& a, f32 amt) {
  return Power(a, amt, false, a);
}

inline ImageBuffer operator+(const ImageBuffer& a, const ImageBuffer& b) {
  return Add(a, b);
}

inline ImageBuffer operator+(const ImageBuffer& a, f32 amt) {
  return Add(a, amt);
}

inline ImageBuffer operator-(const ImageBuffer& a, const ImageBuffer& b) {
  return Sub(a, b);
}

inline ImageBuffer operator-(const ImageBuffer& a, f32 amt) {
  return Add(a, -amt);
}

inline ImageBuffer operator*(const ImageBuffer& a, const ImageBuffer& b) {
  return Mul(a, b);
}

inline ImageBuffer operator*(const ImageBuffer& a, f32 amt) {
  return Mul(a, amt);
}

inline ImageBuffer operator/(const ImageBuffer& a, const ImageBuffer& b) {
  return Div(a, b);
}

inline ImageBuffer operator/(const ImageBuffer& a, f32 amt) {
  return Mul(a, 1/amt);
}

inline ImageBuffer operator^(const ImageBuffer& a, f32 amt) {
  return Power(a, amt, false);
}

}

#endif
