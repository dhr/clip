#pragma once
#ifndef CLIP_CL_STATE_H
#define CLIP_CL_STATE_H

#include <iostream>
#include <map>
#include <stdexcept>
#include <sstream>
#include <vector>
#include <tr1/functional>

#define __CL_ENABLE_EXCEPTIONS

#ifdef __APPLE__
#include <unistd.h>
#include <OpenCL/cl_ext.h>
#endif

#pragma GCC system_header
#include <cl.hpp>

#include "clip/basictypes.hpp"
#include "clip/imagebuffertypes.hpp"
#include "clip/kernels.hpp"

namespace clip {

enum ComputeDeviceType {
  CPU = CL_DEVICE_TYPE_CPU,
  GPU = CL_DEVICE_TYPE_GPU,
  Accelerator = CL_DEVICE_TYPE_ACCELERATOR
};

enum ValueType {
  Float16,
  Float32
};

i32 SizeofValueType(ValueType type) {
  switch (type) {
    case Float16:
      return 2;
    
    case Float32:
      return 4;
    
    default:
      throw std::invalid_argument("Unrecognized value type");
  }
}

ValueType ValueTypeForBitDepth(i32 depth) {
  switch (depth) {
    case 16:
      return Float16;
    
    case 32:
      return Float32;
      
    default:
      throw std::invalid_argument("No corresponding value type for bit depth");
  }
}

typedef i32 ContextID;

namespace detail {

inline void notificationFunction(const char *errInfo,
                                 const void *, size_t, void *) {
	std::cerr << errInfo << std::endl;
}

inline std::vector< std::tr1::function<void ()> >& initClients() {
  static std::vector< std::tr1::function<void ()> > initClients;
  return initClients;
}

inline std::map<std::string, cl::Kernel>& kernelCache() {
  static std::map<std::string, cl::Kernel> cache;
  return cache;
};

inline std::vector<cl::Device>& devices() {
  static std::vector<cl::Device> devices;
  return devices;
}

inline cl::Context& context() {
  static cl::Context context;
  return context;
}

inline ContextID& contextID() {
  static ContextID contextID = 0;
  return contextID;
}

inline std::map<std::string, cl::Program>& programs() {
  static std::map<std::string, cl::Program> programs;
  return programs;
}

inline cl::CommandQueue& queue() {
  static cl::CommandQueue queue;
  return queue;
}

inline ValueType& imageBufferValueType() {
  static ValueType imageBufferValueType;
  return imageBufferValueType;
}

inline ValueType& filterValueType() {
  static ValueType imageBufferValueType;
  return imageBufferValueType;
}

inline i32& enqueuesPerFinish() {
  static i32 enqueuesPerFinish = 1;
  return enqueuesPerFinish;
}

}

inline const cl::Context& CurrentContext() {
  return detail::context();
}

inline const ContextID CurrentContextID() {
  return detail::contextID();
}

inline const cl::CommandQueue& CurrentQueue() {
  return detail::queue();
}

inline ValueType CurrentImBufValueType() {
  return detail::imageBufferValueType();
}

inline ValueType CurrentFilterValueType() {
  return detail::filterValueType();
}

inline const cl::Device& CurrentDevice() {
  return detail::devices()[0];
}

inline ComputeDeviceType CurrentDeviceType() {
  cl::Device device = CurrentDevice();
  return ComputeDeviceType(device.getInfo<CL_DEVICE_TYPE>());
}

inline void AddProgram(const std::string& progName, cl::Program& program,
                       const std::string& options = "") {
  std::map<std::string, cl::Program>& programs = detail::programs();
  if (programs.find(progName) != programs.end())
    throw std::invalid_argument("Program name already in use");
  programs[progName] = program;

  std::stringstream ss;
  
  switch (CurrentImBufValueType()) {
    case Float16:
      ss << "-D IMVAL_HALF ";
      break;
      
    case Float32:
      ss << "-D IMVAL_FLOAT ";
      break;
    
    default:
      throw std::logic_error("Unrecognized value type");
  }
  
  switch (CurrentFilterValueType()) {
    case Float16:
      ss << "-D FILTVAL_HALF ";
      break;
      
    case Float32:
      ss << "-D FILTVAL_FLOAT ";
      break;
    
    default:
      throw std::logic_error("Unrecognized value type");
  }
  
  ss << "-cl-strict-aliasing -cl-fast-relaxed-math ";
  
  try {
	  program.build(detail::devices(), (ss.str() + options).c_str());
  }
  catch (const cl::Error& err) {
  	const cl::Device& d = CurrentDevice();
  	std::string buildInfo = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(d);
  	std::cerr << buildInfo << std::endl;
  	throw err;
  }
  
  std::vector<cl::Kernel> newKernels;
  program.createKernels(&newKernels);
  
  std::map<std::string, cl::Kernel>& cache = detail::kernelCache();  
  std::vector<cl::Kernel>::iterator it, end;
  for (it = newKernels.begin(), end = newKernels.end(); it != end; ++it) {
    std::string name(it->getInfo<CL_KERNEL_FUNCTION_NAME>());
    cache[name] = *it;
  }
}

inline cl::Program AddProgram(const std::string& name, const std::string& src,
                              const std::string& options = "") {
  cl::Program::Sources sources(1, std::make_pair(src.c_str(), src.size()));
  cl::Program program(detail::context(), sources);
  AddProgram(name, program);
  return program;
}

inline cl::Program GetProgram(const std::string& name) {
  std::map<std::string, cl::Program>::iterator it;
  it = detail::programs().find(name);
  if (it != detail::programs().end())
    return it->second;
  
  throw std::invalid_argument("Program \"" + name + "\" not found");
}

inline bool HasKernel(const std::string& name) {
  return detail::kernelCache().count(name) > 0;
}
  
inline cl::Kernel& GetKernel(const std::string& name) {
  std::map<std::string, cl::Kernel>& cache = detail::kernelCache();
  if (HasKernel(name))
    return cache[name];
    
  throw std::invalid_argument("Kernel \"" + name + "\" not found");
}

class CachedKernel {
  std::string name_;
  cl::Kernel* cached_;
  ContextID contextID_;
  
 public:
  CachedKernel() {}
  
  CachedKernel(const std::string& name)
  : name_(name), cached_(NULL), contextID_() {}
  
  inline cl::Kernel& get() {
    if (contextID_ != CurrentContextID()) {
      cached_ = &GetKernel(name_);
      contextID_ = CurrentContextID();
    }
    return *cached_;
  }
};

inline void AddInitClient(std::tr1::function<void ()> init) {
  detail::initClients().push_back(init);
}

inline void ClipInit(cl::Context context, cl::CommandQueue queue,
                     ValueType imValType = Float32,
                     ValueType filtValType = Float32) {
  detail::programs().clear();
  
  detail::context() = context;
  ++detail::contextID();
  
  detail::queue() = queue;
  detail::imageBufferValueType() = imValType;
  detail::filterValueType() = filtValType;
  
  AddProgram("basic", BasicKernels());
  AddProgram("improc", ImProcKernels());
  
  std::vector< std::tr1::function<void ()> >& initClients =
    detail::initClients();
  std::vector< std::tr1::function<void ()> >::iterator it, end;
  for (it = initClients.begin(), end = initClients.end(); it != end; ++it)
    (*it)();
}

#ifdef __APPLE__
#define CLIP_DEFAULT_NOTIFICATION_FUNCTION clLogMessagesToStdoutAPPLE
#else
#define CLIP_DEFAULT_NOTIFICATION_FUNCTION detail::notificationFunction
#endif

inline void ClipInit(const std::vector<cl::Device>& devices,
                     ValueType imValType = Float32,
                     ValueType filtValType = Float32,
                     void (*errFn)(const char*, const void*, size_t, void*)
                       = CLIP_DEFAULT_NOTIFICATION_FUNCTION) {
  detail::devices() = devices;
  cl::Context context(detail::devices(), NULL, errFn);
  cl::CommandQueue queue = cl::CommandQueue(context, CurrentDevice());
  
  ClipInit(context, queue, imValType, filtValType);
}

inline void ClipInit(cl::Device device,
                     ValueType imValType = Float32,
                     ValueType filtValType = Float32,
                     void (*errFn)(const char*, const void*, size_t, void*)
                       = CLIP_DEFAULT_NOTIFICATION_FUNCTION) {
  std::vector<cl::Device> devices;
  devices.push_back(device);
  ClipInit(devices, imValType, filtValType, errFn);
}

inline void ClipInit(i32 platformNum, i32 deviceNum,
                     ValueType imValType = Float32,
                     ValueType filtValType = Float32,
                     void (*errFn)(const char*, const void*, size_t, void*)
                       = CLIP_DEFAULT_NOTIFICATION_FUNCTION) {
  std::vector<cl::Platform> platforms;
  cl::Platform::get(&platforms);
  
  if (platformNum < 0 || platformNum >= i32(platforms.size()))
    throw std::invalid_argument("Invalid platform number");
  
  std::vector<cl::Device> devices;
  platforms[platformNum].getDevices(CL_DEVICE_TYPE_ALL, &devices);
  
  if (deviceNum < 0 || deviceNum >= i32(devices.size()))
    throw std::invalid_argument("Invalid device number");
  
  ClipInit(devices[deviceNum], imValType, filtValType, errFn);
}

inline void ClipInit(ComputeDeviceType preferredType = GPU,
                     ValueType imValType = Float32,
                     ValueType filtValType = Float32,
                     void (*errFn)(const char*, const void*, size_t, void*)
                       = CLIP_DEFAULT_NOTIFICATION_FUNCTION) {
  std::vector<cl::Platform> platforms;
  std::vector<cl::Device> devices;
  
	cl::Platform::get(&platforms);
  
  if (platforms.size() == 0)
    throw std::runtime_error("No OpenCL platforms found");
  
  std::vector<cl::Platform>::iterator it, end;
  for (it = platforms.begin(), end = platforms.end(); it != end; ++it) {
    it->getDevices(CL_DEVICE_TYPE_ALL, &devices);
    std::vector<cl::Device>::iterator jt, jnd;
    for (jt = devices.begin(), jnd = devices.end(); jt != jnd; ++jt) {
      if (jt->getInfo<CL_DEVICE_TYPE>() == preferredType) {
        ClipInit(*jt, imValType, filtValType, errFn);
        return;
      }
    }
  }
  
  platforms[0].getDevices(CL_DEVICE_TYPE_DEFAULT, &devices);
  ClipInit(devices, imValType, filtValType, errFn);
}

inline i32 EnqueuesPerFinish() { return detail::enqueuesPerFinish(); }
inline void SetEnqueuesPerFinish(i32 n) { detail::enqueuesPerFinish() = n; }
  
inline void Enqueue(cl::Kernel& k,
                    cl::NDRange offset,
                    cl::NDRange itemRange,
                    cl::NDRange groupRange) {
  static i32 counter = 0;
  CurrentQueue().enqueueNDRangeKernel(k, offset, itemRange, groupRange);
  if (++counter%EnqueuesPerFinish() == 0) {
    CurrentQueue().finish();
  }
}

}

#endif
