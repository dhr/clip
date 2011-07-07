#pragma once
#ifndef CLIP_CL_STATE_H
#define CLIP_CL_STATE_H

#include <iostream>
#include <map>
#include <stdexcept>
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
#include "clip/kernels.hpp"

namespace clip {

enum ComputeDeviceType {
  CPU = CL_DEVICE_TYPE_CPU,
  GPU = CL_DEVICE_TYPE_GPU,
  Accelerator = CL_DEVICE_TYPE_ACCELERATOR
};

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

inline cl::Device CurrentDevice() {
  return CurrentQueue().getInfo<CL_QUEUE_DEVICE>();
}

inline const ComputeDeviceType CurrentDeviceType() {
  cl::Device device = CurrentDevice();
  return ComputeDeviceType(device.getInfo<CL_DEVICE_TYPE>());
}

inline void AddProgram(const std::string& progName, cl::Program& program) {
  std::map<std::string, cl::Program>& programs = detail::programs();
  programs[progName] = program;

  std::string options("-cl-strict-aliasing -cl-fast-relaxed-math");
  program.build(detail::devices(), options.c_str());
  
  std::vector<cl::Kernel> newKernels;
  program.createKernels(&newKernels);
  
  std::map<std::string, cl::Kernel>& cache = detail::kernelCache();  
  std::vector<cl::Kernel>::iterator it, end;
  for (it = newKernels.begin(), end = newKernels.end(); it != end; ++it) {
    std::string name(it->getInfo<CL_KERNEL_FUNCTION_NAME>());
    cache[name] = *it;
  }
}

inline cl::Program AddProgram(const std::string& name, const std::string& src) {
  cl::Program::Sources sources(1, std::make_pair(src.c_str(), src.size()));
  cl::Program program(detail::context(), sources);
  AddProgram(name, program);
  return program;
}

inline cl::Program GetProgram(const std::string& name) {
  if (detail::programs().count(name) > 0)
    return detail::programs()[name];
  
  throw std::invalid_argument("Program \"" + name + "\" not found.");
}

inline bool HasKernel(const std::string& name) {
  return detail::kernelCache().count(name) > 0;
}
  
inline cl::Kernel& GetKernel(const std::string& name) {
  std::map<std::string, cl::Kernel>& cache = detail::kernelCache();
  if (HasKernel(name))
    return cache[name];
    
  throw std::invalid_argument("Kernel \"" + name + "\" not found.");
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

inline void ClipInit(cl::Context context, cl::CommandQueue queue) {
  detail::programs().clear();
  
  detail::context() = context;
  ++detail::contextID();
  
  detail::queue() = queue;
  
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
                     void (*errFn)(const char*, const void*, size_t, void*)
                       = CLIP_DEFAULT_NOTIFICATION_FUNCTION) {
  cl::Context context(devices, NULL, errFn);
  cl::CommandQueue queue = cl::CommandQueue(context, devices[0]);
  
  ClipInit(context, queue);
}

inline void ClipInit(cl::Device device,
                     void (*errFn)(const char*, const void*, size_t, void*)
                       = CLIP_DEFAULT_NOTIFICATION_FUNCTION) {
  std::vector<cl::Device> devices;
  devices.push_back(device);
  ClipInit(devices, errFn);
}

inline void ClipInit(ComputeDeviceType preferredType = GPU,
                     void (*errFn)(const char*, const void*, size_t, void*)
                       = CLIP_DEFAULT_NOTIFICATION_FUNCTION) {
  std::vector<cl::Platform> platforms;
  std::vector<cl::Device> devices;
  
	cl::Platform::get(&platforms);
  
  if (platforms.size() == 0)
    throw std::runtime_error("No OpenCL platforms found.");
  
  std::vector<cl::Platform>::iterator it, end;
  for (it = platforms.begin(), end = platforms.end(); it != end; ++it) {
    it->getDevices(preferredType, &devices);
    if (devices.size() != 0) break;
  }
  
  if (devices.size() == 0) {
    for (it = platforms.begin(), end = platforms.end(); it != end; ++it) {
      it->getDevices(CL_DEVICE_TYPE_DEFAULT, &devices);
      if (devices.size() != 0) break;
    }
    
    if (devices.size() == 0)
      throw std::runtime_error("Couldn't find any OpenCL devices.");
  }
  
  ClipInit(devices, errFn);
}

inline void Enqueue(cl::Kernel& k,
                    cl::NDRange offset,
                    cl::NDRange itemRange,
                    cl::NDRange groupRange) {
  static i32 counter = 0;
  CurrentQueue().enqueueNDRangeKernel(k, offset, itemRange, groupRange);
  if (++counter%5000 == 0) {
    CurrentQueue().finish();
  }
}

}

#endif
