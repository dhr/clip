#ifndef CLIP_BUILTINKERNELS_H
#define CLIP_BUILTINKERNELS_H

#define CLIP_STRINGIFY(src) #src

inline const char* BasicKernels() {
  static const char* kernels =
    #include "kernels/util.cl"
    #include "kernels/basic.cl"
    ;
  return kernels;
}

inline const char* ImProcKernels() {
  static const char* kernels =
    #include "kernels/util.cl"
    #include "kernels/improc.cl"
    ;
  return kernels;
}

#endif