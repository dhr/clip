#ifndef CLIP_BUILTINKERNELS_H
#define CLIP_BUILTINKERNELS_H

#define CLIP_STRINGIFY(src) #src

inline const char* BasicKernels() {
  static const char* kernels =
    #include "kernels/util.clstr"
    #include "kernels/basic.clstr"
    ;
  return kernels;
}

inline const char* ImProcKernels() {
  static const char* kernels =
    #include "kernels/util.clstr"
    #include "kernels/improcutil.clstr"
    #include "kernels/improc.clstr"
    ;
  return kernels;
}

#endif