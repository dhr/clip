#define CAT2(a, b) a##b
#define CAT3(a, b, c) a##b##c
#define CAT4(a, b, c, d) a##b##c##d
#define CAT5(a, b, c, d, e) a##b##c##d##e

#define ALLOW_EXP2(a, b) CAT2(a, b)
#define ALLOW_EXP3(a, b, c) CAT3(a, b, c)
#define ALLOW_EXP4(a, b, c, d) CAT4(a, b, c, d)
#define ALLOW_EXP5(a, b, c, d, e) CAT5(a, b, c, d, e)

#define vload_float(off, p) (*(p + off))
#define vload_float1 vload_float
#define vload_float2 vload2
#define vload_float4 vload4
#define vload_float8 vload8
#define vload_float16 vload16

#define vstore_float(val, off, p) (*(p + off) = (val))
#define vstore_float1 vstore_float
#define vstore_float2 vstore2
#define vstore_float4 vstore4
#define vstore_float8 vstore8
#define vstore_float16 vstore16

#define vload_half1 vload_half
#define vstore_half1 vstore_half

#ifndef CALC_WIDTH
# define CALC_WIDTH 4
#endif

#define calc_width CALC_WIDTH

#if calc_width != 1 && calc_width != 2 && calc_width != 4 && \
    calc_width != 8 && calc_width != 16
# error "Invalid calculation width (should be 1, 2, 4, 8, or 16)"
#endif

#define half1 half
#define float1 float
#define int1 int

#define calc_t ALLOW_EXP2(float, calc_width)
#define bool_t ALLOW_EXP2(int, calc_width)

#define load__(type, sz, off, p) \
  ALLOW_EXP3(vload_, type, sz)((off), (p))
#define store__(type, sz, val, off, p) \
  ALLOW_EXP3(vstore_, type, sz)((val), (off), (p))

#if defined(USE_TEXTURES)
# if calc_width != 4
#   error "Calculation width must be 4 when using textures"
# elif defined(MEMVAL_HALF)
#   error "Specifying the memory value type not allowed when using textures"
# else // Using floats as memory values and float4's as calculation type
#   define memval_t float
  
    const sampler_t default_sampler = CLK_NORMALIZED_COORDS_FALSE |
                                      CLK_ADDRESS_CLAMP |
                                      CLK_FILTER_NEAREST;
    
    typedef int2 index_t;
    
    index_t get_global_index(void);
    index_t get_global_index(void) {
      return (int2) (get_global_id(0), get_global_id(1));
    }

#   define input_t read_only image2d_t
#   define output_t write_only image2d_t

#   define load(off, p) read_imagef((p), default_sampler, (off))
#   define store(val, off, p) write_imagef((p), (off), (val))
# endif
#else // Using global memory
# if defined(MEMVAL_HALF)
#   define memval_t half
# else
#   define memval_t float
# endif

  typedef int index_t;
  
  index_t get_global_index(void);
  index_t get_global_index(void) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    int width = get_global_size(0);
    return y*width + x;
  }

# define input_t global memval_t*
# define output_t global memval_t*

# define loadm_(sz, off, p) load__(memval_t, sz, (off), (p))
# define storem_(sz, val, off, p) store__(memval_t, sz, (val), (off), (p))

# define loadm1(off, p) loadm_(1, (off), (p))
# define storem1(val, off, p) storem_(1, (val), (off), (p))

# define loadm4(off, p) loadm_(4, (off), (p))
# define storem4(val, off, p) storem_(4, (val), (off), (p))

# define load(off, p) loadm_(calc_width, (off), (p))
# define store(val, off, p) storem_(calc_width, (val), (off), (p))
#endif

#if calc_width == 1
# define iif(cond, a, b) ((cond) ? (a) : (b))
#else
# define iif(cond, a, b) (select((b), (a), (cond)))
#endif

#define PI 3.14159265358979323846f
