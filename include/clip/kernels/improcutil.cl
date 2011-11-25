#if defined(FILTVAL_HALF)
# define filtval_t half
# define sparse_filtval sparse_half
#else
# define filtval_t float
# define sparse_filtval sparse_float
#endif

#define vload_sparse_half(off, p) load_sparse_half_helper((p)[(off)])
#define vload_sparse_float(off, p) ((p)[(off)])

#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable

#define loadf_(sz, off, p) load__(filtval_t, sz, (off), (p))
#define storef_(sz, val, off, p) store__(filtval_t, sz, (val), (off), (p))

#define loads1(off, p) load__(sparse_filtval,, (off), (p))
#define stores1(val, off, p) store__(sparse_filtval,, (val), (off), (p))

#define loadf1(off, p) loadf_(1, (off), (p))
#define storef1(val, off, p) storef_(1, (val), (off), (p))

#define loadf4(off, p) loadf_(4, (off), (p))
#define storef4(val, off, p) storef_(4, (val), (off), (p))

typedef struct sparse_float {
  short x;
  short y;
  float val;
} sparse_float;

#if defined(FILTVAL_HALF)
typedef struct sparse_half {
  short x;
  short y;
  half val;
} sparse_half;

sparse_float load_sparse_half_helper(sparse_half sfv);
sparse_float load_sparse_half_helper(sparse_half sfv) {
  sparse_float sfl;
  sfl.x = sfv.x;
  sfl.y = sfv.y;
  sfl.val = vload_half(0, &sfv.val);
  
  return sfl;
}

kernel void
sparsefloat2half(global sparse_float* sv_float,
                 global sparse_half* sv_half) {
  index_t indx = get_global_index();
  sparse_float sf = sv_float[indx];

  sparse_half sh;
  sh.x = sf.x;
  sh.y = sf.y;
  vstore_half(sf.val, 0, &sh.val);
  
  sv_half[indx] = sh;
}

kernel void
sparsehalf2float(global sparse_half* sv_half,
                 global sparse_float* sv_float) {
  index_t indx = get_global_index();
  sparse_half sh = sv_half[indx];

  sparse_float sf;
  sf.x = sh.x;
  sf.y = sh.y;
  sf.val = vload_half(0, &sh.val);
  
  sv_float[indx] = sf;
}
#endif
