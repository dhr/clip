"\n\
#define filtval2 ALLOW_EXP2(filtval, 2)\n\
#define filtval4 ALLOW_EXP2(filtval, 4)\n\
#define filtval8 ALLOW_EXP2(filtval, 8)\n\
#define filtval16 ALLOW_EXP2(filtval, 16)\n\
\n\
#if FILTVAL_HALF\n\
#define filtval half\n\
#define sparse_filtval sparse_half\n\
#else\n\
#define filtval float\n\
#define sparse_filtval sparse_float\n\
#endif\n\
\n\
#define vload_sparse_half(off, p) load_sparse_half_helper((p)[(off)])\n\
#define vload_sparse_float(off, p) ((p)[(off)])\n\
"
CLIP_STRINGIFY(
  typedef struct sparse_half {
    short x;
    short y;
    half val;
  } sparse_half;
  
  typedef struct sparse_float {
    short x;
    short y;
    float val;
  } sparse_float;
  
  sparse_float load_sparse_half_helper(sparse_half sfv);
  sparse_float load_sparse_half_helper(sparse_half sfv) {
    sparse_float sfl;
    sfl.x = sfv.x;
    sfl.y = sfv.y;
    sfl.val = load(filtval, 0, &sfv.val);
    
    return sfl;
  }
  
  kernel void
  sparsefloat2half(global sparse_float* sv_float,
                   global sparse_half* sv_half) {
    int indx = get_global_index();
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
    int indx = get_global_index();
    sparse_half sh = sv_half[indx];

    sparse_float sf;
    sf.x = sh.x;
    sf.y = sh.y;
    sf.val = vload_half(0, &sh.val);
    
    sv_float[indx] = sf;
  }
)
