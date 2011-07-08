"\
#define VECTORIZE(nm, n) nm##n\n\
\n\
#if IMVAL_HALF\n\
#define load_imval(off, p) vload_half((off), (p))\n\
#define load_imval2(off, p) vload_half2((off), (p))\n\
#define load_imval4(off, p) vload_half4((off), (p))\n\
#define load_imval8(off, p) vload_half8((off), (p))\n\
#define load_imval16(off, p) vload_half16((off), (p))\n\
\n\
#define store_imval(val, off, p) vstore_half((val), (off), (p))\n\
#define store_imval2(val, off, p) vstore_half2((val), (off), (p))\n\
#define store_imval4(val, off, p) vstore_half4((val), (off), (p))\n\
#define store_imval8(val, off, p) vstore_half8((val), (off), (p))\n\
#define store_imval16(val, off, p) vstore_half16((val), (off), (p))\n\
#else\n\
#define load_imval(off, p) (((float) *(p + off))/IMVAL_UNIT)\n\
#define load_imval2(off, p) (((float2) vload2((off), (p)))/IMVAL_UNIT)\n\
#define load_imval4(off, p) (((float4) vload2((off), (p)))/IMVAL_UNIT)\n\
#define load_imval8(off, p) (((float8) vload2((off), (p)))/IMVAL_UNIT)\n\
#define load_imval16(off, p) (((float16) vload2((off), (p)))/IMVAL_UNIT)\n\
\n\
#define store_imval(val, off, p) \\\n\
  (*(p + off) = (imval) clamp((val)*IMVAL_UNIT, IMVAL_MIN, IMVAL_MAX))\n\
#define store_imval2(val, off, p) \\\n\
  vstore2((imval2) clamp((val)*IMVAL_UNIT, IMVAL_MIN, IMVAL_MAX), (off), (p))\n\
#define store_imval4(val, off, p) \\\n\
  vstore4((imval4) clamp((val)*IMVAL_UNIT, IMVAL_MIN, IMVAL_MAX), (off), (p))\n\
#define store_imval8(val, off, p) \\\n\
  vstore8((imval8) clamp((val)*IMVAL_UNIT, IMVAL_MIN, IMVAL_MAX), (off), (p))\n\
#define store_imval16(val, off, p) \\\n\
  vstore16((imval16) clamp((val)*IMVAL_UNIT, IMVAL_MIN, IMVAL_MAX), \\\n\
           (off), (p))\n\
#endif\n\
"
CLIP_STRINGIFY(
int get_global_index(void);

int get_global_index(void) {
  int x = get_global_id(0);
	int y = get_global_id(1);
  int width = get_global_size(0);
  return y*width + x;
}

typedef imval IMVAL_TYPE;
typedef imval2 VECTORIZE(IMVAL_TYPE, 2);
typedef imval4 VECTORIZE(IMVAL_TYPE, 4);
typedef imval8 VECTORIZE(IMVAL_TYPE, 8);
typedef imval16 VECTORIZE(IMVAL_TYPE, 16);
)
