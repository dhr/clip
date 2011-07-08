"\
#define CAT2(a, b) a##b\n\
#define CAT3(a, b, c) a##b##c\n\
#define CAT4(a, b, c, d) a##b##c##d\n\
#define CAT5(a, b, c, d, e) a##b##c##d##e\n\
#define ALLOW_EXP2(a, b) CAT2(a, b)\n\
#define ALLOW_EXP3(a, b, c) CAT3(a, b, c)\n\
#define ALLOW_EXP4(a, b, c, d) CAT4(a, b, c, d)\n\
#define ALLOW_EXP5(a, b, c, d, e) CAT5(a, b, c, d, e)\n\
\n\
#if IMVAL_HALF\n\
#define IMVAL_TYPE half\n\
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
#elif IMVAL_FLOAT\n\
#define IMVAL_TYPE float\n\
#define load_imval(off, p) (*(p + off))\n\
#define load_imval2(off, p) (vload2((off), (p)))\n\
#define load_imval4(off, p) (vload4((off), (p)))\n\
#define load_imval8(off, p) (vload8((off), (p)))\n\
#define load_imval16(off, p) (vload16((off), (p)))\n\
\n\
#define store_imval(val, off, p) (*(p + off) = (val))\n\
#define store_imval2(val, off, p) vstore2((val), (off), (p))\n\
#define store_imval4(val, off, p) vstore4((val), (off), (p))\n\
#define store_imval8(val, off, p) vstore8((val), (off), (p))\n\
#define store_imval16(val, off, p) vstore16((val), (off), (p))\n\
#else\n\
#define CVT_FLT(n, v) ALLOW_EXP2(convert_float, n)(v)\n\
#define CVT_IMV(n, v) \\\n\
  ALLOW_EXP4(convert_, IMVAL_TYPE, _sat, n)((v)*IMVAL_UNIT)\n\
\n\
#define load_imval(off, p) (CVT_FLT(, *(p + off))/IMVAL_UNIT)\n\
#define load_imval2(off, p) (CVT_FLT(2, vload2((off), (p)))/IMVAL_UNIT)\n\
#define load_imval4(off, p) (CVT_FLT(4, vload4((off), (p)))/IMVAL_UNIT)\n\
#define load_imval8(off, p) (CVT_FLT(8, vload8((off), (p)))/IMVAL_UNIT)\n\
#define load_imval16(off, p) (CVT_FLT(16, vload16((off), (p)))/IMVAL_UNIT)\n\
\n\
#define store_imval(val, off, p) (*(p + off) = CVT_IMV(, val))\n\
#define store_imval2(val, off, p) vstore2(CVT_IMV(2, val), (off), (p))\n\
#define store_imval4(val, off, p) vstore4(CVT_IMV(4, val), (off), (p))\n\
#define store_imval8(val, off, p) vstore8(CVT_IMV(8, val), (off), (p))\n\
#define store_imval16(val, off, p) vstore16(CVT_IMV(16, val), (off), (p))\n\
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

typedef IMVAL_TYPE imval;
typedef ALLOW_EXP2(IMVAL_TYPE, 2) imval2;
typedef ALLOW_EXP2(IMVAL_TYPE, 4) imval4;
typedef ALLOW_EXP2(IMVAL_TYPE, 8) imval8;
typedef ALLOW_EXP2(IMVAL_TYPE, 16) imval16;

kernel void imval2float(global imval* input, global float* output) {
  int indx = get_global_index();
  output[indx] = load_imval(indx, input);
}

kernel void float2imval(global float* input, global imval* output) {
  int indx = get_global_index();
  store_imval(input[indx], indx, output);
}
)
