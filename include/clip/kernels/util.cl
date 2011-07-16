"\
#define CAT2(a, b) a##b\n\
#define CAT3(a, b, c) a##b##c\n\
#define CAT4(a, b, c, d) a##b##c##d\n\
#define CAT5(a, b, c, d, e) a##b##c##d##e\n\
\n\
#define ALLOW_EXP2(a, b) CAT2(a, b)\n\
#define ALLOW_EXP3(a, b, c) CAT3(a, b, c)\n\
#define ALLOW_EXP4(a, b, c, d) CAT4(a, b, c, d)\n\
#define ALLOW_EXP5(a, b, c, d, e) CAT5(a, b, c, d, e)\n\
\n\
#define vload_float(off, p) (*(p + off))\n\
#define vload_float2 vload2\n\
#define vload_float4 vload4\n\
#define vload_float8 vload8\n\
#define vload_float16 vload16\n\
\n\
#define vstore_float(val, off, p) (*(p + off) = (val))\n\
#define vstore_float2 vstore2\n\
#define vstore_float4 vstore4\n\
#define vstore_float8 vstore8\n\
#define vstore_float16 vstore16\n\
\n\
#define imval2 ALLOW_EXP2(imval, 2)\n\
#define imval4 ALLOW_EXP2(imval, 4)\n\
#define imval8 ALLOW_EXP2(imval, 8)\n\
#define imval16 ALLOW_EXP2(imval, 16)\n\
\n\
#define load(type, off, p) CAT2(vload_, type)((off), (p))\n\
#define load2(type, off, p) CAT3(vload_, type, 2)((off), (p))\n\
#define load4(type, off, p) CAT3(vload_, type, 4)((off), (p))\n\
#define load8(type, off, p) CAT3(vload_, type, 8)((off), (p))\n\
#define load16(type, off, p) CAT3(vload_, type, 16)((off), (p))\n\
\n\
#define store(type, val, off, p) CAT2(vstore_, type)((val), (off), (p))\n\
#define store2(type, val, off, p) CAT3(vstore_, type, 2)((val), (off), (p))\n\
#define store4(type, val, off, p) CAT3(vstore_, type, 4)((val), (off), (p))\n\
#define store8(type, val, off, p) CAT3(vstore_, type, 8)((val), (off), (p))\n\
#define store16(type, val, off, p) CAT3(vstore_, type, 16)((val), (off), (p))\n\
\n\
#if IMVAL_HALF\n\
#define imval half\n\
#else\n\
#define imval float\n\
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

kernel void half2float(global half* input, global float* output) {
  int indx = get_global_index();
  output[indx] = vload_half(indx, input);
}

kernel void float2half(global float* input, global half* output) {
  int indx = get_global_index();
  vstore_half(input[indx], indx, output);
}
)
