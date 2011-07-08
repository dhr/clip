CLIP_STRINGIFY(
kernel void writeval(global imval *buffer, float value) {
  int indx = get_global_index();
  store_imval(value, indx, buffer);
}

kernel void scale(global imval* input, float amount,
                  global imval* output) {
  int indx = get_global_index();
	store_imval(amount*load_imval(indx, input), indx, output);
}

kernel void power(global imval* input,
                  float amount,
                  int absify,
                  global imval* output) {
  int indx = get_global_index();
  float val = absify ? fabs(load_imval(indx, input)) : load_imval(indx, input);
	store_imval(pow(val, amount), indx, output);
}

kernel void negate(global imval* input,
                   global imval* output) {
  int indx = get_global_index();
	store_imval(-load_imval(indx, input), indx, output);
}

kernel void halfrect(global imval* input,
                     global imval* output) {
  int indx = get_global_index();
  store_imval(fmax(0.f, load_imval(indx, input)), indx, output);
}

kernel void threshold(global imval* input,
                      float thresh,
                      global imval* output) {
  int indx = get_global_index();
  float val = load_imval(indx, input);
  store_imval(val >= thresh ? val : 0.f, indx, output);
}

kernel void pointwisethreshold(global imval* input,
                               global imval* thresholds,
                               global imval* output) {
  int indx = get_global_index();
  float val = load_imval(indx, input);
  float thresh = load_imval(indx, thresholds);
  store_imval(val >= thresh ? val : 0.f, indx, output);
}

kernel void bound(global imval* input,
                  global imval* output) {
  int indx = get_global_index();
  store_imval(clamp(load_imval(indx, input), 0.f, 1.f), indx, output);
}

kernel void subtract(global imval* input1,
                     global imval* input2,
                     global imval* output) {
	int indx = get_global_index();
  store_imval(load_imval(indx, input1) -
              load_imval(indx, input2), indx, output);
}

kernel void addscalar(global imval* input,
                      float amt,
                      global imval* output) {
  int indx = get_global_index();
	store_imval(load_imval(indx, input) + amt, indx, output);
}

kernel void sum2(global imval* input1,
                 global imval* input2,
                 global imval* output) {
	int indx = get_global_index();
  store_imval(load_imval(indx, input1) +
              load_imval(indx, input2), indx, output);
}

kernel void sum4(global imval* input1,
                 global imval* input2,
                 global imval* input3,
                 global imval* input4,
                 global imval* output) {
	int indx = get_global_index();
  store_imval(load_imval(indx, input1) + load_imval(indx, input2) +
              load_imval(indx, input3) + load_imval(indx, input4),
              indx, output);
}

kernel void abssum2(global imval* input1,
                    global imval* input2,
                    global imval* output) {
	int indx = get_global_index();
  store_imval(fabs(load_imval(indx, input1)) + fabs(load_imval(indx, input2)),
              indx, output);
}

kernel void abssum4(global imval* input1,
                    global imval* input2,
                    global imval* input3,
                    global imval* input4,
                    global imval* output) {
	int indx = get_global_index();
  store_imval(fabs(load_imval(indx, input1)) + fabs(load_imval(indx, input2)) +
              fabs(load_imval(indx, input3)) + fabs(load_imval(indx, input4)),
              indx, output);
}

kernel void max2(global imval* input1,
                 global imval* input2,
                 global imval* output) {
	int indx = get_global_index();
  store_imval(fmax(load_imval(indx, input1), load_imval(indx, input2)),
              indx, output);
}

kernel void max4(global imval* input1,
                 global imval* input2,
                 global imval* input3,
                 global imval* input4,
                 global imval* output) {
	int indx = get_global_index();
  float result = fmax(load_imval(indx, input1), load_imval(indx, input2));
  result = fmax(result, load_imval(indx, input3));
  result = fmax(result, load_imval(indx, input4));
  store_imval(result, indx, output);
}

kernel void mul(global imval* input1,
                global imval* input2,
                global imval* output) {
	int indx = get_global_index();
  store_imval(load_imval(indx, input1)*load_imval(indx, input2), indx, output);
}

kernel void div(global imval* input1,
                global imval* input2,
                global imval* output) {
	int indx = get_global_index();
  store_imval(native_divide(load_imval(indx, input1),
                            load_imval(indx, input2)),
              indx, output);
}

kernel void muladd(global imval* input1,
                   global imval* input2,
                   float scale,
                   global imval* output) {
	int indx = get_global_index();
  store_imval(load_imval(indx, input1) + scale*load_imval(indx, input2),
              indx, output);
}
)
