CLIP_STRINGIFY(
kernel void writeval(global imval *buffer, float value) {
  int indx = get_global_index();
  store_imval(value, indx, buffer);
}

kernel void scale(global float *input,
                  float amount,
                  global float *output) {
  int indx = get_global_index();
	store_imval(amount*load_imval(indx, input), indx, output);
}

kernel void power(global float *input,
                  float amount,
                  int absify,
                  global float *output) {
  int indx = get_global_index();
  float val = absify ? fabs(load_imval(indx, input)) : load_imval(indx, input);
	store_imval(pow(val, amount), indx, output);
}

kernel void negate(global float *input,
                   global float *output) {
  int indx = get_global_index();
	store_imval(-load_imval(indx, input), indx, output);
}

kernel void halfrect(global float *input,
                     global float *output) {
  int indx = get_global_index();
  store_imval(fmax(0.f, load_imval(indx, input)), indx, output);
}

kernel void threshold(global float *input,
                      float thresh,
                      global float *output) {
  int indx = get_global_index();
  float val = load_imval(indx, input);
  store_imval(val >= thresh ? val : 0.f, indx, output);
}

kernel void pointwisethreshold(global float *input,
                               global float *thresholds,
                               global float *output) {
  int indx = get_global_index();
  float val = load_imval(indx, input);
  float thresh = thresholds[indx];
  store_imval(val >= thresh ? val : 0.f, indx, output);
}

kernel void bound(global float *input,
                  global float *output) {
  int indx = get_global_index();
  store_imval(clamp(load_imval(indx, input), 0.f, 1.f), indx, output);
}

kernel void subtract(global float *input1,
                     global float *input2,
                     global float *output) {
	int indx = get_global_index();
  store_imval(input1[indx] - input2[indx], indx, output);
}

kernel void addscalar(global float *input,
                      float amt,
                      global float *output) {
  int indx = get_global_index();
	store_imval(load_imval(indx, input) + amt, indx, output);
}

kernel void sum2(global float *input1,
                 global float *input2,
                 global float *output) {
	int indx = get_global_index();
  store_imval(input1[indx] + input2[indx], indx, output);
}

kernel void sum4(global float *input1,
                 global float *input2,
                 global float *input3,
                 global float *input4,
                 global float *output) {
	int indx = get_global_index();
  store_imval(input1[indx] + input2[indx] + input3[indx] + input4[indx], indx, output);
}

kernel void abssum2(global float *input1,
                    global float *input2,
                    global float *output) {
	int indx = get_global_index();
  store_imval(fabs(input1[indx]) + fabs(input2[indx]), indx, output);
}

kernel void abssum4(global float *input1,
                    global float *input2,
                    global float *input3,
                    global float *input4,
                    global float *output) {
	int indx = get_global_index();
  store_imval(fabs(input1[indx]) + fabs(input2[indx]) +, indx, output)
  fabs(input3[indx]) + fabs(input4[indx]);
}

kernel void max2(global float *input1,
                 global float *input2,
                 global float *output) {
	int indx = get_global_index();
  store_imval(fmax(input1[indx], input2[indx]), indx, output);
}

kernel void max4(global float *input1,
                 global float *input2,
                 global float *input3,
                 global float *input4,
                 global float *output) {
	int indx = get_global_index();
  float result = fmax(input1[indx], input2[indx]);
  result = fmax(result, input3[indx]);
  result = fmax(result, input4[indx]);
  store_imval(result, indx, output);
}

kernel void mul(global float *input1,
                global float *input2,
                global float *output) {
	int indx = get_global_index();
  store_imval(input1[indx]*input2[indx], indx, output);
}

kernel void div(global float *input1,
                global float *input2,
                global float *output) {
	int indx = get_global_index();
  store_imval(input1[indx]/input2[indx], indx, output);
}

kernel void muladd(global float *input1,
                   global float *input2,
                   float scale,
                   global float *output) {
	int indx = get_global_index();
  store_imval(input1[indx] + scale*input2[indx], indx, output);
}
)
