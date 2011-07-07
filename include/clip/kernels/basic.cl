CLIP_STRINGIFY(
kernel void writeval(global float *buffer, float value) {
  int indx = get_global_index();
  buffer[indx] = value;
}

kernel void scale(global float *input,
                  float amount,
                  global float *output) {
  int indx = get_global_index();
	output[indx] = amount*input[indx];
}

kernel void power(global float *input,
                  float amount,
                  int absify,
                  global float *output) {
  int indx = get_global_index();
  float val = absify ? fabs(input[indx]) : input[indx];
	output[indx] = pow(val, amount);
}

kernel void negate(global float *input,
                   global float *output) {
  int indx = get_global_index();
	output[indx] = -input[indx];
}

kernel void halfrect(global float *input,
                     global float *output) {
  int indx = get_global_index();
  output[indx] = fmax(0.f, input[indx]);
}

kernel void threshold(global float *input,
                      float thresh,
                      global float *output) {
  int indx = get_global_index();
  float val = input[indx];
  output[indx] = val >= thresh ? val : 0.f;
}

kernel void pointwisethreshold(global float *input,
                               global float *thresholds,
                               global float *output) {
  int indx = get_global_index();
  float val = input[indx];
  float thresh = thresholds[indx];
  output[indx] = val >= thresh ? val : 0.f;
}

kernel void bound(global float *input,
                  global float *output) {
  int indx = get_global_index();
  output[indx] = clamp(input[indx], 0.f, 1.f);
}

kernel void subtract(global float *input1,
                     global float *input2,
                     global float *output) {
	int indx = get_global_index();
  output[indx] = input1[indx] - input2[indx];
}

kernel void addscalar(global float *input,
                      float amt,
                      global float *output) {
  int indx = get_global_index();
	output[indx] = input[indx] + amt;
}

kernel void sum2(global float *input1,
                 global float *input2,
                 global float *output) {
	int indx = get_global_index();
  output[indx] = input1[indx] + input2[indx];
}

kernel void sum4(global float *input1,
                 global float *input2,
                 global float *input3,
                 global float *input4,
                 global float *output) {
	int indx = get_global_index();
  output[indx] = input1[indx] + input2[indx] + input3[indx] + input4[indx];
}

kernel void abssum2(global float *input1,
                    global float *input2,
                    global float *output) {
	int indx = get_global_index();
  output[indx] = fabs(input1[indx]) + fabs(input2[indx]);
}

kernel void abssum4(global float *input1,
                    global float *input2,
                    global float *input3,
                    global float *input4,
                    global float *output) {
	int indx = get_global_index();
  output[indx] = fabs(input1[indx]) + fabs(input2[indx]) +
  fabs(input3[indx]) + fabs(input4[indx]);
}

kernel void max2(global float *input1,
                 global float *input2,
                 global float *output) {
	int indx = get_global_index();
  output[indx] = fmax(input1[indx], input2[indx]);
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
  output[indx] = result;
}

kernel void mul(global float *input1,
                global float *input2,
                global float *output) {
	int indx = get_global_index();
  output[indx] = input1[indx]*input2[indx];
}

kernel void div(global float *input1,
                global float *input2,
                global float *output) {
	int indx = get_global_index();
  output[indx] = input1[indx]/input2[indx];
}

kernel void muladd(global float *input1,
                   global float *input2,
                   float scale,
                   global float *output) {
	int indx = get_global_index();
  output[indx] = input1[indx] + scale*input2[indx];
}
)
