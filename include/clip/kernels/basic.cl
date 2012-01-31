kernel void writeval(output_t buffer, float value) {
  index_t indx = get_global_index();
  store((calc_t)(value), indx, buffer);
}

kernel void scale(input_t input, float amount,
                  output_t output) {
  index_t indx = get_global_index();
	store(amount*load(indx, input), indx, output);
}

kernel void power(input_t input,
                  float amount,
                  int absify,
                  output_t output) {
  index_t indx = get_global_index();
  calc_t val = load(indx, input);
  val = absify ? fabs(val) : val;
	store(pow(val, (calc_t)(amount)), indx, output);
}

kernel void negate(input_t input,
                   output_t output) {
  index_t indx = get_global_index();
	store(-load(indx, input), indx, output);
}

kernel void absval(input_t input,
                   output_t output) {
  index_t indx = get_global_index();
	store(fabs(load(indx, input)), indx, output);
}

kernel void squareroot(input_t input,
                       output_t output) {
  index_t indx = get_global_index();
  store(native_sqrt(load(indx, input)), indx, output);
}

kernel void halfrect(input_t input,
                     output_t output) {
  index_t indx = get_global_index();
  store(fmax((calc_t) 0.f, load(indx, input)), indx, output);
}

kernel void threshold(input_t input,
                      float thresh,
                      output_t output) {
  index_t indx = get_global_index();
  calc_t val = load(indx, input);
  store(iif(val >= thresh, val, 0.f), indx, output);
}

kernel void pointwisethreshold(input_t input,
                               input_t thresholds,
                               output_t output) {
  index_t indx = get_global_index();
  calc_t val = load(indx, input);
  calc_t thresh = load(indx, thresholds);
  store(iif(val >= thresh, val, 0.f), indx, output);
}

kernel void bound(input_t input,
                  output_t output) {
  index_t indx = get_global_index();
  store(clamp(load(indx, input), 0.f, 1.f), indx, output);
}

kernel void subtract(input_t input1,
                     input_t input2,
                     output_t output) {
	index_t indx = get_global_index();
  store(load(indx, input1) - load(indx, input2), indx, output);
}

kernel void addscalar(input_t input,
                      float amt,
                      output_t output) {
  index_t indx = get_global_index();
	store(load(indx, input) + amt, indx, output);
}

kernel void sum2(input_t input1,
                 input_t input2,
                 output_t output) {
	index_t indx = get_global_index();
  store(load(indx, input1) + load(indx, input2), indx, output);
}

kernel void sum4(input_t input1,
                 input_t input2,
                 input_t input3,
                 input_t input4,
                 output_t output) {
	index_t indx = get_global_index();
  store(load(indx, input1) + load(indx, input2) +
        load(indx, input3) + load(indx, input4),
        indx, output);
}

kernel void abssum2(input_t input1,
                    input_t input2,
                    output_t output) {
	index_t indx = get_global_index();
  store(fabs(load(indx, input1)) + fabs(load(indx, input2)), indx, output);
}

kernel void abssum4(input_t input1,
                    input_t input2,
                    input_t input3,
                    input_t input4,
                    output_t output) {
	index_t indx = get_global_index();
  store(fabs(load(indx, input1)) + fabs(load(indx, input2)) +
        fabs(load(indx, input3)) + fabs(load(indx, input4)),
        indx, output);
}

kernel void max2(input_t input1,
                 input_t input2,
                 output_t output) {
	index_t indx = get_global_index();
  store(fmax(load(indx, input1), load(indx, input2)), indx, output);
}

kernel void max4(input_t input1,
                 input_t input2,
                 input_t input3,
                 input_t input4,
                 output_t output) {
	index_t indx = get_global_index();
  calc_t result = fmax(load(indx, input1), load(indx, input2));
  result = fmax(result, load(indx, input3));
  result = fmax(result, load(indx, input4));
  store(result, indx, output);
}

kernel void mul(input_t input1,
                input_t input2,
                output_t output) {
	index_t indx = get_global_index();
  store(load(indx, input1)*load(indx, input2), indx, output);
}

kernel void div(input_t input1,
                input_t input2,
                output_t output) {
	index_t indx = get_global_index();
  store(native_divide(load(indx, input1), load(indx, input2)), indx, output);
}

kernel void muladd(input_t input1,
                   input_t input2,
                   float scale,
                   output_t output) {
	index_t indx = get_global_index();
  store(load(indx, input1) + scale*load(indx, input2), indx, output);
}
