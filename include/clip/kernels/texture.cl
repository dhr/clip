int2 get_global_index(void);
float4 contpart(float4 x, float4 degree);
float4 smoothpart(float4 x, float4 degree);

const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |
                          CLK_ADDRESS_CLAMP |
                          CLK_FILTER_NEAREST;

const sampler_t fsampler = CLK_NORMALIZED_COORDS_FALSE |
                           CLK_ADDRESS_CLAMP |
                           CLK_FILTER_NEAREST;

#if 1
kernel void filter(read_only image2d_t image,
                   //read_only image2d_t filter,
                   constant float4 *filter,
                   int filt_width, int filt_height,
                   write_only image2d_t output) {
  int2 gid = (int2) (get_global_id(0), get_global_id(1));
  int2 fc;
  
  filt_width /= 4;
  
  int half_filt_width = filt_width/2;
  int half_filt_height = filt_height/2;
  int filt_x_lim = filt_width - half_filt_width;
  int filt_y_lim = filt_height - half_filt_height;
  
  int2 foff = (int2) (half_filt_width, half_filt_height);
  int2 xoff = (int2) (1, 0);
  
  float4 sum = 0.f;
  if (filt_width%2 == 0) {
    float4 imval1, imval2, fval;
    
    fc.x = -half_filt_width - 1;
    fc.y = -half_filt_height;
    imval2 = read_imagef(image, sampler, gid + fc);
    while (true) {
      if (++fc.x >= filt_x_lim) {
        if (++fc.y >= filt_y_lim) break;
        fc.x = -half_filt_width;
        imval2 = read_imagef(image, sampler, gid + fc);
      }
      
      //fval = read_imagef(filter, fsampler, fc + foff);
      fval = filter[(fc.y + foff.y)*filt_width + fc.x + foff.x];
      
      imval1 = imval2;
      imval2 = read_imagef(image, sampler, gid + fc + xoff);
      
      sum.x += fval.x*imval1.x;
      sum.x += fval.y*imval1.y;
      sum.x += fval.z*imval1.z;
      sum.x += fval.w*imval1.w;
      
      sum.y += fval.x*imval1.y;
      sum.y += fval.y*imval1.z;
      sum.y += fval.z*imval1.w;
      sum.y += fval.w*imval2.x;
      
      sum.z += fval.x*imval1.z;
      sum.z += fval.y*imval1.w;
      sum.z += fval.z*imval2.x;
      sum.z += fval.w*imval2.y;
      
      sum.w += fval.x*imval1.w;
      sum.w += fval.y*imval2.x;
      sum.w += fval.z*imval2.y;
      sum.w += fval.w*imval2.z;
    }
  }
  else {
    float4 imval1, imval2, imval3, fval;
    
    fc.x = -half_filt_width - 1;
    fc.y = -half_filt_height;
    imval2 = read_imagef(image, sampler, gid + fc);
    imval3 = read_imagef(image, sampler, gid + fc + xoff);
    while (true) {
      if (++fc.x >= filt_x_lim) {
        if (++fc.y >= filt_y_lim) break;
        fc.x = -half_filt_width;
        imval2 = read_imagef(image, sampler, gid + fc - xoff);
        imval3 = read_imagef(image, sampler, gid + fc);
      }
      
      //fval = read_imagef(filter, fsampler, fc + foff);
      fval = filter[(fc.y + foff.y)*filt_width + fc.x + foff.x];
      
      imval1 = imval2;
      imval2 = imval3;
      imval3 = read_imagef(image, sampler, gid + fc + xoff);
      
      sum.x += fval.x*imval1.z;
      sum.x += fval.y*imval1.w;
      sum.x += fval.z*imval2.x;
      sum.x += fval.w*imval2.y;
      
      sum.y += fval.x*imval1.w;
      sum.y += fval.y*imval2.x;
      sum.y += fval.z*imval2.y;
      sum.y += fval.w*imval2.z;
      
      sum.z += fval.x*imval2.x;
      sum.z += fval.y*imval2.y;
      sum.z += fval.z*imval2.z;
      sum.z += fval.w*imval2.w;
      
      sum.w += fval.x*imval2.y;
      sum.w += fval.y*imval2.z;
      sum.w += fval.z*imval2.w;
      sum.w += fval.w*imval3.x;
    }
  }
  
  write_imagef(output, gid, sum);
}
#elif 1
kernel void filter(read_only image2d_t image,
                   //read_only image2d_t filter,
                   constant float4 *filter,
                   int filt_width, int filt_height,
                   write_only image2d_t output) {
  int2 gid = (int2) (get_global_id(0), get_global_id(1));
  int2 fc;
  
  filt_width /= 4;
  
  int half_filt_width = filt_width/2;
  int half_filt_height = filt_height/2;
  int filt_y_lim = filt_height - half_filt_height;
  
  int2 foff = (int2) (half_filt_width, half_filt_height);
  int2 xoff = (int2) (1, 0);
  
  float4 sum = 0.f;
  if (filt_width%2 == 0) {
    float4 imval1, imval2, fval;
    
    for (fc.y = -half_filt_height; fc.y < filt_y_lim; fc.y++) {
      fc.x = -half_filt_width;
      
      imval2 = read_imagef(image, sampler, gid + fc);
      for (; fc.x < half_filt_width; fc.x++) {
        //fval = read_imagef(filter, fsampler, fc + foff);
        fval = filter[(fc.y + foff.y)*filt_width + fc.x + foff.x];
        
        imval1 = imval2;
        imval2 = read_imagef(image, sampler, gid + fc + xoff);
        
        sum.x += fval.x*imval1.x;
        sum.x += fval.y*imval1.y;
        sum.x += fval.z*imval1.z;
        sum.x += fval.w*imval1.w;
        
        sum.y += fval.x*imval1.y;
        sum.y += fval.y*imval1.z;
        sum.y += fval.z*imval1.w;
        sum.y += fval.w*imval2.x;
        
        sum.z += fval.x*imval1.z;
        sum.z += fval.y*imval1.w;
        sum.z += fval.z*imval2.x;
        sum.z += fval.w*imval2.y;
        
        sum.w += fval.x*imval1.w;
        sum.w += fval.y*imval2.x;
        sum.w += fval.z*imval2.y;
        sum.w += fval.w*imval2.z;
      }
    }
  }
  else {
    float4 imval1, imval2, imval3, fval;
    
    for (fc.y = -half_filt_height; fc.y < filt_y_lim; fc.y++) {
      fc.x = -half_filt_width;
      
      imval2 = read_imagef(image, sampler, gid + fc - xoff);
      imval3 = read_imagef(image, sampler, gid + fc);
      for (; fc.x < half_filt_width; fc.x++) {
        //fval = read_imagef(filter, fsampler, fc + foff);
        fval = filter[(fc.y + foff.y)*filt_width + fc.x + foff.x];
        
        imval1 = imval2;
        imval2 = imval3;
        imval3 = read_imagef(image, sampler, gid + fc + xoff);
        
        sum.x += fval.x*imval1.z;
        sum.x += fval.y*imval1.w;
        sum.x += fval.z*imval2.x;
        sum.x += fval.w*imval2.y;
        
        sum.y += fval.x*imval1.w;
        sum.y += fval.y*imval2.x;
        sum.y += fval.z*imval2.y;
        sum.y += fval.w*imval2.z;
        
        sum.z += fval.x*imval2.x;
        sum.z += fval.y*imval2.y;
        sum.z += fval.z*imval2.z;
        sum.z += fval.w*imval2.w;
        
        sum.w += fval.x*imval2.y;
        sum.w += fval.y*imval2.z;
        sum.w += fval.z*imval2.w;
        sum.w += fval.w*imval3.x;
      }
    }
  }
  
  write_imagef(output, gid, sum);
}
#elif 0
kernel void filter(read_only image2d_t image,
                   local float *im_cache,
                   constant float *filter,
                   int filt_width, int filt_height,
                   constant short2 *coords,
                   int num_elems,
                   write_only image2d_t output) {
  int2 gid = (int2) (get_global_id(0), get_global_id(1));
  int2 fc;
  
  int filt_indx = 0;
  int filt_lim = 1;
  
  int half_filt_width = filt_width/2;
  int half_filt_height = filt_height/2;
  int filt_y_lim = filt_height - half_filt_height;
  
  int apron_rem = half_filt_width%4;
  int apron_padding = apron_rem ? 4 - apron_rem : 0;
  int apron_width = half_filt_width + apron_padding;
  
  int2 local_base = (int2) (apron_width, half_filt_height);
  int2 local_coord = local_base + lid;
  int im_cache_width = 4*(group_size.x + 2*(apron_width));
  int local_indx = local_coord.y*im_cache_width + 4*local_coord.x;
  
  vstore4(read_imagef(image, sampler, gid), 0, im_cache + local_indx);
  
  int x_apron_off = lid.x < local_base.x ? -local_base.x : 0;
  x_apron_off = lid.x >= group_size.x - local_base.x ?
  local_base.x : x_apron_off;
  
  int y_apron_off = lid.y < local_base.y ? -local_base.y : 0;
  y_apron_off = (lid.y >= group_size.y - local_base.y ?
                 local_base.y : y_apron_off);
  int y_apron_indx_off = y_apron_off*im_cache_width;
  
  if (x_apron_off) {
    vstore4(read_imagef(image, sampler,
                        (int2) (gid.x + x_apron_off, gid.y)),
            x_apron_off, im_cache + local_indx);
  }
  
  if (y_apron_off) {
    vstore4(read_imagef(image, sampler,
                        (int2) (gid.x, gid.y + y_apron_off)),
            0, im_cache + local_indx + y_apron_indx_off);
  }
  
  if (x_apron_off && y_apron_off) {
    vstore4(read_imagef(image, sampler,
                        (int2) (gid.x + x_apron_off, gid.y + y_apron_off)),
            x_apron_off, im_cache + local_indx + y_apron_indx_off);
  }
  
  barrier(CLK_LOCAL_MEM_FENCE);
  
  int2 foff = (int2) (half_filt_width, half_filt_height);
  int2 xoff = (int2) (1, 0);
  
  float4 sum = 0.f;
  float4 imval1, imval2, fval;
  
  for (fc.y = -half_filt_height; fc.y < filt_y_lim; fc.y++) {
    fc.x = -half_filt_width;
    
    imval2 = read_imagef(image, sampler, gid + fc);
    for (; fc.x < half_filt_width; fc.x++) {
      //fval = read_imagef(filter, fsampler, fc + foff);
      fval = filter[(fc.y + foff.y)*filt_width + fc.x + foff.x];
      
      imval1 = imval2;
      imval2 = read_imagef(image, sampler, gid + fc + xoff);
      
      sum.x += fval.x*imval1.x;
      sum.x += fval.y*imval1.y;
      sum.x += fval.z*imval1.z;
      sum.x += fval.w*imval1.w;
      
      sum.y += fval.x*imval1.y;
      sum.y += fval.y*imval1.z;
      sum.y += fval.z*imval1.w;
      sum.y += fval.w*imval2.x;
      
      sum.z += fval.x*imval1.z;
      sum.z += fval.y*imval1.w;
      sum.z += fval.z*imval2.x;
      sum.z += fval.w*imval2.y;
      
      sum.w += fval.x*imval1.w;
      sum.w += fval.y*imval2.x;
      sum.w += fval.z*imval2.y;
      sum.w += fval.w*imval2.z;
    }
  }
  
  write_imagef(output, gid, sum);
}
#else
kernel void filter(read_only image2d_t image,
                   local float *im_cache,
                   constant float4 *filter,
                   int filt_width, int filt_height,
                   write_only image2d_t output) {
  int2 gid = (int2) (get_global_id(0), get_global_id(1));
  int2 lid = (int2) (get_local_id(0), get_local_id(1));
  int2 group_size = (int2) (get_local_size(0), get_local_size(1));
  
  int2 filt_coord;
  int filt_tex_width = filt_width/4;
  int half_filt_tex_width = filt_tex_width/2;
  int half_filt_height = filt_height/2;
  
  int odd = filt_tex_width%2;
  
  // Lower-left base of the central (non-apron) image tile in shared memory
  int2 local_base = (int2) (half_filt_tex_width, half_filt_height);
  int2 local_coord = local_base + lid;
  int im_cache_width = 4*(group_size.x + 2*(half_filt_tex_width));
  int local_indx = local_coord.y*im_cache_width + 4*local_coord.x;
  
  vstore4(read_imagef(image, sampler, gid), 0, im_cache + local_indx);
  
  int x_apron_off = lid.x < local_base.x ? -local_base.x : 0;
  x_apron_off = lid.x >= group_size.x - local_base.x ?
  local_base.x : x_apron_off;
  
  int y_apron_off = lid.y < local_base.y ? -local_base.y : 0;
  y_apron_off = (lid.y >= group_size.y - local_base.y ?
                 local_base.y : y_apron_off);
  int y_apron_indx_off = y_apron_off*im_cache_width;
  
  if (x_apron_off) {
    vstore4(read_imagef(image, sampler,
                        (int2) (gid.x + x_apron_off, gid.y)),
            x_apron_off, im_cache + local_indx);
  }
  
  if (y_apron_off) {
    vstore4(read_imagef(image, sampler,
                        (int2) (gid.x, gid.y + y_apron_off)),
            0, im_cache + local_indx + y_apron_indx_off);
  }
  
  if (x_apron_off && y_apron_off) {
    vstore4(read_imagef(image, sampler,
                        (int2) (gid.x + x_apron_off, gid.y + y_apron_off)),
            x_apron_off, im_cache + local_indx + y_apron_indx_off);
  }
  
  barrier(CLK_LOCAL_MEM_FENCE);
  
  float4 sum = 0.f;
  
  float4 fval, imval1, imval2;
  for (int y = 0; y < filt_height; y++) {
    int filt_row_base = y*filt_tex_width;
    int im_cache_row_base = (lid.y + y)*im_cache_width + 4*lid.x;
    
    imval2 = vload4(0, im_cache + im_cache_row_base);
    for (int x = 0; x < filt_tex_width; x++) {
      fval = filter[filt_row_base + x];
      
      imval1 = imval2;
      imval2 = vload4(x + 1, im_cache + im_cache_row_base);
      
      sum.x += fval.x*imval1.x;
      sum.x += fval.y*imval1.y;
      sum.x += fval.z*imval1.z;
      sum.x += fval.w*imval1.w;
      
      sum.y += fval.x*imval1.y;
      sum.y += fval.y*imval1.z;
      sum.y += fval.z*imval1.w;
      sum.y += fval.w*imval2.x;
      
      sum.z += fval.x*imval1.z;
      sum.z += fval.y*imval1.w;
      sum.z += fval.z*imval2.x;
      sum.z += fval.w*imval2.y;
      
      sum.w += fval.x*imval1.w;
      sum.w += fval.y*imval2.x;
      sum.w += fval.z*imval2.y;
      sum.w += fval.w*imval2.z;
    }
  }
  
  write_imagef(output, gid, sum);
}
#endif

int2 get_global_index(void) {
  return (int2) (get_global_id(0), get_global_id(1));
}

kernel void scale(read_only image2d_t input,
                  float amount,
                  write_only image2d_t output) {
  int2 gid = get_global_index();
	write_imagef(output, gid, amount*read_imagef(input, sampler, gid));
}

kernel void power(read_only image2d_t input,
                  float amount,
                  int absify,
                  write_only image2d_t output) {
  int2 gid = get_global_index();
  float4 val = read_imagef(input, sampler, gid);
  val = absify ? fabs(val) : val;
  float4 amount4 = amount;
	write_imagef(output, gid, pow(val, amount4));
}

kernel void negate(read_only image2d_t input,
                   write_only image2d_t output) {
  int2 gid = get_global_index();
	write_imagef(output, gid, -read_imagef(input, sampler, gid));
}

kernel void halfrect(read_only image2d_t input,
                     write_only image2d_t output) {
  int2 gid = get_global_index();
  write_imagef(output, gid, fmax((float4) 0,
                                 read_imagef(input, sampler, gid)));
}

kernel void threshold(read_only image2d_t input,
                      float thresh,
                      write_only image2d_t output) {
  int2 gid = get_global_index();
  float4 val = read_imagef(input, sampler, gid);
  write_imagef(output, gid, select(0.f, val, val >= (float4) thresh));
}

kernel void pointwisethreshold(read_only image2d_t input,
                               read_only image2d_t thresholds,
                               write_only image2d_t output) {
  int2 gid = get_global_index();
  float4 val = read_imagef(input, sampler, gid);
  float4 thresh = read_imagef(thresholds, sampler, gid);
  write_imagef(output, gid, select(0.f, val, val >= thresh));
}

kernel void bound(read_only image2d_t input,
                  write_only image2d_t output) {
  int2 gid = get_global_index();
  write_imagef(output, gid, clamp(read_imagef(input, sampler, gid),
                                  (float4) 0, (float4) 1));
}

kernel void subtract(read_only image2d_t input1,
                     read_only image2d_t input2,
                     write_only image2d_t output) {
	int2 gid = get_global_index();
  float4 result = read_imagef(input1, sampler, gid) -
  read_imagef(input2, sampler, gid);
  write_imagef(output, gid, result);
}

kernel void sum2(read_only image2d_t input1,
                 read_only image2d_t input2,
                 write_only image2d_t output) {
	int2 gid = get_global_index();
  float4 result = read_imagef(input1, sampler, gid) +
  read_imagef(input2, sampler, gid);
  write_imagef(output, gid, result);
}

kernel void sum4(read_only image2d_t input1,
                 read_only image2d_t input2,
                 read_only image2d_t input3,
                 read_only image2d_t input4,
                 write_only image2d_t output) {
	int2 gid = get_global_index();
  float4 result = read_imagef(input1, sampler, gid) +
  read_imagef(input2, sampler, gid) +
  read_imagef(input3, sampler, gid) +
  read_imagef(input4, sampler, gid);
  write_imagef(output, gid, result);
}

kernel void abssum2(read_only image2d_t input1,
                    read_only image2d_t input2,
                    write_only image2d_t output) {
	int2 gid = get_global_index();
  float4 result = fabs(read_imagef(input1, sampler, gid)) +
  fabs(read_imagef(input2, sampler, gid));
  write_imagef(output, gid, result);
}

kernel void abssum4(read_only image2d_t input1,
                    read_only image2d_t input2,
                    read_only image2d_t input3,
                    read_only image2d_t input4,
                    write_only image2d_t output) {
	int2 gid = get_global_index();
  float4 result = fabs(read_imagef(input1, sampler, gid)) +
  fabs(read_imagef(input2, sampler, gid)) +
  fabs(read_imagef(input3, sampler, gid)) +
  fabs(read_imagef(input4, sampler, gid));
  write_imagef(output, gid, result);
}

kernel void max2(read_only image2d_t input1,
                 read_only image2d_t input2,
                 write_only image2d_t output) {
	int2 gid = get_global_index();
  float4 val1 = read_imagef(input1, sampler, gid);
  float4 val2 = read_imagef(input2, sampler, gid);
  float4 result = fmax(val1, val2);
  write_imagef(output, gid, result);
}

kernel void max4(read_only image2d_t input1,
                 read_only image2d_t input2,
                 read_only image2d_t input3,
                 read_only image2d_t input4,
                 write_only image2d_t output) {
	int2 gid = get_global_index();
  float4 val1 = read_imagef(input1, sampler, gid);
  float4 val2 = read_imagef(input2, sampler, gid);
  float4 val3 = read_imagef(input3, sampler, gid);
  float4 val4 = read_imagef(input4, sampler, gid);
  float4 result = fmax(val1, val2);
  result = fmax(result, val3);
  result = fmax(result, val4);
  write_imagef(output, gid, result);
}

kernel void mul(read_only image2d_t input1,
                read_only image2d_t input2,
                write_only image2d_t output) {
	int2 gid = get_global_index();
  float4 result = read_imagef(input1, sampler, gid)*
  read_imagef(input2, sampler, gid);
  write_imagef(output, gid, result);
}

kernel void div(read_only image2d_t input1,
                read_only image2d_t input2,
                write_only image2d_t output) {
	int2 gid = get_global_index();
  float4 result = read_imagef(input1, sampler, gid)/
  read_imagef(input2, sampler, gid);
  write_imagef(output, gid, result);
}

kernel void muladd(read_only image2d_t input1,
                   read_only image2d_t input2,
                   float scale,
                   write_only image2d_t output) {
	int2 gid = get_global_index();
  float4 result = read_imagef(input1, sampler, gid) +
  scale*read_imagef(input2, sampler, gid);
  write_imagef(output, gid, result);
}

kernel void grad2polar(read_only image2d_t xs,
                       read_only image2d_t ys,
                       write_only image2d_t mag,
                       write_only image2d_t angle) {
	int2 gid = get_global_index();
  float4 x = read_imagef(xs, sampler, gid);
  float4 y = read_imagef(ys, sampler, gid);
  write_imagef(angle, gid, atan2(x, -y));
  write_imagef(mag, gid, hypot(x, y));
}

kernel void unitvec(read_only image2d_t angles,
                    write_only image2d_t vs,
                    write_only image2d_t us) {
	int2 gid = get_global_index();
  float4 angle = read_imagef(angles, sampler, gid);
  write_imagef(us, gid, cos(angle));
  write_imagef(vs, gid, sin(angle));
}

kernel void ktkn(read_only image2d_t us,
                 read_only image2d_t vs,
                 read_only image2d_t vxs,
                 read_only image2d_t vys,
                 write_only image2d_t kns,
                 write_only image2d_t kts) {
	int2 gid = get_global_index();
  float4 u = read_imagef(us, sampler, gid);
  float4 v = read_imagef(vs, sampler, gid);
  float4 vx = read_imagef(vxs, sampler, gid);
  float4 vy = read_imagef(vys, sampler, gid);
  write_imagef(kts, gid, (v*vy + u*vx)/u);
  write_imagef(kns, gid, (u*vy - v*vx)/u);
}

kernel void rescale(read_only image2d_t input,
                    float min, float max, float targ_min, float targ_max,
                    int filter,
                    write_only image2d_t output) {
	int2 gid = get_global_index();
  float4 inval = read_imagef(input, sampler, gid);
  float4 outval = targ_min + (inval - min)/(max - min)*(targ_max - targ_min);
  if (filter) outval = select(0.f, outval, inval > min && inval < max);
  write_imagef(output, gid, outval);
}

#define PI 3.14159265358979323846f

kernel void flowdiscr(read_only image2d_t confs,
                      read_only image2d_t thetas,
                      float targ_theta, float theta_step, int npis,
                      read_only image2d_t kts,
                      read_only image2d_t kns,
                      float targ_kt, float targ_kn, float k_step,
                      write_only image2d_t output) {
  int2 gid = get_global_index();
  
  float4 conf = read_imagef(confs, sampler, gid);
  float4 theta = read_imagef(thetas, sampler, gid);
  float4 kt = read_imagef(kts, sampler, gid);
  float4 kn = read_imagef(kns, sampler, gid);
  
  int4 test = theta < 0.f;
  theta = select(theta, theta + npis*PI, test);
  
  if (npis == 1) {
    kt = select(kt, -kt, test);
    kn = select(kn, -kn, test);
  }
  
  test = ((fabs(theta - targ_theta) < theta_step/2 ||
           fabs(theta + 2*PI - targ_theta) < theta_step/2 ||
           fabs(theta - 2*PI - targ_theta) < theta_step/2));
  //    test = test && (fabs(kt - targ_kt) < k_step/2 &&
  //                    fabs(kn - targ_kn) < k_step/2));
  
  write_imagef(output, gid, select(0.f, conf, test));
}

float4 contpart(float4 x, float4 degree) {
	float4 invDeg = 1.f/degree;
  x += invDeg/2.f;
  
  float4 result = degree*x;
	float4 temp = select((float4) 1.f, (float4) 0.f, x <= 0.f);
  int4 test;
  test.x = x.x > 0.f && x.x < invDeg.x ? -1 : 0;
  test.y = x.y > 0.f && x.y < invDeg.y ? -1 : 0;
  test.z = x.z > 0.f && x.z < invDeg.z ? -1 : 0;
  test.w = x.w > 0.f && x.w < invDeg.w ? -1 : 0;
  return select(temp, result, test);
}

float4 smoothpart(float4 x, float4 degree) {
	x *= degree;
  
  float4 temp = exp(-1.f/(0.5f + x));
  float4 result = temp/(temp + exp(-1.f/(0.5f - x)));
  int4 test;
  test.x = x.x > -0.5f && x.x < 0.5f ? -1 : 0;
  test.y = x.y > -0.5f && x.y < 0.5f ? -1 : 0;
  test.z = x.z > -0.5f && x.z < 0.5f ? -1 : 0;
  test.w = x.w > -0.5f && x.w < 0.5f ? -1 : 0;
	temp = select((float4) 1.f, (float4) 0.f, x <= -0.5f);
  return select(temp, result, test);
}

kernel void stabilize(read_only image2d_t data1,
                      read_only image2d_t slice_sum,
                      int n, float stab_amt,
                      write_only image2d_t output) {
  int2 gid = get_global_index();
  float4 inval = read_imagef(data1, sampler, gid);
  float4 sliceval = read_imagef(slice_sum, sampler, gid);
  write_imagef(output, gid,
               inval + stab_amt*(fabs(inval) - sliceval/((float) n)));
}

kernel void surround2(read_only image2d_t data1,
                      read_only image2d_t data2,
                      float degree,
                      write_only image2d_t output) {
  int2 gid = get_global_index();
  
  float4 data1_val = read_imagef(data1, sampler, gid);
  float4 data2_val = read_imagef(data2, sampler, gid);
  
  float4 maxval = fmax(fabs(data1_val), fabs(data2_val));
  maxval = select(maxval, (float4) 1.f, maxval == 0.f);
  
  float4 part = smoothpart(data1_val, degree/maxval);
  float4 pos = part;
  float4 neg = 1.f - part;
  float4 ambig = 2.f*pos*neg;
  
  part = smoothpart(data2_val, degree/maxval);
  pos *= part;
  neg *= 1.f - part;
  
  float4 result = data1_val + (pos + neg + ambig)*data2_val;
  write_imagef(output, gid, result);
}

kernel void lland2(read_only image2d_t data1,
                   read_only image2d_t data2,
                   float degree_val,
                   int adapt,
                   float scale,
                   write_only image2d_t output) {
  int2 gid = get_global_index();
  
  float4 data1_val = read_imagef(data1, sampler, gid);
  float4 data2_val = read_imagef(data2, sampler, gid);
  
  float4 degree = degree_val;
  
  if (adapt) {
    float4 maxval = fmax(data1_val, data2_val);
    degree /= maxval;
  }
  
  float4 cp1 = contpart(data1_val, degree);
  float4 cp2 = contpart(data2_val, degree);
  float4 prod_plus_1 = cp1*cp2 + 1.f;
  float4 result = (prod_plus_1 - cp1)*data1_val +
  (prod_plus_1 - cp2)*data2_val;
  write_imagef(output, gid, scale*result);
}

kernel void llor2(read_only image2d_t data1,
                  read_only image2d_t data2,
                  float degree_val,
                  int adapt,
                  float scale,
                  write_only image2d_t output) {
  int2 gid = get_global_index();
  
  float4 data1_val = read_imagef(data1, sampler, gid);
  float4 data2_val = read_imagef(data2, sampler, gid);
  
  float4 degree = degree_val;
  
  if (adapt) {
    float4 maxval = fmax(data1_val, data2_val);
    degree /= maxval;
  }
  
  float4 cp1 = contpart(-data1_val, degree);
  float4 cp2 = contpart(-data2_val, degree);
  float4 prod_plus_1 = cp1*cp2 + 1.f;
  float4 result = (prod_plus_1 - cp1)*data1_val +
  (prod_plus_1 - cp2)*data2_val;
  write_imagef(output, gid, scale*result);
}

kernel void lland3(read_only image2d_t data1,
                   read_only image2d_t data2,
                   read_only image2d_t data3,
                   float degree_val,
                   int adapt,
                   float scale,
                   write_only image2d_t output) {
  int2 gid = get_global_index();
  
  float4 data1_val = read_imagef(data1, sampler, gid);
  float4 data2_val = read_imagef(data2, sampler, gid);
  float4 data3_val = read_imagef(data3, sampler, gid);
  
  float4 degree = degree_val;
  
  if (adapt) {
    float4 maxval = fmax(data1_val, data2_val);
    maxval = fmax(maxval, data3_val);
    degree /= maxval;
  }
  
  float4 cp1 = contpart(data1_val, degree);
  float4 cp2 = contpart(data2_val, degree);
  float4 cp3 = contpart(data3_val, degree);
  float4 prod_plus_1 = cp1*cp2*cp3 + 1.f;
  float4 result = (prod_plus_1 - cp1)*data1_val +
  (prod_plus_1 - cp2)*data2_val +
  (prod_plus_1 - cp3)*data3_val;
  write_imagef(output, gid, scale*result);
}

kernel void llor3(read_only image2d_t data1,
                  read_only image2d_t data2,
                  read_only image2d_t data3,
                  float degree_val,
                  int adapt,
                  float scale,
                  write_only image2d_t output) {
  int2 gid = get_global_index();
  
  float4 data1_val = read_imagef(data1, sampler, gid);
  float4 data2_val = read_imagef(data2, sampler, gid);
  float4 data3_val = read_imagef(data3, sampler, gid);
  
  float4 degree = degree_val;
  
  if (adapt) {
    float4 maxval = fmax(data1_val, data2_val);
    maxval = fmax(maxval, data3_val);
    degree /= maxval;
  }
  
  float4 cp1 = contpart(-data1_val, degree);
  float4 cp2 = contpart(-data2_val, degree);
  float4 cp3 = contpart(-data3_val, degree);
  float4 prod_plus_1 = cp1*cp2*cp3 + 1.f;
  float4 result = (prod_plus_1 - cp1)*data1_val +
  (prod_plus_1 - cp2)*data2_val +
  (prod_plus_1 - cp3)*data3_val;
  write_imagef(output, gid, scale*result);
}

kernel void lland4(read_only image2d_t data1,
                   read_only image2d_t data2,
                   read_only image2d_t data3,
                   read_only image2d_t data4,
                   float degree_val,
                   int adapt,
                   float scale,
                   write_only image2d_t output) {
  int2 gid = get_global_index();
  
  float4 data1_val = read_imagef(data1, sampler, gid);
  float4 data2_val = read_imagef(data2, sampler, gid);
  float4 data3_val = read_imagef(data3, sampler, gid);
  float4 data4_val = read_imagef(data4, sampler, gid);
  
  float4 degree = degree_val;
  
  if (adapt) {
    float4 maxval = fmax(data1_val, data2_val);
    maxval = fmax(maxval, data3_val);
    maxval = fmax(maxval, data4_val);
    degree /= maxval;
  }
  
  float4 cp1 = contpart(data1_val, degree);
  float4 cp2 = contpart(data2_val, degree);
  float4 cp3 = contpart(data3_val, degree);
  float4 cp4 = contpart(data4_val, degree);
  float4 prod_plus_1 = cp1*cp2*cp3*cp4 + 1.f;
  float4 result = (prod_plus_1 - cp1)*data1_val +
  (prod_plus_1 - cp2)*data2_val +
  (prod_plus_1 - cp3)*data3_val +
  (prod_plus_1 - cp4)*data4_val;
  write_imagef(output, gid, scale*result);
}

kernel void llor4(read_only image2d_t data1,
                  read_only image2d_t data2,
                  read_only image2d_t data3,
                  read_only image2d_t data4,
                  float degree_val,
                  int adapt,
                  float scale,
                  write_only image2d_t output) {
  int2 gid = get_global_index();
  
  float4 data1_val = read_imagef(data1, sampler, gid);
  float4 data2_val = read_imagef(data2, sampler, gid);
  float4 data3_val = read_imagef(data3, sampler, gid);
  float4 data4_val = read_imagef(data4, sampler, gid);
  
  float4 degree = degree_val;
  
  if (adapt) {
    float4 maxval = fmax(data1_val, data2_val);
    maxval = fmax(maxval, data3_val);
    maxval = fmax(maxval, data4_val);
    degree /= maxval;
  }
  
  float4 cp1 = contpart(-data1_val, degree);
  float4 cp2 = contpart(-data2_val, degree);
  float4 cp3 = contpart(-data3_val, degree);
  float4 cp4 = contpart(-data4_val, degree);
  float4 prod_plus_1 = cp1*cp2*cp3*cp4 + 1.f;
  float4 result = (prod_plus_1 - cp1)*data1_val +
  (prod_plus_1 - cp2)*data2_val +
  (prod_plus_1 - cp3)*data3_val +
  (prod_plus_1 - cp4)*data4_val;
  write_imagef(output, gid, scale*result);
}

kernel void lland5(read_only image2d_t data1,
                   read_only image2d_t data2,
                   read_only image2d_t data3,
                   read_only image2d_t data4,
                   read_only image2d_t data5,
                   float degree_val,
                   int adapt,
                   float scale,
                   write_only image2d_t output) {
  int2 gid = get_global_index();
  
  float4 data1_val = read_imagef(data1, sampler, gid);
  float4 data2_val = read_imagef(data2, sampler, gid);
  float4 data3_val = read_imagef(data3, sampler, gid);
  float4 data4_val = read_imagef(data4, sampler, gid);
  float4 data5_val = read_imagef(data5, sampler, gid);
  
  float4 degree = degree_val;
  
  if (adapt) {
    float4 maxval = fmax(data1_val, data2_val);
    maxval = fmax(maxval, data3_val);
    maxval = fmax(maxval, data4_val);
    maxval = fmax(maxval, data5_val);
    degree /= maxval;
  }
  
  float4 cp1 = contpart(data1_val, degree);
  float4 cp2 = contpart(data2_val, degree);
  float4 cp3 = contpart(data3_val, degree);
  float4 cp4 = contpart(data4_val, degree);
  float4 cp5 = contpart(data5_val, degree);
  float4 prod_plus_1 = cp1*cp2*cp3*cp4*cp5 + 1.f;
  float4 result = (prod_plus_1 - cp1)*data1_val +
  (prod_plus_1 - cp2)*data2_val +
  (prod_plus_1 - cp3)*data3_val +
  (prod_plus_1 - cp4)*data4_val +
  (prod_plus_1 - cp5)*data5_val;
  write_imagef(output, gid, scale*result);
}

kernel void llor5(read_only image2d_t data1,
                  read_only image2d_t data2,
                  read_only image2d_t data3,
                  read_only image2d_t data4,
                  read_only image2d_t data5,
                  float degree_val,
                  int adapt,
                  float scale,
                  write_only image2d_t output) {
  int2 gid = get_global_index();
  
  float4 data1_val = read_imagef(data1, sampler, gid);
  float4 data2_val = read_imagef(data2, sampler, gid);
  float4 data3_val = read_imagef(data3, sampler, gid);
  float4 data4_val = read_imagef(data4, sampler, gid);
  float4 data5_val = read_imagef(data5, sampler, gid);
  
  float4 degree = degree_val;
  
  if (adapt) {
    float4 maxval = fmax(data1_val,
                         data2_val);
    maxval = fmax(maxval, data3_val);
    maxval = fmax(maxval, data4_val);
    maxval = fmax(maxval, data5_val);
    degree /= maxval;
  }
  
  float4 cp1 = contpart(-data1_val, degree);
  float4 cp2 = contpart(-data2_val, degree);
  float4 cp3 = contpart(-data3_val, degree);
  float4 cp4 = contpart(-data4_val, degree);
  float4 cp5 = contpart(-data5_val, degree);
  float4 prod_plus_1 = cp1*cp2*cp3*cp4*cp5 + 1.f;
  float4 result = (prod_plus_1 - cp1)*data1_val +
  (prod_plus_1 - cp2)*data2_val +
  (prod_plus_1 - cp3)*data3_val +
  (prod_plus_1 - cp4)*data4_val +
  (prod_plus_1 - cp5)*data5_val;
  write_imagef(output, gid, scale*result);
}

kernel void lland6(read_only image2d_t data1,
                   read_only image2d_t data2,
                   read_only image2d_t data3,
                   read_only image2d_t data4,
                   read_only image2d_t data5,
                   read_only image2d_t data6,
                   float degree_val,
                   int adapt,
                   float scale,
                   write_only image2d_t output) {
  int2 gid = get_global_index();
  
  float4 data1_val = read_imagef(data1, sampler, gid);
  float4 data2_val = read_imagef(data2, sampler, gid);
  float4 data3_val = read_imagef(data3, sampler, gid);
  float4 data4_val = read_imagef(data4, sampler, gid);
  float4 data5_val = read_imagef(data5, sampler, gid);
  float4 data6_val = read_imagef(data6, sampler, gid);
  
  float4 degree = degree_val;
  
  if (adapt) {
    float4 maxval = fmax(data1_val, data2_val);
    maxval = fmax(maxval, data3_val);
    maxval = fmax(maxval, data4_val);
    maxval = fmax(maxval, data5_val);
    maxval = fmax(maxval, data6_val);
    degree /= maxval;
  }
  
  float4 cp1 = contpart(data1_val, degree);
  float4 cp2 = contpart(data2_val, degree);
  float4 cp3 = contpart(data3_val, degree);
  float4 cp4 = contpart(data4_val, degree);
  float4 cp5 = contpart(data5_val, degree);
  float4 cp6 = contpart(data6_val, degree);
  float4 prod_plus_1 = cp1*cp2*cp3*cp4*cp5*cp6 + 1.f;
  float4 result = (prod_plus_1 - cp1)*data1_val +
  (prod_plus_1 - cp2)*data2_val +
  (prod_plus_1 - cp3)*data3_val +
  (prod_plus_1 - cp4)*data4_val +
  (prod_plus_1 - cp5)*data5_val +
  (prod_plus_1 - cp6)*data6_val;
  write_imagef(output, gid, scale*result);
}

kernel void llor6(read_only image2d_t data1,
                  read_only image2d_t data2,
                  read_only image2d_t data3,
                  read_only image2d_t data4,
                  read_only image2d_t data5,
                  read_only image2d_t data6,
                  float degree_val,
                  int adapt,
                  float scale,
                  write_only image2d_t output) {
  int2 gid = get_global_index();
  
  float4 data1_val = read_imagef(data1, sampler, gid);
  float4 data2_val = read_imagef(data2, sampler, gid);
  float4 data3_val = read_imagef(data3, sampler, gid);
  float4 data4_val = read_imagef(data4, sampler, gid);
  float4 data5_val = read_imagef(data5, sampler, gid);
  float4 data6_val = read_imagef(data6, sampler, gid);
  
  float4 degree = degree_val;
  
  if (adapt) {
    float4 maxval = fmax(data1_val, data2_val);
    maxval = fmax(maxval, data3_val);
    maxval = fmax(maxval, data4_val);
    maxval = fmax(maxval, data5_val);
    maxval = fmax(maxval, data6_val);
    degree /= maxval;
  }
  
  float4 cp1 = contpart(-data1_val, degree);
  float4 cp2 = contpart(-data2_val, degree);
  float4 cp3 = contpart(-data3_val, degree);
  float4 cp4 = contpart(-data4_val, degree);
  float4 cp5 = contpart(-data5_val, degree);
  float4 cp6 = contpart(-data6_val, degree);
  float4 prod_plus_1 = cp1*cp2*cp3*cp4*cp5*cp6 + 1.f;
  float4 result = (prod_plus_1 - cp1)*data1_val +
  (prod_plus_1 - cp2)*data2_val +
  (prod_plus_1 - cp3)*data3_val +
  (prod_plus_1 - cp4)*data4_val +
  (prod_plus_1 - cp5)*data5_val +
  (prod_plus_1 - cp6)*data6_val;
  write_imagef(output, gid, scale*result);
}