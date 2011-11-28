#if defined(USE_CPU)
kernel void filter(input_t image,
                   constant filtval_t* filter,
                   int filt_width, int filt_height,
                   output_t output) {
  int im_width = get_global_size(0);
  int im_height = get_global_size(1);
  
  int im_x = get_global_id(0);
  int im_y = get_global_id(1);
  
  int out_indx = im_y*im_width + im_x;
  
  int half_filt_width = filt_width/2;
  int half_filt_height = filt_height/2;
  
  int im_base_x = im_x - half_filt_width;
  int im_base_y = im_y - half_filt_height;
  
  int min_valid_filt_y = im_y < half_filt_height ?
                         half_filt_height - im_y : 0;
  int max_valid_filt_y = im_base_y + filt_height > im_height ?
                         im_height - im_base_y - 1 : filt_height - 1;
  int min_valid_filt_x = im_x < half_filt_width ?
                         half_filt_width - im_x : 0;
  int max_valid_filt_x = im_base_x + filt_width > im_width ?
                         im_width - im_base_x - 1 : filt_width - 1;
  
  float4 sum4 = 0;
  for (int filt_y = min_valid_filt_y; filt_y <= max_valid_filt_y; ++filt_y) {
    int filt_offset_base = filt_y*filt_width;
    int filt_offset = filt_offset_base + min_valid_filt_x;
    
    int im_y = im_base_y + filt_y;
    int im_indx_base = im_y*im_width + im_base_x + min_valid_filt_x;
    
    int filt_x = min_valid_filt_x;
    int filt_x4 = 0;
    while (filt_x <= max_valid_filt_x - 3) {
      float4 filt4 = loadf4(filt_x4, filter + filt_offset);
      float4 im4 = loadm4(filt_x4, image + im_indx_base);
      sum4 += filt4*im4;
      filt_x += 4;
      ++filt_x4;
    }
    
    int x_mod = max_valid_filt_x - filt_x + 1;
    int filt_indx = filt_offset_base + filt_x;
    int im_indx = im_indx_base + filt_x;
    
    if (x_mod == 1) {
      sum4.x += loadf1(filt_indx, filter)*loadm1(im_indx, image);
    }
    else if (x_mod == 2) {
      sum4.x += loadf1(filt_indx, filter)*loadm1(im_indx, image);
      sum4.y += loadf1(filt_indx + 1, filter)*loadm1(im_indx + 1, image);
    }
    else if (x_mod == 3) {
      sum4.x += loadf1(filt_indx, filter)*loadm1(im_indx, image);
      sum4.y += loadf1(filt_indx + 1, filter)*loadm1(im_indx + 1, image);
      sum4.z += loadf1(filt_indx + 2, filter)*loadm1(im_indx + 2, image);
    }
  }
  
  storem1(sum4.x + sum4.y + sum4.z + sum4.w, out_indx, output);
}

kernel void filter_sparse(input_t image,
                          constant sparse_filtval* filter,
                          int filt_width,
                          int filt_height,
                          int num_filt_elems,
                          output_t output) {
  int im_width = get_global_size(0);
  int im_height = get_global_size(1);
  
  int im_x = get_global_id(0);
  int im_y = get_global_id(1);
  
  int out_indx = im_y*im_width + im_x;
  
  int half_filt_width = filt_width/2;
  int half_filt_height = filt_height/2;
  
  int im_base_x = im_x - half_filt_width;
  int im_base_y = im_y - half_filt_height;
  
  int min_valid_filt_y = im_y < half_filt_height ? half_filt_height - im_y : 0;
  int max_valid_filt_y = im_base_y + filt_height > im_height ?
                         im_height - im_base_y - 1 : filt_height - 1;
  int min_valid_filt_x = im_x < half_filt_width ? half_filt_width - im_x : 0;
  int max_valid_filt_x = im_base_x + filt_width > im_width ?
                         im_width - im_base_x - 1 : filt_width - 1;

  float sum = 0;  
  for (int i = 0; i < num_filt_elems; ++i) {
    sparse_float sv = loads1(i, filter);
    if (sv.x < min_valid_filt_x || sv.x > max_valid_filt_x ||
        sv.y < min_valid_filt_y || sv.y > max_valid_filt_y) {
      continue;
    }
    
    int im_indx = (im_base_y + sv.y)*im_width + im_base_x + sv.x;
    sum += loadm1(im_indx, image)*sv.val;
  }
  
  storem1(sum, out_indx, output);
}
#elif !defined(USE_TEXTURES)
kernel void filter(input_t image,
                   local float* im_cache,
                   constant filtval_t* filter,
                   local float* filt_cache,
                   int filt_width, int filt_height,
                   output_t output) {
  int im_x = get_global_id(0);
  int im_y = get_global_id(1);
  int im_width = get_global_size(0);
  int im_height = get_global_size(1);
  int local_x = get_local_id(0);
  int local_y = get_local_id(1);
  int group_id_x = get_group_id(0);
  //  int group_id_y = get_group_id(1);
  //  int num_groups_x = get_num_groups(0);
  //  int num_groups_y = get_num_groups(1);
  int group_width = 16; // im_width/num_groups_x;
  int group_height = 16; // im_height/num_groups_y;
  
  int im_indx = im_y*im_width + im_x;
  
  int half_filt_width = filt_width/2;
  int half_filt_height = filt_height/2;
  
  int apron_rem = half_filt_width%4;
  int apron_padding = apron_rem ? 4 - apron_rem : 0;
  int apron_width = half_filt_width + apron_padding;
  int n_lr_apron_quads = apron_width/4;
  
  int bottom_apron_height = half_filt_height;
  int top_apron_height = half_filt_height - !(filt_height%2);
  
  int im_cache_width = 4*group_width + 2*apron_width;
  int im_cache_quad_width = im_cache_width/4;
  
  local float *local_row_base = im_cache +
    (local_y + bottom_apron_height)*im_cache_width + apron_width;
  
  for (int y = local_y; y < filt_height; y += group_height) {
    int row_base = y*filt_width;
    for (int x = local_x; x < filt_width; x += group_width)
      filt_cache[row_base + x] = loadf1(row_base + x, filter);
  }
  
  barrier(CLK_LOCAL_MEM_FENCE);
  
  vstore4(loadm4(im_indx, image), local_x, local_row_base);
  
  if (local_x - group_width >= -n_lr_apron_quads) {
    int quad_offs = -group_width;
    int im_indx_off = (im_x + quad_offs >= 0 ? quad_offs : -im_x);
    vstore4(loadm4(im_indx + im_indx_off, image),
            local_x + quad_offs, local_row_base);
  }
  
  if (local_x < n_lr_apron_quads) {
    int quad_offs = group_width;
    int im_indx_off = (im_x + quad_offs < im_width ?
                       quad_offs : group_width - local_x - 1);
    vstore4(loadm4(im_indx + im_indx_off, image),
            local_x + quad_offs, local_row_base);
  }
  
  if (local_y - group_height >= -bottom_apron_height) {
    int im_quad_offs = -group_height*im_width;
    int local_quad_offs = -group_height*im_cache_quad_width + local_x;
    int im_indx_off =
      (im_y - group_height >= 0 ? im_quad_offs : -im_y*im_width);
    vstore4(loadm4(im_indx + im_indx_off, image),
            local_quad_offs, local_row_base);
    
    if (local_x - group_width >= -n_lr_apron_quads) {
      int quad_offs = -group_width;
      int new_im_indx_off = im_indx_off +
        (im_x + quad_offs >= 0 ? quad_offs : -im_x);
      vstore4(loadm4(im_indx + new_im_indx_off, image),
              local_quad_offs + quad_offs, local_row_base);
    }
    
    if (local_x < n_lr_apron_quads) {
      int quad_offs = group_width;
      int new_im_indx_off = im_indx_off +
        (im_x + quad_offs < im_width ?
         quad_offs : group_width - local_x - 1);
      vstore4(loadm4(im_indx + new_im_indx_off, image),
              local_quad_offs + quad_offs, local_row_base);
    }
  }
  
  if (local_y < top_apron_height) {
    int im_quad_offs = group_height*im_width;
    int local_quad_offs = group_height*im_cache_quad_width + local_x;
    int im_indx_off = (im_y + group_height < im_height ?
                       im_quad_offs : (group_height - 1 - local_y)*im_width);
    vstore4(loadm4(im_indx + im_indx_off, image),
            local_quad_offs, local_row_base);
    
    if (local_x - group_width >= -n_lr_apron_quads) {
      int quad_offs = -group_width;
      int new_im_indx_off = im_indx_off +
        (im_x + quad_offs >= 0 ? quad_offs : -im_x);
      vstore4(loadm4(im_indx + new_im_indx_off, image),
              local_quad_offs + quad_offs, local_row_base);
    }
    
    if (local_x < n_lr_apron_quads) {
      int quad_offs = group_width;
      int new_im_indx_off =
        im_indx_off + (im_x + quad_offs < im_width ?
                       quad_offs : group_width - local_x - 1);
      vstore4(loadm4(im_indx + new_im_indx_off, image),
              local_quad_offs + quad_offs, local_row_base);
    }
  }
  
  barrier(CLK_LOCAL_MEM_FENCE);
  
  float4 sum = 0.f;
  for (int filt_y = 0; filt_y < filt_height; filt_y++) {
    int filt_row_base = filt_y*filt_width;
    int im_cache_row_base = (filt_y + local_y)*im_cache_width + apron_padding;
    
    for (int filt_x = 0; filt_x < filt_width; filt_x++) {
      int im_cache_indx = im_cache_row_base + filt_x + local_x;
      float filt_val = loadf1(filt_row_base + filt_x, filter);
      
      sum.x += filt_val*im_cache[im_cache_indx];
      sum.y += filt_val*im_cache[im_cache_indx + group_width];
      sum.z += filt_val*im_cache[im_cache_indx + 2*group_width];
      sum.w += filt_val*im_cache[im_cache_indx + 3*group_width];
    }
  }
  
  int out_indx_base = 4*(im_y*im_width + group_id_x*group_width) + local_x;
  
  storem1(sum.x, out_indx_base, output);
  storem1(sum.y, out_indx_base + group_width, output);
  storem1(sum.z, out_indx_base + 2*group_width, output);
  storem1(sum.w, out_indx_base + 3*group_width, output);
}

kernel void filter_sparse(input_t image,
                          local float* im_cache,
                          constant sparse_filtval* filter,
                          local sparse_float* filt_cache,
                          int filt_width,
                          int filt_height,
                          int num_filt_elems,
                          output_t output) {
  int im_x = get_global_id(0);
  int im_y = get_global_id(1);
  int im_width = get_global_size(0);
  int im_height = get_global_size(1);
  int local_x = get_local_id(0);
  int local_y = get_local_id(1);
  int group_id_x = get_group_id(0);
//  int group_id_y = get_group_id(1);
//  int num_groups_x = get_num_groups(0);
//  int num_groups_y = get_num_groups(1);
  int group_width = 16; // im_width/num_groups_x;
  int group_height = 16; // im_height/num_groups_y;
  
  int im_indx = im_y*im_width + im_x;
  
  int half_filt_width = filt_width/2;
  int half_filt_height = filt_height/2;
  
  int apron_rem = half_filt_width%4;
  int apron_padding = apron_rem ? 4 - apron_rem : 0;
  int apron_width = half_filt_width + apron_padding;
  int n_lr_apron_quads = apron_width/4;
  
  int bottom_apron_height = half_filt_height;
  int top_apron_height = half_filt_height - !(filt_height%2);
  
  int im_cache_width = 4*group_width + 2*apron_width;
  int im_cache_quad_width = im_cache_width/4;
  
  local float *local_row_base = im_cache +
    (local_y + bottom_apron_height)*im_cache_width + apron_width;
  
  for (int i = local_x; i < num_filt_elems; i += group_width)
    filt_cache[i] = loads1(i, filter);
  
  barrier(CLK_LOCAL_MEM_FENCE);
  
  vstore4(loadm4(im_indx, image), local_x, local_row_base);
  
  if (local_x - group_width >= -n_lr_apron_quads) {
    int quad_offs = -group_width;
    int im_indx_off = (im_x + quad_offs >= 0 ? quad_offs : -im_x);
    vstore4(loadm4(im_indx + im_indx_off, image),
            local_x + quad_offs, local_row_base);
  }
  
  if (local_x < n_lr_apron_quads) {
    int quad_offs = group_width;
    int im_indx_off = (im_x + quad_offs < im_width ?
                       quad_offs : group_width - local_x - 1);
    vstore4(loadm4(im_indx + im_indx_off, image),
            local_x + quad_offs, local_row_base);
  }
  
  if (local_y - group_height >= -bottom_apron_height) {
    int im_quad_offs = -group_height*im_width;
    int local_quad_offs = -group_height*im_cache_quad_width + local_x;
    int im_indx_off = im_y - group_height >= 0 ?
                      im_quad_offs : -im_y*im_width;
    vstore4(loadm4(im_indx + im_indx_off, image),
            local_quad_offs, local_row_base);
    
    if (local_x - group_width >= -n_lr_apron_quads) {
      int quad_offs = -group_width;
      int new_im_indx_off = im_indx_off +
        (im_x + quad_offs >= 0 ? quad_offs : -im_x);
      vstore4(loadm4(im_indx + new_im_indx_off, image),
              local_quad_offs + quad_offs, local_row_base);
    }
    
    if (local_x < n_lr_apron_quads) {
      int quad_offs = group_width;
      int new_im_indx_off = im_indx_off +
        (im_x + quad_offs < im_width ?
         quad_offs : group_width - local_x - 1);
      vstore4(loadm4(im_indx + new_im_indx_off, image),
              local_quad_offs + quad_offs, local_row_base);
    }
  }
  
  if (local_y < top_apron_height) {
    int im_quad_offs = group_height*im_width;
    int local_quad_offs = group_height*im_cache_quad_width + local_x;
    int im_indx_off =
      (im_y + group_height < im_height ?
       im_quad_offs : (group_height - 1 - local_y)*im_width);
    vstore4(loadm4(im_indx + im_indx_off, image),
            local_quad_offs, local_row_base);
    
    if (local_x - group_width >= -n_lr_apron_quads) {
      int quad_offs = -group_width;
      int new_im_indx_off = im_indx_off +
        (im_x + quad_offs >= 0 ? quad_offs : -im_x);
      vstore4(loadm4(im_indx + new_im_indx_off, image),
              local_quad_offs + quad_offs, local_row_base);
    }
    
    if (local_x < n_lr_apron_quads) {
      int quad_offs = group_width;
      int new_im_indx_off =
        im_indx_off + (im_x + quad_offs < im_width ?
                       quad_offs : group_width - local_x - 1);
      vstore4(loadm4(im_indx + new_im_indx_off, image),
              local_quad_offs + quad_offs, local_row_base);
    }
  }
  
  barrier(CLK_LOCAL_MEM_FENCE);
  
  float4 sum = 0.f;
  for (int filt_i = 0; filt_i < num_filt_elems; filt_i++) {
    sparse_float filt_val = filt_cache[filt_i];
    int im_cache_row_base =
      (filt_val.y + local_y)*im_cache_width + apron_padding;
    int im_cache_indx = im_cache_row_base + filt_val.x + local_x;
      
    sum.x += filt_val.val*im_cache[im_cache_indx];
    sum.y += filt_val.val*im_cache[im_cache_indx + group_width];
    sum.z += filt_val.val*im_cache[im_cache_indx + 2*group_width];
    sum.w += filt_val.val*im_cache[im_cache_indx + 3*group_width];
  }
  
  int out_indx_base = 4*(im_y*im_width + group_id_x*group_width) + local_x;
  
  storem1(sum.x, out_indx_base, output);
  storem1(sum.y, out_indx_base + group_width, output);
  storem1(sum.z, out_indx_base + 2*group_width, output);
  storem1(sum.w, out_indx_base + 3*group_width, output);
}
#else
#if 1
kernel void filter(input_t image,
                   constant filtval_t* filter,
                   int filt_width, int filt_height,
                   output_t output) {
  int2 gid = get_global_index();
  int2 fc;
  
  filt_width /= 4;
  int half_filt_width = filt_width/2;
  int half_filt_height = filt_height/2;
  int filt_y_lim = filt_height - half_filt_height;
  
  int2 foff = (int2)(half_filt_width, half_filt_height);
  int2 xoff = (int2)(1, 0);
  
  float4 sum = 0.f;
  if (filt_width%2 == 0) {
    float4 imval1, imval2, fval;
    
    for (fc.y = -half_filt_height; fc.y < filt_y_lim; fc.y++) {
      fc.x = -half_filt_width;
      
      imval2 = load(gid + fc, image);
      for (; fc.x < half_filt_width; fc.x++) {
        fval = loadf4((fc.y + foff.y)*filt_width + fc.x + foff.x, filter);
        
        imval1 = imval2;
        imval2 = load(gid + fc + xoff, image);
        
        sum += fval.x*imval1;
        sum += fval.y*((float4)(imval1.y, imval1.z, imval1.w, imval2.x));
        sum += fval.z*((float4)(imval1.z, imval1.w, imval2.x, imval2.y));
        sum += fval.w*((float4)(imval1.w, imval2.x, imval2.y, imval2.z));
      }
    }
  }
  else {
    float4 imval1, imval2, imval3, fval;
    
    for (fc.y = -half_filt_height; fc.y < filt_y_lim; fc.y++) {
      fc.x = -half_filt_width;
      
      imval2 = load(gid + fc - xoff, image);
      imval3 = load(gid + fc, image);
      for (; fc.x <= half_filt_width; fc.x++) {
        fval = loadf4((fc.y + foff.y)*filt_width + fc.x + foff.x, filter);
        
        imval1 = imval2;
        imval2 = imval3;
        imval3 = load(gid + fc + xoff, image);
        
        sum += fval.x*((float4)(imval1.z, imval1.w, imval2.x, imval2.y));
        sum += fval.y*((float4)(imval1.w, imval2.x, imval2.y, imval2.z));
        sum += fval.z*imval2;
        sum += fval.w*((float4)(imval2.y, imval2.z, imval2.w, imval3.x));
      }
    }
  }
  
  store(sum, gid, output);
}

kernel void filter_sparse(input_t image,
                          constant sparse_filtval* filter,
                          int filt_width, int filt_height,
                          int num_filt_elems,
                          output_t output) {
  int2 gid = get_global_index();
  
  int half_filt_width = filt_width/2;
  int half_filt_height = filt_height/2;
  
  int2 fcenter = (int2)((half_filt_width + 3)/4, half_filt_height);
  int2 xinc = (int2)(1, 0);
  int2 origin = gid - fcenter;
  int center_mod = 4 - (half_filt_width%4);
  
  float4 sum = 0.f;
  for (int i = 0; i < num_filt_elems; ++i) {
    sparse_float sf = loads1(i, filter);
    int2 foff = (int2)((sf.x + center_mod)/4, sf.y);
    float4 im1 = load(origin + foff, image);
    
    float4 shifted, im2;
    switch ((sf.x + center_mod)%4) {
      case 0:
        shifted = im1;
        break;
        
      case 1:
        im2 = load(origin + foff + xinc, image);
        shifted = (float4)(im1.y, im1.z, im1.w, im2.x);
        break;
        
      case 2:
        im2 = load(origin + foff + xinc, image);
        shifted = (float4)(im1.z, im1.w, im2.x, im2.y);
        break;
        
      case 3:
        im2 = load(origin + foff + xinc, image);
        shifted = (float4)(im1.w, im2.x, im2.y, im2.z);
        break;
    }
    
    sum += sf.val*shifted;
  }
  
  store(sum, gid, output);
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
  
  vstore4(load(gid, image), 0, im_cache + local_indx);
  
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
  
  store(sum, gid, output);
}
#endif
#endif
