CLIP_STRINGIFY(
kernel void filter_cpu(global imval* image, constant float *filter,
                       int filt_width, int filt_height,
                       global imval* output) {
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
  for (int filt_y = min_valid_filt_y;
       filt_y <= max_valid_filt_y;
       ++filt_y) {
    int filt_offset_base = filt_y*filt_width;
    int filt_offset = filt_offset_base + min_valid_filt_x;
    
    int im_y = im_base_y + filt_y;
    int im_indx_base = im_y*im_width + im_base_x + min_valid_filt_x;
    
    int filt_x = min_valid_filt_x;
    int filt_x4 = 0;
    while (filt_x <= max_valid_filt_x - 3) {
      float4 filt4 = vload4(filt_x4, filter + filt_offset);
      float4 im4 = load_imval4(im_indx_base + filt_x4, image);
      sum4 += filt4*im4;
      filt_x += 4;
      ++filt_x4;
    }
    
    int x_mod = max_valid_filt_x - filt_x + 1;
    int filt_indx = filt_offset_base + filt_x;
    int im_indx = im_indx_base + filt_x;
    
    if (x_mod == 1)
      sum4.x += filter[filt_indx]*load_imval(im_indx, image);
    else if (x_mod == 2) {
      sum4.x += filter[filt_indx]*load_imval(im_indx, image);
      sum4.y += filter[filt_indx + 1]*load_imval(im_indx + 1, image);
    }
    else if (x_mod == 3) {
      sum4.x += filter[filt_indx]*load_imval(im_indx, image);
      sum4.y += filter[filt_indx + 1]*load_imval(im_indx + 1, image);
      sum4.z += filter[filt_indx + 2]*load_imval(im_indx + 2, image);
    }
  }
  
  store_imval(sum4.x + sum4.y + sum4.z + sum4.w, out_indx, output);
}

kernel void filter_gpu(global imval4* image,
                       local float *im_cache,
                       constant float *filter,
                       local float *filt_cache,
                       int filt_width,
                       int filt_height,
                       global imval* output) {
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
      filt_cache[row_base + x] = filter[row_base + x];
  }
  
  barrier(CLK_LOCAL_MEM_FENCE);
  
  vstore4(load_imval(im_indx, image), local_x, local_row_base);
  
  if (local_x - group_width >= -n_lr_apron_quads) {
    int quad_offs = -group_width;
    int im_indx_off = (im_x + quad_offs >= 0 ? quad_offs : -im_x);
    vstore4(load_imval(im_indx + im_indx_off, image),
            local_x + quad_offs, local_row_base);
  }
  
  if (local_x < n_lr_apron_quads) {
    int quad_offs = group_width;
    int im_indx_off = (im_x + quad_offs < im_width ?
                       quad_offs : group_width - local_x - 1);
    vstore4(load_imval(im_indx + im_indx_off, image),
            local_x + quad_offs, local_row_base);
  }
  
  if (local_y - group_height >= -bottom_apron_height) {
    int im_quad_offs = -group_height*im_width;
    int local_quad_offs = -group_height*im_cache_quad_width + local_x;
    int im_indx_off =
    (im_y - group_height >= 0 ? im_quad_offs : -im_y*im_width);
    vstore4(load_imval(im_indx + im_indx_off, image),
            local_quad_offs, local_row_base);
    
    if (local_x - group_width >= -n_lr_apron_quads) {
      int quad_offs = -group_width;
      int new_im_indx_off = im_indx_off +
      (im_x + quad_offs >= 0 ? quad_offs : -im_x);
      vstore4(load_imval(im_indx + new_im_indx_off, image),
              local_quad_offs + quad_offs, local_row_base);
    }
    
    if (local_x < n_lr_apron_quads) {
      int quad_offs = group_width;
      int new_im_indx_off = im_indx_off +
      (im_x + quad_offs < im_width ?
       quad_offs : group_width - local_x - 1);
      vstore4(load_imval(im_indx + new_im_indx_off, image),
              local_quad_offs + quad_offs, local_row_base);
    }
  }
  
  if (local_y < top_apron_height) {
    int im_quad_offs = group_height*im_width;
    int local_quad_offs = group_height*im_cache_quad_width + local_x;
    int im_indx_off = (im_y + group_height < im_height ?
                       im_quad_offs : (group_height - 1 - local_y)*im_width);
    vstore4(load_imval(im_indx + im_indx_off, image),
            local_quad_offs, local_row_base);
    
    if (local_x - group_width >= -n_lr_apron_quads) {
      int quad_offs = -group_width;
      int new_im_indx_off = im_indx_off +
      (im_x + quad_offs >= 0 ? quad_offs : -im_x);
      vstore4(load_imval(im_indx + new_im_indx_off, image),
              local_quad_offs + quad_offs, local_row_base);
    }
    
    if (local_x < n_lr_apron_quads) {
      int quad_offs = group_width;
      int new_im_indx_off = im_indx_off +
      (im_x + quad_offs < im_width ?
       quad_offs : group_width - local_x - 1);
      vstore4(load_imval(im_indx + new_im_indx_off, image),
              local_quad_offs + quad_offs, local_row_base);
    }
  }
  
  barrier(CLK_LOCAL_MEM_FENCE);
  
  float4 sum = 0.f;
  for (int filt_y = 0; filt_y < filt_height; filt_y++) {
    int filt_row_base = filt_y*filt_width;
    int im_cache_row_base =
    (filt_y + local_y)*im_cache_width + apron_padding;
    
    for (int filt_x = 0; filt_x < filt_width; filt_x++) {
      int im_cache_indx = im_cache_row_base + filt_x + local_x;
      float filt_val = filter[filt_row_base + filt_x];
      
      sum.x += filt_val*im_cache[im_cache_indx];
      sum.y += filt_val*im_cache[im_cache_indx + group_width];
      sum.z += filt_val*im_cache[im_cache_indx + 2*group_width];
      sum.w += filt_val*im_cache[im_cache_indx + 3*group_width];
    }
  }
  
  int out_indx_base = 4*(im_y*im_width + group_id_x*group_width) + local_x;
  
  store_imval(sum.x, out_indx_base, output);
  store_imval(sum.y, out_indx_base + group_width, output);
  store_imval(sum.z, out_indx_base + 2*group_width, output);
  store_imval(sum.w, out_indx_base + 3*group_width, output);
}

typedef struct sparse_value {
  short x;
  short y;
  float val;
} sparse_value;

kernel void filter_sparse_cpu(global imval* image,
                              constant sparse_value *filter,
                              int filt_width,
                              int filt_height,
                              int num_filt_elems,
                              global imval* output) {
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
    sparse_value sv = filter[i];
    if (sv.x < min_valid_filt_x || sv.x > max_valid_filt_x ||
        sv.y < min_valid_filt_y || sv.y > max_valid_filt_y) {
      continue;
    }
    
    int im_indx = (im_base_y + sv.y)*im_width + im_base_x + sv.x;
    sum += load_imval(im_indx, image)*sv.val;
  }
  
  store_imval(sum, out_indx, output);
}

kernel void filter_sparse_gpu(global imval4* image,
                              local float *im_cache,
                              constant sparse_value *filter,
                              local sparse_value *filt_cache,
                              int filt_width,
                              int filt_height,
                              int num_filt_elems,
                              global imval* output) {
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
    filt_cache[i] = filter[i];
  
  barrier(CLK_LOCAL_MEM_FENCE);
  
  vstore4(load_imval(im_indx, image), local_x, local_row_base);
  
  if (local_x - group_width >= -n_lr_apron_quads) {
    int quad_offs = -group_width;
    int im_indx_off = (im_x + quad_offs >= 0 ? quad_offs : -im_x);
    vstore4(load_imval(im_indx + im_indx_off, image),
            local_x + quad_offs, local_row_base);
  }
  
  if (local_x < n_lr_apron_quads) {
    int quad_offs = group_width;
    int im_indx_off = (im_x + quad_offs < im_width ?
                       quad_offs : group_width - local_x - 1);
    vstore4(load_imval(im_indx + im_indx_off, image),
            local_x + quad_offs, local_row_base);
  }
  
  if (local_y - group_height >= -bottom_apron_height) {
    int im_quad_offs = -group_height*im_width;
    int local_quad_offs = -group_height*im_cache_quad_width + local_x;
    int im_indx_off =
    (im_y - group_height >= 0 ? im_quad_offs : -im_y*im_width);
    vstore4(load_imval(im_indx + im_indx_off, image),
            local_quad_offs, local_row_base);
    
    if (local_x - group_width >= -n_lr_apron_quads) {
      int quad_offs = -group_width;
      int new_im_indx_off = im_indx_off +
      (im_x + quad_offs >= 0 ? quad_offs : -im_x);
      vstore4(load_imval(im_indx + new_im_indx_off, image),
              local_quad_offs + quad_offs, local_row_base);
    }
    
    if (local_x < n_lr_apron_quads) {
      int quad_offs = group_width;
      int new_im_indx_off = im_indx_off +
      (im_x + quad_offs < im_width ?
       quad_offs : group_width - local_x - 1);
      vstore4(load_imval(im_indx + new_im_indx_off, image),
              local_quad_offs + quad_offs, local_row_base);
    }
  }
  
  if (local_y < top_apron_height) {
    int im_quad_offs = group_height*im_width;
    int local_quad_offs = group_height*im_cache_quad_width + local_x;
    int im_indx_off =
    (im_y + group_height < im_height ?
     im_quad_offs : (group_height - 1 - local_y)*im_width);
    vstore4(load_imval(im_indx + im_indx_off, image),
            local_quad_offs, local_row_base);
    
    if (local_x - group_width >= -n_lr_apron_quads) {
      int quad_offs = -group_width;
      int new_im_indx_off = im_indx_off +
      (im_x + quad_offs >= 0 ? quad_offs : -im_x);
      vstore4(load_imval(im_indx + new_im_indx_off, image),
              local_quad_offs + quad_offs, local_row_base);
    }
    
    if (local_x < n_lr_apron_quads) {
      int quad_offs = group_width;
      int new_im_indx_off = im_indx_off +
      (im_x + quad_offs < im_width ?
       quad_offs : group_width - local_x - 1);
      vstore4(load_imval(im_indx + new_im_indx_off, image),
              local_quad_offs + quad_offs, local_row_base);
    }
  }
  
  barrier(CLK_LOCAL_MEM_FENCE);
  
  float4 sum = 0.f;
  for (int filt_i = 0; filt_i < num_filt_elems; filt_i++) {
    sparse_value filt_val = filt_cache[filt_i];
    int im_cache_row_base =
      (filt_val.y + local_y)*im_cache_width + apron_padding;
    int im_cache_indx = im_cache_row_base + filt_val.x + local_x;
      
    sum.x += filt_val.val*im_cache[im_cache_indx];
    sum.y += filt_val.val*im_cache[im_cache_indx + group_width];
    sum.z += filt_val.val*im_cache[im_cache_indx + 2*group_width];
    sum.w += filt_val.val*im_cache[im_cache_indx + 3*group_width];
  }
  
  int out_indx_base = 4*(im_y*im_width + group_id_x*group_width) + local_x;
  
  store_imval(sum.x, out_indx_base, output);
  store_imval(sum.y, out_indx_base + group_width, output);
  store_imval(sum.z, out_indx_base + 2*group_width, output);
  store_imval(sum.w, out_indx_base + 3*group_width, output);
}
)
