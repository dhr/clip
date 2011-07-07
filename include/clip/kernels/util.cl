CLIP_STRINGIFY(
int get_global_index(void);

int get_global_index(void) {
  int x = get_global_id(0);
	int y = get_global_id(1);
  int width = get_global_size(0);
  return y*width + x;
}
)
