#pragma once
#ifndef CLIP_DETAIL_BUFFERCACHE_H
#define CLIP_DETAIL_BUFFERCACHE_H

#include <map>

#include "clip/detail/imagebufferimpl.hpp"

namespace clip {
namespace detail {

typedef std::pair< std::pair<ImageBufferType, ValueType>,
                   std::pair<i32, i32> > BufferCacheKey;

inline BufferCacheKey MakeCacheKey(ImageBufferType type,
                                   ValueType valType,
                                   i32 width,
                                   i32 height) {
  return std::make_pair(std::make_pair(type, valType),
                        std::make_pair(width, height));
}

inline BufferCacheKey MakeCacheKey(ImageBufferImpl *impl) {
  return MakeCacheKey(impl->type(), impl->valType(),
                      impl->paddedWidth(), impl->paddedHeight());
}

class BufferCache {
  static inline std::multimap<BufferCacheKey, ImageBufferImplPtr>& cache() {
    static std::multimap<BufferCacheKey, ImageBufferImplPtr> cache;
    return cache;
  }
  
  static inline bool& cacheValid() {
    static bool valid = false;
    return valid;
  }
  
 public:
  BufferCache() { cacheValid() = true; }
  ~BufferCache() { cacheValid() = false; }

  static inline BufferCache& instance() {
    static BufferCache instance;
    return instance;
  }

  inline ImageBufferImplPtr retrieve(const BufferCacheKey& key) {
    std::multimap<BufferCacheKey, ImageBufferImplPtr>::iterator it =
      cache().find(key);
    if (it == cache().end())
      return ImageBufferImplPtr();
    ImageBufferImplPtr found = it->second;
    cache().erase(it);
    return found;
  }
  
  inline ImageBufferImplPtr retrieve(ImageBufferType type,
                                     ValueType valType,
                                     i32 width,
                                     i32 height,
                                     i32 xAlign,
                                     i32 yAlign) {
    i32 paddedWidth, paddedHeight;
    CalcPaddedSizes(width, height, xAlign, yAlign, &paddedWidth, &paddedHeight);
    BufferCacheKey key = MakeCacheKey(type, valType, paddedWidth, paddedHeight);
    ImageBufferImplPtr p = retrieve(key);
    
    if (!p.get()) return p;
    
    // We're a friend, so we can perform (necessary) surgery
    p->width_ = width;
    p->height_ = height;
    p->xAlign_ = xAlign;
    p->yAlign_ = yAlign;
    p->paddedWidth_ = paddedWidth;
    p->paddedHeight_ = paddedHeight;
    return p;
  }
  
  inline ImageBufferImplPtr retrieve(ImageBufferImpl *impl) {
    return retrieve(MakeCacheKey(impl));
  }
  
  inline void release(const BufferCacheKey& key,
                      ImageBufferImplPtr& ptr) {
    if (cacheValid())
      cache().insert(std::make_pair(key, ptr));
  }
  
  inline void release(ImageBufferImplPtr& ptr) {
    release(MakeCacheKey(ptr.get()), ptr);
  }
  
  inline void clear() {
    cache().clear();
  }
};

class BufferCacheInitializer {
  static void init() {
    BufferCache::instance().clear();
  }
  
 public:
  BufferCacheInitializer() {
    static bool hasRun = false;
    if (!hasRun && (hasRun = true))
      AddInitClient(init);
  }
};

static BufferCacheInitializer bufferCacheInitializer;

}}

#endif
