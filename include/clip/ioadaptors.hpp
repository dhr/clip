#pragma once
#ifndef CLIP_IOADAPTORS_H
#define CLIP_IOADAPTORS_H

#include <list>

#include "clip/basictypes.hpp"
#include "clip/imagebuffer.hpp"

namespace clip {

typedef std::list<ImageBuffer> ImBufList;

class InputAdaptor {
 public:
  virtual ImageBuffer peek() = 0;
  virtual ImageBuffer next() = 0;
};

class OutputAdaptor {
  public:
   virtual void output(ImageBuffer buf) = 0;
};

template<typename InputIterator>
class InputIteratorAdaptor : public InputAdaptor {
  InputIterator it_;
  
 public:
  explicit InputIteratorAdaptor(InputIterator it) : it_(it) {}
  
  inline ImageBuffer peek() { return *it_; }
  inline ImageBuffer next() { return *it_++; }
};

template<typename T>
InputIteratorAdaptor<T> MakeInIterAdaptor(T iter) {
  return InputIteratorAdaptor<T>(iter);
}

template<typename OutputIterator>
class OutputIteratorAdaptor : public OutputAdaptor {
  OutputIterator it_;
  
 public:
  explicit OutputIteratorAdaptor(OutputIterator it) : it_(it) {}
  
  inline void output(ImageBuffer buf) { *it_ = buf; ++it_; }
};

template<typename T>
OutputIteratorAdaptor<T> MakeOutIterAdaptor(T iter) {
  return OutputIteratorAdaptor<T>(iter);
}

class PopAdaptor : public InputAdaptor {
  ImBufList* list_;
  
 public:
  explicit PopAdaptor(ImBufList& list) : list_(&list) {}
  
  inline ImageBuffer peek() { return list_->back(); }
  
  inline ImageBuffer next() {
    ImageBuffer back = list_->back();
    list_->pop_back();
    return back;
  }
};

class PushAdaptor : public OutputAdaptor {
  ImBufList* list_;
  
 public:
  explicit PushAdaptor(ImBufList& list) : list_(&list) {}
  
  inline void output(ImageBuffer buf) {
    list_->push_back(buf);
  }
};

class CircularAdaptor : public InputAdaptor {
  ImBufList* list_;
  ImBufList::reverse_iterator it_;
  
 public:
  explicit CircularAdaptor(ImBufList& list)
  : list_(&list), it_(list_->rbegin()) {}
  
  inline ImageBuffer peek() { return *it_; }
  
  inline ImageBuffer next() {
    ImageBuffer temp = *it_;
    advance(1);
    return temp;
  }
  
  void advance(i32 n) {
    if (n > 0) {
      while (n-- > 0) {
        it_++;
        if (it_ == list_->rend()) it_ = list_->rbegin();
      }
    }
    else {
      while (n++ < 0) {
        if (it_ == list_->rbegin()) it_ = list_->rend();
        it_--;
      }
    }
  }
  
  inline void reset() { it_ = list_->rbegin(); }
};

inline void PopulateListFromInput(InputAdaptor& input, i32 n, ImBufList& list,
                                  bool pushBack = false) {
  if (pushBack) {
    for (i32 i = 0; i < n; i++) {
      list.push_back(input.next());
    }
  }
  else {
    for (i32 i = 0; i < n; i++) {
      list.push_front(input.next());
    }
  }
}

}

#endif
