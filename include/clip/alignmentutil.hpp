#pragma once
#ifndef CLIP_ALIGNMENTUTIL_H
#define CLIP_ALIGNMENTUTIL_H

#include <cstring>

namespace clip {

inline void CalcAlignedSizes(i32 width, i32 height,
                             i32 xAlign, i32 yAlign,
                             i32* paddedWidth, i32* paddedHeight) {
  i32 xRem = width%xAlign;
  i32 yRem = height%yAlign;
  *paddedWidth = width + (xAlign - xRem)*(xRem != 0);
  *paddedHeight = height + (yAlign - yRem)*(yRem != 0);
}
  
inline void PadData(const f32 *data, i32 width, i32 height,
                    i32 leftPad, i32 rightPad,
                    i32 bottomPad, i32 topPad,
                    f32 padVal, f32* padDest) {
  assert(padDest != data && "You can't pad data in place");
  
  i32 paddedWidth = width + leftPad + rightPad;
  i32 paddedHeight = height + bottomPad + topPad;
  assert(paddedWidth >= 0 && paddedHeight >= 0 && "Invalid padding");
  
  i32 srcStartX = -std::min(leftPad, 0);
  i32 srcStartY = -std::min(bottomPad, 0);
  i32 dstStartX = std::max(0, leftPad);
  i32 dstStartY = std::max(0, bottomPad);
  i32 copyWidth = std::min(width, paddedWidth);
  i32 copyHeight = std::min(height, paddedHeight);
  
  for (i32 y = 0; y < copyHeight; y++) {
    memcpy(padDest + (y + dstStartY)*paddedWidth + dstStartX,
           data + (y + srcStartY)*width + srcStartX,
           copyWidth*sizeof(f32));
  }
}

}

#endif