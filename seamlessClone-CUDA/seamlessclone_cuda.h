#include "opencv2/core.hpp"
#include "seamlessClone_imp.cu"

extern "C"
{
Mat my_seamlessclone_api_imp( void* face, void* body, void* mask, int centerX, int centerY, int gpu_id )
{
  //int argc = 7;
  //const char* argv[7] = 
  //{
  //  "./seamlessClone",
  //  "./images/src.yml",
  //  "./images/dst.yml",
  //  "./images/src_mask.yml",
  //  "800",
  //  "150",
  //  "1"
  //};
  Mat retMat = seamlessClone_imp(face, body, mask, centerX, centerY, gpu_id);
  return retMat;
}

}
    