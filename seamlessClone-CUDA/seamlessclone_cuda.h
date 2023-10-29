#include "opencv2/core.hpp"
#include "seamlessClone_imp.cu"

extern "C"
{
Mat my_seamlessclone_api_imp_run( void* instance_ptr, void* face, void* body, void* mask, int centerX, int centerY, int gpu_id, bool bSync=true )
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
  Mat retMat = seamlessClone_imp_run(instance_ptr, face, body, mask, centerX, centerY, gpu_id, bSync);
  return retMat;
}

void* my_seamlessclone_api_imp_create_instance( int gpu_id )
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
  void* instance_ptr = seamlessClone_imp_create_instance(gpu_id);
  return instance_ptr;
}

void my_seamlessclone_api_imp_destroy( void* instance_ptr )
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
  seamlessClone_imp_destroy( instance_ptr );
  return;
}

void my_seamlessclone_api_imp_sync( void* instance_ptr )
{
  seamlessClone_imp_sync( instance_ptr );
  return;
}

}
    