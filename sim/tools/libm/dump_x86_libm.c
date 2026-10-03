// Dump the x86_64 libsystem_m sinf code (code.bin) and its constant area (data.bin, sinf+0x45b00..).
// Build/run: clang -arch x86_64 dump_x86_libm.c -o d && ./d   (runs under Rosetta)

#include <math.h>
#include <stdio.h>
#include <string.h>
#include <dlfcn.h>
int main(){ unsigned char*p=(unsigned char*)dlsym(RTLD_DEFAULT,"sinf");
 FILE*f=fopen("code.bin","wb"); fwrite(p,1,0x400,f); fclose(f);
 f=fopen("data.bin","wb"); fwrite(p+0x45b00,1,0x4a400-0x45b00,f); fclose(f); }
