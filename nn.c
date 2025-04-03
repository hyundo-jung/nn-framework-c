#define NN_IMPLEMENTATION
#include "nn.h"
#include <stdio.h>

int main(void)
{
    Mat m = mat_alloc(3, 1);
    mat_print(m);
    return 0;
}