#define NN_IMPLEMENTATION
#include "nn.h"
#include <stdio.h>
#include <time.h>

int main(void)
{
    srand(time(0));
    Mat a = mat_alloc(2, 3);
    mat_fill(a, 2);
    mat_print(a);

    Mat b = mat_alloc(3, 2);
    mat_fill(b, 5);
    mat_print(b);

    Mat dst = mat_alloc(2, 2);

    printf("--------------\n");
    mat_dot(dst, a, b);
    mat_print(dst);
    return 0;
}