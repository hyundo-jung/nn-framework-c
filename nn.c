#define NN_IMPLEMENTATION
#include "nn.h"
#include <time.h>

typedef struct {
    Mat a0;

    Mat w1, b1, a1;
    Mat w2, b2, a2;

} Xor;

Xor xor_alloc(void)
{
    Xor m;
    m.a0 = mat_alloc(1, 2);

    // first layer of XOR
    m.w1 = mat_alloc(2, 2);
    m.b1 = mat_alloc(1, 2);

    // second layer
    m.w2 = mat_alloc(2, 1);
    m.b2 = mat_alloc(1, 1);

    // activation
    m.a1 = mat_alloc(1, 2);
    m.a2 = mat_alloc(1, 1);
    return m;
}

float forward_xor(Xor m)
{
    mat_dot(m.a1, m.a0, m.w1);
    mat_sum(m.a1, m.b1);
    mat_sig(m.a1);

    mat_dot(m.a2, m.a1, m.w2);
    mat_sum(m.a2, m.b2);
    mat_sig(m.a2);

    return *m.a2.es;
}

float cost(Xor m, Mat ti, Mat to)
{
    assert(ti.rows == to.rows);
    assert(to.cols == m.a2.cols);
    size_t n = ti.rows;
    size_t q = to.cols;

    float c = 0.0f;
    for (size_t i = 0; i < n; i++)
    {
        // MAT_AT(m.a0, 0, 0) = MAT_AT(ti, n, 0);
        // MAT_AT(m.a0, 0, 0) = MAT_AT(ti, n, 1);
        Mat x = mat_row(ti, i);
        Mat y = mat_row(to, i);

        mat_copy(m.a0, x);
        forward_xor(m);

        for (size_t j = 0; j < q; j++)
        {
            float d = MAT_AT(m.a2, i, j) - MAT_AT(y, i, j);
            c += d*d;
        }
    }
    c /= n;
    return c;
}

float td[][3] = {
    {1, 2, 3},
    {4, 5, 6},
    {7, 8, 9},
    {10, 11, 12},
};

void finite_diff(Xor m, Xor g, float eps, Mat ti, Mat to)
{
    float saved;
    float c = cost(m, ti, to);

    // w1 b1
    for (size_t i = 0; i < m.w1.rows; i++)
    {
        for (size_t j = 0; j < m.w1.cols; j++)
        {
            saved = MAT_AT(m.w1, i, j);
            MAT_AT(m.w1, i, j) += eps;
            MAT_AT(g.w1, i, j) = (cost(m, ti, to) - c)/ eps;
            MAT_AT(m.w1, i, j) = saved;
        }
    }

    for (size_t i = 0; i < m.b1.rows; i++)
    {
        for (size_t j = 0; j < m.b1.cols; j++)
        {
            saved = MAT_AT(m.b1, i, j);
            MAT_AT(m.b1, i, j) += eps;
            MAT_AT(g.b1, i, j) = (cost(m, ti, to) - c)/ eps;
            MAT_AT(m.b1, i, j) = saved;
        }
    }

    // w2 b2
    for (size_t i = 0; i < m.w2.rows; i++)
    {
        for (size_t j = 0; j < m.w2.cols; j++)
        {
            saved = MAT_AT(m.w1, i, j);
            MAT_AT(m.w2, i, j) += eps;
            MAT_AT(g.w2, i, j) = (cost(m, ti, to) - c)/ eps;
            MAT_AT(m.w2, i, j) = saved;
        }
    }

    for (size_t i = 0; i < m.b2.rows; i++)
    {
        for (size_t j = 0; j < m.b2.cols; j++)
        {
            saved = MAT_AT(m.b2, i, j);
            MAT_AT(m.b2, i, j) += eps;
            MAT_AT(g.b2, i, j) = (cost(m, ti, to) - c)/ eps;
            MAT_AT(m.b2, i, j) = saved;
        }
    }
}

void xor_learn(Xor m, Xor g, float rate)
{
        // w1 b1
        for (size_t i = 0; i < m.w1.rows; i++)
        {
            for (size_t j = 0; j < m.w1.cols; j++)
            {
                MAT_AT(m.w1, i, j) -= rate*MAT_AT(g.w1, i, j);
            }
        }
    
        for (size_t i = 0; i < m.b1.rows; i++)
        {
            for (size_t j = 0; j < m.b1.cols; j++)
            {
                MAT_AT(m.b1, i, j) -= rate*MAT_AT(g.b1, i, j);
            }
        }
    
        // w2 b2
        for (size_t i = 0; i < m.w2.rows; i++)
        {
            for (size_t j = 0; j < m.w2.cols; j++)
            {
                MAT_AT(m.w2, i, j) -= rate*MAT_AT(g.w2, i, j);
            }
        }
    
        for (size_t i = 0; i < m.b2.rows; i++)
        {
            for (size_t j = 0; j < m.b2.cols; j++)
            {
                MAT_AT(m.b2, i, j) -= rate*MAT_AT(g.b2, i, j);
            }
        }
}

int main(void)
{
    srand(time(0));

    size_t stride = 3;
    size_t n = sizeof(td) / sizeof(td[0]);
    Mat ti = {
        .rows = n,
        .cols = 2,
        .stride = stride,
        .es = td[0],
    };

    Mat to = {
        .rows = n,
        .cols = 1,
        .stride = stride,
        .es = td[0] + 2,
    };

    Xor m = xor_alloc();
    Xor g = xor_alloc();

    mat_rand(m.w1, 0, 1);
    mat_rand(m.b1, 0, 1);
    mat_rand(m.w2, 0, 1);
    mat_rand(m.b2, 0, 1);  

    float eps = 1e-1;
    float rate = 1e-1;

    printf("cost: %f \n", cost(m, ti, to));
    finite_diff(m, g, eps, ti, to);
    xor_learn(m, g, rate);
    printf("cost: %f \n", cost(m, ti, to));


    #if 0
    for (size_t i = 0; i < 2; i++)
    {
        for (size_t j = 0; j < 2; j++)
        {
            MAT_AT(m.a0, 0, 0) = i;
            MAT_AT(m.a0, 0, 1) = j;
            forward_xor(m);
            float y = *(m.a2.es);

            printf("%zu ^ %zu = %f\n", i, j, y);
        }
    }
    #endif
  
    return 0;
}