#ifndef NN_H_
#define NN_H_

#include <stddef.h>
#include <stdio.h>
#include <math.h>

#ifndef NN_MALLOC
#include <stdlib.h>
#define NN_MALLOC malloc
#endif // NN_MALLOC

#ifndef NN_ASSERT
#include <assert.h>
#define NN_ASSERT assert
#endif // NN_ASSERT

#define ARRAY_LEN(xs) sizeof((xs)) / sizeof((xs)[0])

float rand_float(void);
float sigmoidf(float x);

typedef struct {
    size_t rows;
    size_t cols;
    size_t stride;
    float* es;
} Mat;

#define MAT_AT(m, i, j) m.es[(i)*m.stride + (j)]

Mat mat_alloc(size_t rows, size_t cols);
void mat_fill(Mat m, float x);
void mat_rand(Mat m, float low, float high);
Mat mat_row(Mat m, size_t row);
void mat_copy(Mat dst, Mat src);
void mat_dot(Mat dst, Mat a, Mat b);
void mat_sum(Mat dst, Mat a);
void mat_sig(Mat m);
void mat_appendCol(Mat dst, Mat src, Mat a);
void mat_appendRow(Mat dst, Mat src, Mat a);
void mat_print(Mat m, const char* name, size_t padding);

#define MAT_PRINT(m) mat_print(m, #m, 0)

typedef struct {
    size_t count; // # of inner layers
    Mat *ws;
    Mat *bs;
    Mat *as; // the amount of activation is count + 1
} NN;

#define NN_INPUT(nn) (nn).as[0]
#define NN_OUTPUT(nn) (nn).as[(nn).count]

NN nn_alloc(size_t *arch, size_t arch_count);
void nn_print(NN nn, const char* name);
#define NN_PRINT(nn) nn_print(nn, #nn);
void nn_rand(NN nn, float low, float high);
void nn_forward(NN nn);
float nn_cost(NN nn, Mat in, Mat out);
void nn_finite_diff(NN nn, NN g, float eps, Mat in, Mat out);
void nn_learn(NN nn, NN g, float rate);

#endif //NN_H_

#ifdef NN_IMPLEMENTATION

float rand_float(void)
{
    return (float) rand() / (float) RAND_MAX;
}

float sigmoidf(float x)
{
    return 1.f / (1.f + expf(-x));
}

Mat mat_alloc(size_t rows, size_t cols)
{
    Mat m;
    m.rows = rows;
    m.cols = cols;
    m.stride = cols;
    m.es = NN_MALLOC(sizeof(*m.es) * rows * cols);
    NN_ASSERT(m.es != NULL);
    return m;
}

void mat_fill(Mat m, float x)
{
    for (size_t i = 0; i < m.rows; i++)
    {
        for (size_t j = 0; j < m.cols; j++)
        {
            MAT_AT(m, i, j) = x;
        }
    }
}

void mat_rand(Mat m, float low, float high)
{
    for (size_t i = 0; i < m.rows; i++)
    {
        for (size_t j = 0; j < m.cols; j++)
        {
            MAT_AT(m, i, j) = rand_float()*(high - low) + low;
        }
    }
}

Mat mat_row(Mat m, size_t row)
{
    return (Mat) {
        .rows = 1,
        .cols = m.cols,
        .stride = m.stride,
        .es = &MAT_AT(m, row, 0), // beginning of specific row
    };
}

void mat_copy(Mat dst, Mat src)
{
    NN_ASSERT(dst.rows == src.rows);
    NN_ASSERT(dst.cols == src.cols);

    for (size_t i = 0; i < src.rows; i++)
    {
        for (size_t j = 0; j < src.cols; j++)
        {
            MAT_AT(dst, i, j) = MAT_AT(src, i, j);
        }
    }
}

void mat_dot(Mat dst, Mat a, Mat b)
{
    NN_ASSERT(a.cols == b.rows); // inner matrix
    size_t n = a.cols;
    NN_ASSERT(dst.rows == a.rows); // outer matrix
    NN_ASSERT(dst.cols == b.cols);

    for (size_t i = 0; i < dst.rows; i++)
    {
        for (size_t j = 0; j < dst.cols; j++)
        {
            MAT_AT(dst, i, j) = 0;
            for (size_t k = 0; k < n; k++)
            {
                MAT_AT(dst, i, j) += MAT_AT(a, i, k) * MAT_AT(b, k, j);    
            }  
        }
    }

    return;
}

void mat_sum(Mat dst, Mat a)
{   
    NN_ASSERT(dst.rows == a.rows);
    NN_ASSERT(dst.cols == a.cols);

    for (size_t i = 0; i < dst.rows; i++)
    {
        for (size_t j = 0; j < dst.cols; j++)
        {
            MAT_AT(dst, i, j) += MAT_AT(a, i, j);
        }
    }
    return;
}

void mat_sig(Mat m)
{
    for (size_t i = 0; i < m.rows; i++)
    {
        for (size_t j = 0; j < m.cols; j++)
        {
            MAT_AT(m, i, j) = sigmoidf(MAT_AT(m, i, j));
        }
    }
}

void mat_appendCol(Mat dst, Mat src, Mat a)
{
    NN_ASSERT(dst.rows == src.rows);
    NN_ASSERT(dst.cols == src.cols + 1);

    NN_ASSERT(a.rows == dst.rows);
    NN_ASSERT(a.cols == 1);

    for (size_t i = 0; i < dst.rows; i++)
    {
        for (size_t j = 0; j < dst.cols - 1; j++)
        {
            MAT_AT(dst, i, j) = MAT_AT(src, i, j);
        }

        MAT_AT(dst, i, dst.cols - 1) = MAT_AT(a, i, 0);
    }
}

void mat_appendRow(Mat dst, Mat src, Mat a)
{
    NN_ASSERT(dst.cols == src.cols);
    NN_ASSERT(dst.rows == src.rows + 1);

    NN_ASSERT(a.cols == dst.cols);
    NN_ASSERT(a.rows == 1);

    for (size_t j = 0; j < dst.cols; j++)
    {
        for (size_t i = 0; i < dst.rows - 1; i++)
        {
            MAT_AT(dst, i, j) = MAT_AT(src, i, j);
        }

        MAT_AT(dst, dst.rows - 1, j) = MAT_AT(a, 0, j);
    }

}

void mat_print(Mat m, const char* name, size_t padding)
{
    printf("%*s%s = [\n", (int)padding, "", name);
    for (size_t i = 0; i < m.rows; i++)
    {
        printf("%*s    ", (int)padding, "");
        for (size_t j = 0; j < m.cols; j++)
        {
            printf("%f ", MAT_AT(m, i, j));
        }
        printf("\n");
    }
    printf("%*s]\n", (int)padding, "");
}

NN nn_alloc(size_t *arch, size_t arch_count)
{
    NN_ASSERT(arch_count > 0);

    NN nn;
    nn.count = arch_count - 1;

    nn.ws = NN_MALLOC(sizeof(*nn.ws) * nn.count);
    NN_ASSERT(nn.ws != NULL); 

    nn.bs = NN_MALLOC(sizeof(*nn.bs) * nn.count);
    NN_ASSERT(nn.bs != NULL);

    nn.as = NN_MALLOC(sizeof(*nn.as) * (nn.count + 1));
    NN_ASSERT(nn.as != NULL); 

    nn.as[0] = mat_alloc(1, arch[0]);

    for (size_t i = 1; i < arch_count; i++)
    {
        nn.ws[i - 1] = mat_alloc(arch[i - 1], arch[i]);
        nn.bs[i - 1] = mat_alloc(1, arch[i]);
        nn.as[i] = mat_alloc(1, arch[i]);
    }

    return nn;
}

void nn_print(NN nn, const char* name)
{
    char buf[256];

    printf("%s = [\n", name);
    for (size_t i = 0; i < nn.count; i++)
    {
        snprintf(buf, sizeof(buf), "ws%zu", i);
        mat_print(nn.ws[i], buf, 4);
        snprintf(buf, sizeof(buf), "bs%zu", i);
        mat_print(nn.bs[i], buf, 4);
    }
    printf("]\n");
}

void nn_rand(NN nn, float low, float high)
{
    for (size_t i = 0; i < nn.count; i++)
    {
        mat_rand(nn.ws[i], low, high);
        mat_rand(nn.bs[i], low, high);
    }
}

void nn_forward(NN nn)
{
    for (size_t i = 0; i < nn.count; i++)
    {
        Mat as_temp = mat_alloc(nn.as[i].rows, nn.as[i].cols + 1);
        Mat ws_temp = mat_alloc(nn.ws[i].rows + 1, nn.ws[i].cols);
        // concern: mallocating matrices internally that user has no control

        Mat ones = mat_alloc(nn.as[i].rows, 1);
        for (size_t i = 0; i < nn.as[i].rows; i++)
            MAT_AT(ones, i, 0) = 1;

        mat_appendCol(as_temp, nn.as[i], ones);
        mat_appendRow(ws_temp, nn.ws[i], nn.bs[i]);

        mat_dot(nn.as[i + 1], as_temp, ws_temp);
        mat_sig(nn.as[i + 1]);
    }
}

float nn_cost(NN nn, Mat in, Mat out)
{
    NN_ASSERT(in.rows == out.rows);
    NN_ASSERT(out.cols == NN_OUTPUT(nn).cols);
    size_t n = in.rows;
    size_t m = out.cols;

    float c = 0.0f;
    for (size_t i = 0; i < n; i++)
    {
        Mat x = mat_row(in, i);
        Mat y = mat_row(out, i);

        mat_copy(NN_INPUT(nn), x);
        nn_forward(nn);

        for (size_t j = 0; j < m; j++)
        {
            float d = MAT_AT(NN_OUTPUT(nn), 0, j) - MAT_AT(y, 0, j);
            c += d*d;
        }
    }
    c /= n;
    return c;
}

void nn_finite_diff(NN nn, NN g, float eps, Mat in, Mat out)
{
    float saved;
    float c = nn_cost(nn, in, out);

    for (size_t i = 0; i < nn.count; i++)
    {
        for (size_t j = 0; j < nn.ws[i].rows; j++)
        {
            for (size_t k = 0; k < nn.ws[i].cols; k++)
            {
                saved = MAT_AT(nn.ws[i], j, k);
                MAT_AT(nn.ws[i], j, k) += eps;
                MAT_AT(g.ws[i], j, k) = (nn_cost(nn, in, out) - c) / eps;
                MAT_AT(nn.ws[i], j, k) = saved;
            }
        }

        for (size_t j = 0; j < nn.bs[i].rows; j++)
        {
            for (size_t k = 0; k < nn.bs[i].cols; k++)
            {
                saved = MAT_AT(nn.bs[i], j, k);
                MAT_AT(nn.bs[i], j, k) += eps;
                MAT_AT(g.bs[i], j, k) = (nn_cost(nn, in, out) - c) / eps;
                MAT_AT(nn.bs[i], j, k) = saved;
            }
        }
    }
}

void nn_learn(NN nn, NN g, float rate)
{
    for (size_t i = 0; i < nn.count; i++)
    {
        for (size_t j = 0; j < nn.ws[i].rows; j++)
        {
            for (size_t k = 0; k < nn.ws[i].cols; k++)
            {
                MAT_AT(nn.ws[i], j, k) -= rate*MAT_AT(g.ws[i], j, k);
            }
        }

        for (size_t j = 0; j < nn.bs[i].rows; j++)
        {
            for (size_t k = 0; k < nn.bs[i].cols; k++)
            {
                MAT_AT(nn.bs[i], j, k) -= rate*MAT_AT(g.bs[i], j, k);
            }
        }
    }
}



#endif // NN_IMPLEMENTATION