#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define DEBUG 1
#define MIN(x, y) ((x) > (y) ? (y) : (x))
#define MAX(x, y) ((x) > (y) ? (y) : (x))

typedef struct {
   int width;
   int height;
   int comp;
   int size;
   float *data;
} image_t;

typedef struct {
   image_t *images;
   int levels;
   int type;
} pyramid_t;

enum { G_PYRAMID, L_PYRAMID };

static const char *inputs_path[] = {
   "images/A.jpg",
   "images/B.jpg",
   "images/C.jpg",
   "images/D.jpg",
};
static const int inputs_length = 4;

static const float L[][3] = {
   {0.0f,  1.0f, 0.0f},
   {1.0f, -4.0f, 1.0f},
   {0.0f,  1.0f, 0.0f},
};

static const float G[][5] = {
   {0.00390625f, 0.015625f, 0.0234375f, 0.015625f, 0.00390625f},
   {0.015625f,   0.0625f,   0.09375f,   0.0625f,   0.015625f},
   {0.0234375f,  0.09375f,  0.140625f,  0.09375f,  0.0234375f},
   {0.015625f,   0.0625f,   0.09375f,   0.0625f,   0.015625f},
   {0.00390625f, 0.015625f, 0.0234375f, 0.015625f, 0.00390625f}
};

static void load_image(image_t *in, const char *filename)
{
   int w, h, c;
   unsigned char *data = stbi_load(filename, &w, &h, &c, 3);
   in->width = w;
   in->height = h;
   in->comp = c;
   in->size = w * h * c;
   in->data = (float *)malloc(sizeof(float) * in->size);
   for (int i = 0; i < in->size; ++i)
      in->data[i] = data[i] / 255.0f;
   stbi_image_free(data);
}

static void cvt_grayscale(image_t *out, image_t *in)
{
   out->width = in->width;
   out->height = in->height;
   out->comp = 1;
   out->size = out->width * out->height;
   out->data = (float *)malloc(sizeof(float) * out->size);
   for (int i = 0; i < out->size; ++i) {
      float r = in->data[i * in->comp + 0];
      float g = in->data[i * in->comp + 1];
      float b = in->data[i * in->comp + 2];
      out->data[i] = 0.299f * r + 0.587f * g + 0.114f * b;
   }
}

static void empty_image(image_t *in, int width, int height, int comp)
{
   in->width = width;
   in->height = height;
   in->comp = comp;
   in->size = width * height * comp;
   in->data = (float *)calloc(in->size, sizeof(float));
}

static void copy_image(image_t *out, image_t *in)
{
   out->width = in->width;
   out->height = in->height;
   out->comp = in->comp;
   out->size = in->size;
   out->data = (float *)malloc(sizeof(float) * out->size);
   memcpy(out->data, in->data, sizeof(float) * out->size);
}

static void convolve(float *out, image_t *in, int off, int dim,
      const float kern[][dim])
{
   int half = dim / 2;
   for (int i = 0; i < in->height; ++i) {
      for (int j = 0; j < in->width; ++j) {
         float sum = 0.0f;
         for (int ii = -half; ii <= half; ++ii) {
            for (int jj = -half; jj <= half; ++jj) {
               int ni = i + ii;
               int nj = j + jj;
               ni = ni < 0 ? 0 : ni >= in->height ? in->height - 1 : ni;
               nj = nj < 0 ? 0 : nj >= in->width ? in->width - 1 : nj;
               sum += in->data[(ni * in->width + nj) * in->comp + off]
                  * kern[ii + half][jj + half];
            }
         }
         out[(i * in->width + j) * in->comp + off] = sum;
      }
   }
}

static float std_deviation(float *data, int length)
{
   float u = 0.0f;
   for (int i = 0; i < length; ++i)
      u += data[i];
   u /= length;
   float v = 0.0f;
   for (int i = 0; i < length; ++i)
      v += (data[i] - u) * (data[i] - u);
   return sqrtf(v / length);
}

static float gaussian(float in, float mean, float spread)
{
   /* unnormalized gaussian function */
   return expf(-0.5f * (((in - mean) * (in - mean)) / (spread * spread)));
}

static void calculate_weight(float *out, image_t *in, image_t *gray)
{
   int size = gray->size;
   float *C = malloc(sizeof(float) * size);
   convolve(C, gray, 0, 3, L);
   for (int i = 0; i < size; ++i)
      C[i] = fabsf(C[i]);
   /* std deviation */
   float *S = malloc(sizeof(float) * size);
   for (int i = 0; i < size; ++i)
      S[i] = std_deviation(in->data + (i * in->comp), 3);
   /* gassian distribution */
   float *E = malloc(sizeof(float) * size);
   float spread = 0.2f;
   for (int i = 0; i < size; ++i) {
      float r = in->data[i * in->comp + 0];
      float g = in->data[i * in->comp + 1];
      float b = in->data[i * in->comp + 2];
      E[i] = gaussian(r, 0.5f, spread) * gaussian(g, 0.5f, spread) * gaussian(b, 0.5f, spread);
   }
   float wc = 1.0f;
   float ws = 1.0f;
   float we = 1.0f;
   for (int i = 0; i < size; ++i)
      out[i] = C[i] * S[i] * E[i];
   free(C);
   free(S);
   free(E);
}

static void normalize_weights(float *weights[], int size, int length)
{
   for (int i = 0; i < size; ++i) {
      float sum = 0.0f;
      for (int j = 0; j < length; ++j)
         sum += weights[j][i];
      if (sum == 0.0f)
         sum = 1.0f;
      for (int j = 0; j < length; ++j)
         weights[j][i] /= sum;
   }
}

static void downsample(image_t *out, image_t *in)
{
   empty_image(out, in->width / 2, in->height / 2, in->comp);
   float *gauss = malloc(sizeof(float) * in->size);
   convolve(gauss, in, 0, 5, G);
   convolve(gauss, in, 1, 5, G);
   convolve(gauss, in, 2, 5, G);
   for (int y = 0; y < out->height; ++y) {
      for (int x = 0; x < out->width; ++x) {
         int uy = 2 * y + 1;
         int ux = 2 * x + 1;
         out->data[(y * out->width + x) * out->comp + 0] = gauss[(uy * in->width + ux) * in->comp + 0];
         out->data[(y * out->width + x) * out->comp + 1] = gauss[(uy * in->width + ux) * in->comp + 1];
         out->data[(y * out->width + x) * out->comp + 2] = gauss[(uy * in->width + ux) * in->comp + 2];
      }
   }
   free(gauss);
}

static void upsample(image_t *out, image_t *in, int odd_w, int odd_h)
{
   image_t u_pad;
   empty_image(&u_pad, in->width + 2, in->height + 2, in->comp);
   /* replicate first and last lines */
   for (int x = 0; x < in->width; ++x) {
      u_pad.data[(x + 1) * u_pad.comp + 0] = in->data[x * in->comp + 0];
      u_pad.data[(x + 1) * u_pad.comp + 1] = in->data[x * in->comp + 1];
      u_pad.data[(x + 1) * u_pad.comp + 2] = in->data[x * in->comp + 2];

      int last_uy = (u_pad.height - 1) * u_pad.width;
      int last_cy = (in->height - 1) * in->width;
      u_pad.data[(last_uy + x + 1) * u_pad.comp + 0] = in->data[(last_cy + x) * in->comp + 0];
      u_pad.data[(last_uy + x + 1) * u_pad.comp + 1] = in->data[(last_cy + x) * in->comp + 1];
      u_pad.data[(last_uy + x + 1) * u_pad.comp + 2] = in->data[(last_cy + x) * in->comp + 2];
   }
   /* fill the data */
   for (int y = 0; y < in->height; ++y) {
      for (int x = 0; x < in->width; ++x) {
         int i = (y + 1) * u_pad.width + (x + 1);
         int j = y * in->width + x;
         u_pad.data[i * u_pad.comp + 0] = in->data[j * in->comp + 0];
         u_pad.data[i * u_pad.comp + 1] = in->data[j * in->comp + 1];
         u_pad.data[i * u_pad.comp + 2] = in->data[j * in->comp + 2];
      }
   }
   /* replicate the first and last column */
   for (int y = 0; y < u_pad.height; ++y) {
      u_pad.data[(y * u_pad.width * u_pad.comp) + 0] = u_pad.data[((y * u_pad.width + 1) * u_pad.comp) + 0];
      u_pad.data[(y * u_pad.width * u_pad.comp) + 1] = u_pad.data[((y * u_pad.width + 1) * u_pad.comp) + 1];
      u_pad.data[(y * u_pad.width * u_pad.comp) + 2] = u_pad.data[((y * u_pad.width + 1) * u_pad.comp) + 2];

      int last_w = u_pad.width - 1;
      u_pad.data[((y * u_pad.width + last_w) * u_pad.comp) + 0] = u_pad.data[((y * u_pad.width + last_w - 1) * u_pad.comp) + 0];
      u_pad.data[((y * u_pad.width + last_w) * u_pad.comp) + 1] = u_pad.data[((y * u_pad.width + last_w - 1) * u_pad.comp) + 1];
      u_pad.data[((y * u_pad.width + last_w) * u_pad.comp) + 2] = u_pad.data[((y * u_pad.width + last_w - 1) * u_pad.comp) + 2];
   }
   image_t u_inter;
   empty_image(&u_inter, in->width * 2 + 4, in->height * 2 + 4, in->comp);
   for (int y = 0; y < u_pad.height; ++y) {
      for (int x = 0; x < u_pad.width; ++x) {
         int ny = 2 * y + 1;
         int nx = 2 * x + 1;
         int i = y * u_pad.width + x;
         int j = ny * u_inter.width + nx;
         u_inter.data[j * u_inter.comp + 0] = 4 * u_pad.data[i * u_pad.comp + 0];
         u_inter.data[j * u_inter.comp + 1] = 4 * u_pad.data[i * u_pad.comp + 1];
         u_inter.data[j * u_inter.comp + 2] = 4 * u_pad.data[i * u_pad.comp + 2];
      }
   }
   free(u_pad.data);
   image_t u;
   empty_image(&u, u_inter.width, u_inter.height, u_inter.comp);
   convolve(u.data, &u_inter, 0, 5, G);
   convolve(u.data, &u_inter, 1, 5, G);
   convolve(u.data, &u_inter, 2, 5, G);
   free(u_inter.data);
   empty_image(out, u.width - 4 + odd_w, u.height - 4 + odd_h, u.comp);
   for (int y = 2; y < u.height && y - 2 < out->height; ++y) {
      for (int x = 2; x < u.width && x - 2 < out->width; ++x) {
         out->data[((y - 2) * out->width + (x - 2)) * out->comp + 0] = u.data[(y * u.width + x ) * u.comp + 0];
         out->data[((y - 2) * out->width + (x - 2)) * out->comp + 1] = u.data[(y * u.width + x ) * u.comp + 1];
         out->data[((y - 2) * out->width + (x - 2)) * out->comp + 2] = u.data[(y * u.width + x ) * u.comp + 2];
      }
   }
   free(u.data);
}

static void build_pyramid(pyramid_t **out, image_t *in, int levels, int type)
{
   pyramid_t *gaussian = malloc(sizeof(pyramid_t));
   gaussian->levels = levels;
   gaussian->images = calloc(levels, sizeof(image_t));
   copy_image(gaussian->images + 0, in);
   for (int l = 1; l < levels; ++l)
      downsample(gaussian->images + l, gaussian->images + l - 1);
   if (type == G_PYRAMID) {
      *out = gaussian;
   }
   if (type == L_PYRAMID) {
      pyramid_t *laplacian = malloc(sizeof(pyramid_t));
      laplacian->levels = levels;
      laplacian->images = calloc(levels, sizeof(image_t));
      copy_image(laplacian->images + levels - 1, gaussian->images + levels - 1);
      for (int l = 0; l < levels - 1; ++l) {
         image_t inter;
         int odd_w = gaussian->images[l].width % 2;
         int odd_h = gaussian->images[l].height % 2;
         upsample(&inter, gaussian->images + l + 1, odd_w, odd_h);
         empty_image(laplacian->images + l, inter.width, inter.height, inter.comp);
         printf("%d\n", laplacian->images[l].size);
         for (int i = 0; i < laplacian->images[l].size; ++i)
            laplacian->images[l].data[i] = gaussian->images[l].data[i] - inter.data[i];
         free(inter.data);
      }
      for (int i = 0; i < levels; ++i)
         free(gaussian->images[i].data);
      free(gaussian);
      *out = laplacian;
   }
}

// static void destroy_pyramid(pyramid_t *pyramid)
// {
//    for (int i = 0; i < pyramid->levels; ++i)
//       free(pyramid->images[i].data);
//    free(pyramid);
// }

static unsigned char saturate_cast_u8(float in)
{
   if (in < 0.0f)
      return 0;
   if (in > 255.0f)
      return 255;
   return (unsigned char)(in + 0.5f);
}

static void output_bmp(const char *filename, image_t *image)
{
   float min = image->data[0];
   float max = image->data[0];
   for (int i = 0; i < image->size; ++i) {
      if (min > image->data[i])
         min = image->data[i];
      if (max < image->data[i])
         max = image->data[i];
   }
   float range = max - min;
#ifdef DEBUG
   printf("%s: (%f %f %f)\n", filename, min, max, range);
#endif
   unsigned char *out = malloc(image->size);
   for (int i = 0; i < image->size; ++i)
      out[i] = saturate_cast_u8((image->data[i] - min) / range * 255.0f);
   stbi_write_bmp(filename, image->width, image->height, image->comp, out);
   free(out);
}

int main(void)
{
   image_t images[inputs_length];
   float *weights[inputs_length];
   for (int i = 0; i < inputs_length; ++i) {
      load_image(images + i, inputs_path[i]);
      image_t gray;
      cvt_grayscale(&gray, images + i);
      weights[i] = malloc(sizeof(float) * gray.size);
      calculate_weight(weights[i], images + i, &gray);
      free(gray.data);
   }
   normalize_weights(weights, images[0].width * images[0].height,
         inputs_length);
   for (int i = 0; i < inputs_length; ++i) {
      char filename[255];
      snprintf(filename, sizeof(filename), "weight_%d.bmp", i);
      image_t weight;
      weight.width = images[0].width;
      weight.height = images[0].height;
      weight.comp = 1;
      weight.size = images[0].width * images[0].height;
      weight.data = weights[i];
      output_bmp(filename, &weight);
   }
   pyramid_t *G_py;
   pyramid_t *L_py;
   build_pyramid(&G_py, images + 0, 4, G_PYRAMID);
   build_pyramid(&L_py, images + 0, 4, L_PYRAMID);

   for (int l = 0; l < G_py->levels; ++l) {
      char filename[255];
      snprintf(filename, sizeof(filename), "gaussian_%d.bmp", l);
      output_bmp(filename, G_py->images + l);
   }
   for (int l = 0; l < L_py->levels; ++l) {
      char filename[255];
      snprintf(filename, sizeof(filename), "laplacian_%d.bmp", l);
      output_bmp(filename, L_py->images + l);
   }
}
