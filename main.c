#include <math.h>
#include <stdio.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

typedef struct {
	int w, h, comp;
	float *data, *gray, *weight;
} image_t;

typedef struct {
	float *C, *S, *E;
} metrics_t;

static const char *images_path[] = {
	"images/A.jpg",
	"images/B.jpg",
	"images/C.jpg",
	"images/D.jpg",
};
static const int images_length = 4;

static float laplacian_kernel[][3] = {
	{0, 1, 0},
	{1, -4, 1},
	{0, 1, 0},
};

static float gaussian(float in, float mean, float spread) {
	/* unnormalized gaussian function */
	return expf(-0.5f * (((in - mean) * (in - mean)) / (spread * spread)));
}

static unsigned char saturate_cast_u8(float in) {
	if (in < 0.0f)
		return 0;
	if (in > 255.0f)
		return 255;
	return (unsigned char)(in + 0.5f);
}

static void output_bmp_data(const char *filename, float *in, int width, int height) {
	unsigned char *out = malloc(width * height);
	/* assumes normalized */
	for (int i = 0; i < width * height; ++i)
		out[i] = saturate_cast_u8(in[i] * 255.0f);
	stbi_write_bmp(filename, width, height, 1, out);
}

static void compute_weight(image_t *img) {
	metrics_t metric;

	metric.C = malloc(sizeof(float) * img->w * img->h);
	metric.S = malloc(sizeof(float) * img->w * img->h);
	metric.E = malloc(sizeof(float) * img->w * img->h);

	for (int y = 0; y < img->h; ++y) {
		for (int x = 0; x < img->w; ++x) {
			float sum = 0.0f;
			for (int dy = -1; dy <= 1; ++dy) {
				for (int dx = -1; dx <= 1; ++dx) {
					int ny = y + dy;
					int nx = x + dx;
					/* boundary */
					ny = ny < 0 ? 0 : ny >= img->h ? img->h - 1 : ny;
					nx = nx < 0 ? 0 : nx >= img->w ? img->w - 1 : nx;

					sum += img->gray[ny * img->w + nx] * laplacian_kernel[dy + 1][dx + 1];
				}
			}
			metric.C[y * img->w + x] = fabsf(sum);
		}
	}

	/* standard deviation */
	for (int i = 0; i < img->w * img->h; ++i) {
		float r = img->data[i * img->comp + 0];
		float g = img->data[i * img->comp + 1];
		float b = img->data[i * img->comp + 2];
		float u = (r + g + b) / 3;
		metric.S[i] = sqrtf(((r - u) * (r - u)
					+ (g - u) * (g - u)
					+ (b - u) * (b - u)) / 3.0f);
	}

	/* gassian distribution */
	float spread = 0.2f;
	for (int i = 0; i < img->w * img->h; ++i) {
		float r = img->data[i * img->comp + 0];
		float g = img->data[i * img->comp + 1];
		float b = img->data[i * img->comp + 2];
		metric.E[i] = gaussian(r, 0.5f, spread)
			* gaussian(g, 0.5f, spread)
			* gaussian(b, 0.5f, spread);
	}

	float wC = 1.0f;
	float wS = 1.0f;
	float wE = 1.0f;
	for (int i = 0; i < img->w * img->h; ++i) {
		img->weight[i] = powf(metric.C[i], wC)
			* powf(metric.S[i], wS)
			* powf(metric.E[i], wE);
	}

	free(metric.C);
	free(metric.S);
	free(metric.E);
}

int main(void) {
	image_t images[images_length];

	for (int i = 0; i < images_length; ++i) {
		images[i].data = stbi_loadf(images_path[i], &images[i].w, &images[i].h,
				&images[i].comp, 0);
		images[i].gray = stbi_loadf(images_path[i], &images[i].w, &images[i].h,
				&images[i].comp, 1);
		images[i].weight = malloc(sizeof(float) * images[i].w * images[i].h);
		compute_weight(&images[i]);
	}

	/* normalize weights */
	for (int i = 0; i < images[0].w * images[0].h; ++i) {
		float sum = 0.0f;

		for (int j = 0; j < images_length; ++j)
			sum += images[j].weight[i];

		if (sum == 0.0f)
			sum = 1.0f;

		for (int j = 0; j < images_length; ++j)
			images[j].weight[i] /= sum;
	}

	for (int i = 0; i < images_length; ++i) {
		char filename[255];
		snprintf(filename, sizeof(filename), "weight_%d.bmp", i + 1);
		output_bmp_data(filename, images[i].weight, images[i].w, images[i].h);
	}

	for (int i = 0; i < images_length; ++i) {
		free(images[i].weight);
		stbi_image_free(images[i].gray);
		stbi_image_free(images[i].data);
	}
}
