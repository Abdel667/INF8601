/* DO NOT EDIT THIS FILE */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "color.h"

unsigned int color_get_interval(double max) {
    if (max < 4.0) {
        max = 4.0;
    }

    return max / 4.0;
}

double color_get_interval_inverted(double max) {
    if (max < 4.0) {
        max = 4.0;
    }

    return 4.0 / max;
}

void color_value(pixel_t* pixel, double value, double max) {
    if (isnan(value)) {
        *pixel = pixel_black;
        return;
    }

    unsigned int interval    = color_get_interval(max);
    double interval_inverted = color_get_interval_inverted(max);

    unsigned int x = (((unsigned int)value % interval) * 255) * interval_inverted;
    unsigned int i = value * interval_inverted;

    switch (i) {
    case 0:
        pixel->bytes[0] = 0;
        pixel->bytes[1] = x;
        pixel->bytes[2] = 255;
        pixel->bytes[3] = 255;
        break;
    case 1:
        pixel->bytes[0] = 0;
        pixel->bytes[1] = 255;
        pixel->bytes[2] = 255 - x;
        pixel->bytes[3] = 255;
        break;
    case 2:
        pixel->bytes[0] = x;
        pixel->bytes[1] = 255;
        pixel->bytes[2] = 0;
        pixel->bytes[3] = 255;
        break;
    case 3:
        pixel->bytes[0] = 255;
        pixel->bytes[1] = 255 - x;
        pixel->bytes[2] = 0;
        pixel->bytes[3] = 255;
        break;
    case 4:
        pixel->bytes[0] = 255;
        pixel->bytes[1] = 0;
        pixel->bytes[2] = x;
        pixel->bytes[3] = 255;
        break;
    default:
        *pixel = pixel_white;
        break;
    }
}

// void hue(struct rgb** image, int width, int height) {
//     int i, j;

//     *image             = (struct rgb*)calloc(width * height, sizeof(struct rgb));
//     struct rgb* img    = *image;
//     int interval       = get_color_interval((float)height);
//     float interval_inv = get_color_interval_inv((float)height);
//     for (j = 0; j < height; j++) {
//         for (i = 0; i < width; i++) {
//             value_color(&img[j * width + i], (float)j, interval, interval_inv);
//         }
//     }
// }
