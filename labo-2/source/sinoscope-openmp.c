#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#include "color.h"
#include "log.h"
#include "sinoscope.h"

int sinoscope_image_openmp(sinoscope_t* sinoscope) {
    if (sinoscope == NULL) {
        LOG_ERROR_NULL_PTR();
        return -1;
    }

    
    float px, py;
    pixel_t pixel;
    int index;
    float value;

#pragma omp parallel private(px, py, index, pixel) 
#pragma omp for schedule(dynamic) reduction(+: value)
for (int i = 0; i < sinoscope->width; i++) {
        py    = sinoscope->dy * i - 2 * M_PI;;
        for (int j = 0; j < sinoscope->height; j++) {
            px    = sinoscope->dx * j - 2 * M_PI;;
            value = 0;

            for (int k = 1; k <= sinoscope->taylor; k += 2) {
                value += sin(px * k * sinoscope->phase1 + sinoscope->time) / k;
                value += cos(py * k * sinoscope->phase0) / k;
            }

            value = (2 * atan(value)) / M_PI;
            value = (value + 1) * 100;

            color_value(&pixel, value, sinoscope->interval, sinoscope->interval_inverse);

            index = (i * 3) + (j * 3) * sinoscope->width;

            sinoscope->buffer[index + 0] = pixel.bytes[0];
            sinoscope->buffer[index + 1] = pixel.bytes[1];
            sinoscope->buffer[index + 2] = pixel.bytes[2];
        }
    }

    return 0;
}
