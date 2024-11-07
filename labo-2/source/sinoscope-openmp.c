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

#pragma omp parallel shared(sinoscope)
    {
        sinoscope_t local_sinoscope;
        local_sinoscope.width            = sinoscope->width;
        local_sinoscope.height           = sinoscope->height;
        local_sinoscope.dx               = sinoscope->dx;
        local_sinoscope.dy               = sinoscope->dy;
        local_sinoscope.taylor           = sinoscope->taylor;
        local_sinoscope.phase0           = sinoscope->phase0;
        local_sinoscope.phase1           = sinoscope->phase1;
        local_sinoscope.time             = sinoscope->time;
        local_sinoscope.interval         = sinoscope->interval;
        local_sinoscope.interval_inverse = sinoscope->interval_inverse;

        float cst = 2 * M_PI;

#pragma omp for collapse(2) schedule(dynamic)
        for (int i = 0; i < local_sinoscope.height; i++) {
            for (int j = 0; j < local_sinoscope.width; j++) {
                float py    = local_sinoscope.dy * i - cst;
                float px    = local_sinoscope.dx * j - cst;
                float value = 0;

#pragma omp simd reduction(+ : value)
                for (int k = 1; k <= sinoscope->taylor; k += 2) {
                    value += sin(px * k * sinoscope->phase1 + sinoscope->time) / k;
                    value += cos(py * k * sinoscope->phase0) / k;
                }

                value = (atan(value) - atan(-value)) / M_PI;
                value = (value + 1) * 100;

                pixel_t pixel;
                color_value(&pixel, value, local_sinoscope.interval, local_sinoscope.interval_inverse);

                int index = (i * 3) + (j * 3) * local_sinoscope.width;

                sinoscope->buffer[index + 0] = pixel.bytes[0];
                sinoscope->buffer[index + 1] = pixel.bytes[1];
                sinoscope->buffer[index + 2] = pixel.bytes[2];
            }
        }
    }

    return 0;
}
