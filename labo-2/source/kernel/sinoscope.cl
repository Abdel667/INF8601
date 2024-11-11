#include "helpers.cl"

typedef struct sinoscope_int_values {
    unsigned int buffer_size;
    unsigned int width;
    unsigned int height;
    unsigned int taylor;
    unsigned int interval;
} sinoscope_int_values_t;

typedef struct sinoscope_float_values {
    float interval_inverse;
    float time;
    float max;
    float phase0;
    float phase1;
    float dx;
    float dy;
} sinoscope_float_values_t;

// OpenCL kernel
__kernel void sinoscope_image_kernel(
    __global unsigned char* buffer,
    sinoscope_int_values_t sinoscope_int_values,
    sinoscope_float_values_t sinoscope_float_values)
{
    int i = get_global_id(0); 
    int j = get_global_id(1); 

    if (i < sinoscope_int_values.width && j < sinoscope_int_values.height) {
        float px = sinoscope_float_values.dx * j - 2 * M_PI;
        float py = sinoscope_float_values.dy * i - 2 * M_PI;
        float value = 0;

        for (int k = 1; k <= sinoscope_int_values.taylor; k += 2) {
            value += sin(px * k * sinoscope_float_values.phase1 + sinoscope_float_values.time) / k;
            value += cos(py * k * sinoscope_float_values.phase0) / k;
        }

        value = (atan(value) - atan(-value)) / M_PI;
        value = (value + 1) * 100;

        pixel_t pixel;
        color_value(&pixel, value, sinoscope_int_values.interval, sinoscope_float_values.interval_inverse);

        int index = (i * 3) + (j * 3) * sinoscope_int_values.width;

        buffer[index + 0] = pixel.bytes[0];
        buffer[index + 1] = pixel.bytes[1];
        buffer[index + 2] = pixel.bytes[2];
    }
}
