#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#include "log.h"
#include "sinoscope.h"
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


int sinoscope_opencl_init(sinoscope_opencl_t* opencl, cl_device_id opencl_device_id, unsigned int width,
                          unsigned int height) {

    /* Initialiser le matériel, contexte et queue de commandes */
	printf("init called: \n");
    cl_int err;
    opencl->device_id = opencl_device_id;
    opencl->context = clCreateContext(0, 1, &opencl->device_id, NULL, NULL, &err);
    if (err != CL_SUCCESS) { 
        return -1; 
    }

    opencl->queue = clCreateCommandQueue(opencl->context, opencl->device_id, 0, &err);
    if (err != CL_SUCCESS) { 
        clReleaseContext(opencl->context);
        return -1; 
    }

    size_t buffer_size = width * height * 3 ;
    unsigned char* host_buffer = (unsigned char*)malloc(buffer_size);
    if (host_buffer == NULL) {
        fprintf(stderr, "Failed to allocate sinoscope buffer\n");
        clReleaseCommandQueue(opencl->queue);
        clReleaseContext(opencl->context);
        return -1; 
    }

    opencl->buffer = clCreateBuffer(opencl->context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, buffer_size, host_buffer, &err);
    free(host_buffer);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to create OpenCL buffer: %d\n", err);
        clReleaseCommandQueue(opencl->queue);
        clReleaseContext(opencl->context);
        return -1; // Handle buffer creation failure
    }

    char* code;
    size_t length;
    opencl_load_kernel_code(&code, &length);
    LOG_ERROR("printing code after reading file.");

    cl_program program = clCreateProgramWithSource(opencl->context, 1, (const char**)&code, &length, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error on creating program from src file: %d\n", err);
        clReleaseMemObject(opencl->buffer);
        clReleaseCommandQueue(opencl->queue);
        clReleaseContext(opencl->context);
        return -1;
    }

    err = clBuildProgram(program, 1, &opencl->device_id, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        // opencl_print_build_log(program, opencl_device_id);
        fprintf(stderr, "Error on building program: %d\n", err);
        clReleaseProgram(program);
        clReleaseMemObject(opencl->buffer);
        clReleaseCommandQueue(opencl->queue);
        clReleaseContext(opencl->context);
        return -1;
    }

    free(code);
    opencl->kernel = clCreateKernel(program, "sinoscope_image_kernel", &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error on creating kernel: %d\n", err);
        clReleaseProgram(program);
        clReleaseMemObject(opencl->buffer);
        clReleaseCommandQueue(opencl->queue);
        clReleaseContext(opencl->context);
        return -1;
    }

    clReleaseProgram(program); // Release program after creating kernel
	printf("init done: \n");
    return 0; // Success
}

void sinoscope_opencl_cleanup(sinoscope_opencl_t* opencl) {
	printf("cleanup launched: \n");
    if (opencl->kernel) {
        clReleaseKernel(opencl->kernel);
        opencl->kernel = NULL; // Set to NULL to avoid dangling pointer
    }
    
    if (opencl->buffer) {
        clReleaseMemObject(opencl->buffer);
        opencl->buffer = NULL; // Set to NULL to avoid dangling pointer
    }
    
    if (opencl->queue) {
        clReleaseCommandQueue(opencl->queue);
        opencl->queue = NULL; // Set to NULL to avoid dangling pointer
    }
    
    if (opencl->context) {
        clReleaseContext(opencl->context);
        opencl->context = NULL; // Set to NULL to avoid dangling pointer
    }
}


int sinoscope_image_opencl(sinoscope_t* sinoscope) {
    cl_int return_value;

    // L’implémentation avec OpenCL doit passer en premier paramètre le buffer partagé.
    return_value = clSetKernelArg(sinoscope->opencl->kernel, 0, sizeof(cl_mem), &(sinoscope->opencl->buffer));
    if (return_value != CL_SUCCESS)
        return -1;

    // Ensuite, le second paramètre est une structure contenant toutes les valeurs entières de sinoscope_t
    sinoscope_int_values_t int_values;
    int_values.buffer_size = sinoscope->buffer_size;
    int_values.width       = sinoscope->width;
    int_values.height      = sinoscope->height;
    int_values.taylor      = sinoscope->taylor;
    int_values.interval    = sinoscope->interval;

    return_value = clSetKernelArg(sinoscope->opencl->kernel, 1, sizeof(sinoscope_int_values_t), &int_values);
    if (return_value != CL_SUCCESS)
        return -1;

    // troisième paramètre est une structure contenant toutes valeurs à virgule flotante.
    sinoscope_float_values_t float_values;
    float_values.interval_inverse = sinoscope->interval_inverse;
    float_values.time             = sinoscope->time;
    float_values.max              = sinoscope->max;
    float_values.phase0           = sinoscope->phase0;
    float_values.phase1           = sinoscope->phase1;
    float_values.dx               = sinoscope->dx;
    float_values.dy               = sinoscope->dy;

    return_value = clSetKernelArg(sinoscope->opencl->kernel, 2, sizeof(sinoscope_float_values_t), &float_values);
    if (return_value != CL_SUCCESS)
        return -1;

    // Fourth step is to enqueue kernel for execution, la répartition du calcul doit se faire en deux dimensions.
    size_t global_work_size[2] = {sinoscope->width, sinoscope->height};
    return_value               = clEnqueueNDRangeKernel(sinoscope->opencl->queue, sinoscope->opencl->kernel, 2, NULL,
                                                        global_work_size, NULL, 0, NULL, NULL);
    if (return_value != CL_SUCCESS)
        return -1;

    return_value = clFinish(sinoscope->opencl->queue);
    if (return_value != CL_SUCCESS)
        return -1;

    // Fifth step is to synchronise and return the results
    return_value = clEnqueueReadBuffer(sinoscope->opencl->queue, sinoscope->opencl->buffer, CL_TRUE, 0,
                                       sinoscope->buffer_size, sinoscope->buffer, 0, NULL, NULL);

    if (return_value != CL_SUCCESS)
        return -1;

    return 0;
}
