/* DO NOT EDIT THIS FILE */

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "log.h"
#include "opencl.h"

static const unsigned int OPENCL_MAX_DEVICE_COUNT   = 20;
static const unsigned int OPENCL_MAX_PLATFORM_COUNT = 20;

char* opencl_kernel_path = NULL;

static char* get_kernel_path(void) {
    char* path = NULL;

    if (opencl_kernel_path != NULL) {
        if (access(opencl_kernel_path, F_OK) < 0) {
            LOG_ERROR("file `%s` cannot be accessed", opencl_kernel_path);
            goto fail_exit;
        }

        path = opencl_kernel_path;
    }

    if (path == NULL) {
        if (access(__KERNEL_FILE__, F_OK) < 0) {
            LOG_ERROR("file `%s` cannot be accessed", __KERNEL_FILE__);
            goto fail_exit;
        }

        path = __KERNEL_FILE__;
    }

    return path;

fail_exit:
    return NULL;
}

int opencl_load_kernel_code(char** code, size_t* len) {
    if (code == NULL || len == NULL) {
        LOG_ERROR_NULL_PTR();
        goto fail_exit;
    }

    char* path = get_kernel_path();
    if (path == NULL) {
        LOG_ERROR("failed to obtain kernel path");
        goto fail_exit;
    }

    FILE* file = fopen(path, "r");
    if (file == NULL) {
        LOG_ERROR_ERRNO("fopen");
        goto fail_exit;
    }

    if (fseek(file, 0, SEEK_END) < 0) {
        LOG_ERROR_ERRNO("fseek");
        goto fail_close_file;
    }

    long file_size_ret = ftell(file);
    if (file_size_ret < 0) {
        LOG_ERROR_ERRNO("ftell");
        goto fail_close_file;
    }

    *len = file_size_ret;
    rewind(file);

    *code = malloc(*len);
    if (*code == NULL) {
        LOG_ERROR_ERRNO("malloc");
        goto fail_close_file;
    }

    size_t file_read = fread(*code, sizeof(char), *len, file);
    if (file_read != *len) {
        LOG_ERROR_ERRNO("fread");
        goto fail_free_code;
    }

    fclose(file);

    return 0;

fail_free_code:
    free(*code);
fail_close_file:
    fclose(file);
fail_exit:
    *code = NULL;
    *len  = 0;
    return -1;
}

int opencl_get_device_id(unsigned int platform_index, unsigned int device_index, cl_device_id* context_device_id) {
    cl_platform_id platform_ids[OPENCL_MAX_PLATFORM_COUNT];
    cl_device_id device_ids[OPENCL_MAX_DEVICE_COUNT];

    cl_uint num_platform_ids;
    cl_int status = clGetPlatformIDs(OPENCL_MAX_PLATFORM_COUNT, platform_ids, &num_platform_ids);
    if (status != CL_SUCCESS) {
        LOG_ERROR("clGetPlatformIDs (%d)", status);
        goto fail_exit;
    }

    if (platform_index >= num_platform_ids) {
        LOG_ERROR("invalid platform index `%d`", platform_index);
        goto fail_exit;
    }

    cl_uint num_device_ids;
    status = clGetDeviceIDs(platform_ids[platform_index], CL_DEVICE_TYPE_ALL, OPENCL_MAX_DEVICE_COUNT, device_ids,
                            &num_device_ids);
    if (status != CL_SUCCESS) {
        LOG_ERROR("clGetDeviceIDs (%d)", status);
        goto fail_exit;
    }

    if (device_index >= num_device_ids) {
        LOG_ERROR("invalid device index");
        goto fail_exit;
    }

    *context_device_id = device_ids[device_index];

    return 0;

fail_exit:
    return -1;
}

int opencl_print_device_info(cl_device_id device_id) {
    cl_int status;

    cl_platform_id platform_id;
    status = clGetDeviceInfo(device_id, CL_DEVICE_PLATFORM, sizeof(platform_id), &platform_id, NULL);
    if (status != CL_SUCCESS) {
        LOG_ERROR("clGetDeviceInfo(CL_DEVICE_PLATFORM) (%d)", status);
        goto fail_exit;
    }

    char platform_vendor[64];
    status = clGetPlatformInfo(platform_id, CL_PLATFORM_VENDOR, sizeof(platform_vendor), platform_vendor, NULL);
    if (status != CL_SUCCESS) {
        LOG_ERROR("clGetPlatformInfo(CL_PLATFORM_VENDOR) (%d)", status);
        goto fail_exit;
    }
    printf("OpenCL Platform Vendor: %s\n", platform_vendor);

    char platform_name[64];
    status = clGetPlatformInfo(platform_id, CL_PLATFORM_NAME, sizeof(platform_name), platform_name, NULL);
    if (status != CL_SUCCESS) {
        LOG_ERROR("clGetPlatformInfo(CL_PLATFORM_NAME) (%d)", status);
        goto fail_exit;
    }
    printf("OpenCL Platform Name: %s\n", platform_name);

    char device_vendor[64];
    status = clGetDeviceInfo(device_id, CL_DEVICE_VENDOR, sizeof(device_vendor), device_vendor, NULL);
    if (status != CL_SUCCESS) {
        LOG_ERROR("clGetDeviceInfo(CL_DEVICE_VENDOR) (%d)", status);
        goto fail_exit;
    }
    printf("OpenCL Device Vendor: %s\n", device_vendor);

    char device_name[128];
    status = clGetDeviceInfo(device_id, CL_DEVICE_NAME, sizeof(device_name), device_name, NULL);
    if (status != CL_SUCCESS) {
        LOG_ERROR("clGetDeviceInfo(CL_DEVICE_NAME) (%d)", status);
        goto fail_exit;
    }
    printf("OpenCL Device Name: %s\n", device_name);

    cl_device_type device_type;
    status = clGetDeviceInfo(device_id, CL_DEVICE_TYPE, sizeof(device_type), &device_type, NULL);
    if (status != CL_SUCCESS) {
        LOG_ERROR("clGetDeviceInfo(CL_DEVICE_TYPE) (%d)", status);
        goto fail_exit;
    }
    if (device_type == CL_DEVICE_TYPE_CPU) {
        printf("OpenCL Device Type: CL_​DEVICE_​TYPE_​CPU\n");
    } else if (device_type == CL_DEVICE_TYPE_GPU) {
        printf("OpenCL Device Type: CL_​DEVICE_​TYPE_​GPU\n");
    } else {
        printf("OpenCL Device Type: UNKNOWN\n");
    }

    cl_uint device_frequency;
    status =
        clGetDeviceInfo(device_id, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(device_frequency), &device_frequency, NULL);
    if (status != CL_SUCCESS) {
        LOG_ERROR("clGetDeviceInfo(CL_DEVICE_MAX_CLOCK_FREQUENCY) (%d)", status);
        goto fail_exit;
    }
    printf("OpenCL Device Max Clock Frequency: %u\n", device_frequency);

    cl_uint device_compute_units;
    status = clGetDeviceInfo(device_id, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(device_compute_units),
                             &device_compute_units, NULL);
    if (status != CL_SUCCESS) {
        LOG_ERROR("clGetDeviceInfo(CL_DEVICE_MAX_COMPUTE_UNITS) (%d)", status);
        goto fail_exit;
    }
    printf("OpenCL Device Max Compute Units: %u\n", device_compute_units);

    return 0;

fail_exit:
    return -1;
}

int opencl_print_build_log(cl_program program, cl_device_id device_id) {
    size_t len;
    char* buffer;

    cl_int status = clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &len);
    if (status != CL_SUCCESS) {
        LOG_ERROR("clGetProgramBuildInfo (%d)", status);
        goto fail_exit;
    }

    buffer = calloc(len, sizeof(char));
    if (buffer == NULL) {
        LOG_ERROR_ERRNO("calloc");
        goto fail_exit;
    }

    status = clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, len, buffer, NULL);
    if (status != CL_SUCCESS) {
        LOG_ERROR("clGetProgramBuildInfo (%d)", status);
        goto fail_free_buffer;
    }

    printf("%s", buffer);
    free(buffer);

    return 0;

fail_free_buffer:
    free(buffer);
fail_exit:
    return -1;
}