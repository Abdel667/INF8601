/* DO NOT EDIT THIS FILE */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "grid.h"
#include "log.h"

grid_t* grid_create(unsigned int width, unsigned int height, unsigned int padding) {
    grid_t* grid = malloc(sizeof(*grid));
    if (grid == NULL) {
        LOG_ERROR_ERRNO("malloc");
        goto fail_malloc1;
    }

    grid->width         = width;
    grid->height        = height;
    grid->padding       = padding;
    grid->width_padded  = width + 2 * padding;
    grid->height_padded = height + 2 * padding;

    grid->data = calloc(grid->width_padded * grid->height_padded, sizeof(*grid->data));
    if (grid->data == NULL) {
        LOG_ERROR_ERRNO("malloc");
        goto fail_malloc2;
    }

    return grid;

fail_malloc2:
    free(grid);
fail_malloc1:
    return NULL;
}

void grid_destroy(grid_t* grid) {
    if (grid != NULL) {
        free(grid->data);
        free(grid);
    }
}

static int grid_assert_equal_dimensions(grid_t* grid, grid_t* other) {
    if (grid->height != other->height || grid->width != other->width) {
        LOG_ERROR("grid dimensions are different");
        goto fail_exit;
    }
    /* TODO */
    /*
            if (grid->padding != dst->padding) {
            LOG_ERROR("grid paddings are different");
            goto fail_exit;
            }

            if (grid->height_padded != other->height_padded || grid->width_padded != other->width_padded) {
            LOG_ERROR("grid padded dimensions are different");
            goto fail_exit;
            }
    */
    return 0;

fail_exit:
    return -1;
}

grid_t* grid_clone(grid_t* grid) {
    if (grid == NULL) {
        LOG_ERROR("NULL pointer received");
        goto fail_exit;
    }

    grid_t* new_grid = grid_create(grid->width, grid->height, grid->padding);
    if (new_grid == NULL) {
        goto fail_exit;
    }

    if (grid_copy_data(grid, new_grid) < 0) {
        goto fail_copy;
    }

    return new_grid;

fail_copy:
    grid_destroy(new_grid);
fail_exit:
    return NULL;
}

grid_t* grid_clone_with_padding(grid_t* grid, unsigned int padding) {
    if (grid == NULL) {
        LOG_ERROR("NULL pointer received");
        goto fail_exit;
    }

    grid_t* new_grid = grid_create(grid->width, grid->height, padding);
    if (new_grid == NULL) {
        goto fail_exit;
    }

    if (grid_copy_data(grid, new_grid) < 0) {
        goto fail_copy;
    }

    return new_grid;

fail_copy:
    grid_destroy(new_grid);
fail_exit:
    return NULL;
}

int grid_copy_data(grid_t* src, grid_t* dst) {
    if (grid_assert_equal_dimensions(src, dst) < 0) {
        goto fail_exit;
    }

    for (int j = 0; j < src->height; j++) {
        for (int i = 0; i < src->width; i++) {
            *grid_get_cell(dst, i, j) = *grid_get_cell(src, i, j);
        }
    }

    return 0;

fail_exit:
    return -1;
}

int grid_copy_block(grid_t* src, unsigned int x1, unsigned int y1, unsigned int w, unsigned int h, grid_t* dst,
                    unsigned int x2, unsigned int y2) {
    if (x1 + w > src->width || y1 + h > src->height) {
        LOG_ERROR("invalid source bounds (%u > %u) || (%u > %u)", x1 + w, src->width, y1 + h, src->height);
        goto fail_exit;
    }

    if (x2 + w > dst->width || y2 + h > dst->height) {
        LOG_ERROR("invalid destination bounds (%u > %u) || (%u > %u)", x2 + w, dst->width, y2 + h, dst->height);
        goto fail_exit;
    }

    for (int j = 0; j < h; j++) {
        for (int i = 0; i < w; i++) {
            *grid_get_cell(dst, x2 + i, y2 + j) = *grid_get_cell(src, x1 + i, y1 + j);
        }
    }

    return 0;

fail_exit:
    return -1;
}

int grid_copy_inner_border(grid_t* src, grid_t* dst) {
    if (src->width != dst->width) {
        LOG_ERROR("width mismatch between the grids (src=%u, dst=%u)", src->width, dst->width);
        goto fail_exit;
    }

    if (src->height != dst->height) {
        LOG_ERROR("height mismatch between the grids (src=%u, dst=%u)", src->height, dst->height);
        goto fail_exit;
    }

    for (int i = 0; i < src->width; i++) {
        *grid_get_cell(dst, i, 0)               = *grid_get_cell(src, i, 0);
        *grid_get_cell(dst, i, dst->height - 1) = *grid_get_cell(src, i, src->height - 1);
    }

    for (int j = 0; j < src->height; j++) {
        *grid_get_cell(dst, 0, j)              = *grid_get_cell(src, 0, j);
        *grid_get_cell(dst, dst->width - 1, j) = *grid_get_cell(src, src->width - 1, j);
    }

    return 0;

fail_exit:
    return -1;
}

int grid_set(grid_t* grid, double value) {
    if (grid == NULL) {
        LOG_ERROR("NULL pointer received");
        goto fail_exit;
    }

    for (int j = 0; j < grid->height_padded; j++) {
        for (int i = 0; i < grid->width_padded; i++) {
            *grid_get_cell_padded(grid, i, j) = value;
        }
    }

    return 0;

fail_exit:
    return -1;
}

int grid_set_min(grid_t* src, grid_t* dst) {
    if (src == NULL || dst == NULL) {
        LOG_ERROR("NULL pointer received");
        goto fail_exit;
    }

    if (grid_assert_equal_dimensions(src, dst) < 0) {
        goto fail_exit;
    }

    for (int j = 0; j < src->height; j++) {
        for (int i = 0; i < src->width; i++) {
            double* act     = grid_get_cell(src, i, j);
            double* desired = grid_get_cell(dst, i, j);
            if (*act < *desired) {
                *act = *desired;
            }
        }
    }

    return 0;

fail_exit:
    return -1;
}

int grid_set_padding_from_inner_bound(grid_t* grid) {
    if (grid == NULL) {
        LOG_ERROR("NULL pointer received");
        goto fail_exit;
    }

    if (grid->padding == 0) {
        LOG_ERROR("grid does not contain any padding");
        goto fail_exit;
    }

    int xs = 0;
    int xe = grid->width - 1;
    int ys = 0;
    int ye = grid->height - 1;

    /* copy line at Y=0 -> Y=-1 and Y=H-1 -> Y=H */
    for (int i = 0; i < grid->width; i++) {
        *grid_get_cell(grid, i, ys - 1) = *grid_get_cell(grid, i, ys);
        *grid_get_cell(grid, i, ye + 1) = *grid_get_cell(grid, i, ye);
    }

    /* copy line at X=0 -> X=-1 and X=W-1 -> X=W */
    for (int j = 0; j < grid->height; j++) {
        *grid_get_cell(grid, xs - 1, j) = *grid_get_cell(grid, xs, j);
        *grid_get_cell(grid, xe + 1, j) = *grid_get_cell(grid, xe, j);
    }

    /* copy corners */
    *grid_get_cell(grid, xs - 1, ys - 1) = *grid_get_cell(grid, xs, ys);
    *grid_get_cell(grid, xe + 1, ys - 1) = *grid_get_cell(grid, xe, ys);
    *grid_get_cell(grid, xe + 1, ye + 1) = *grid_get_cell(grid, xe, ye);
    *grid_get_cell(grid, xs - 1, ye + 1) = *grid_get_cell(grid, xs, ye);

    return 0;

fail_exit:
    return -1;
}

int grid_multiply(grid_t* grid, double factor) {
    if (grid == NULL) {
        LOG_ERROR("NULL pointer received");
        goto fail_exit;
    }

    for (int j = 0; j < grid->height_padded; j++) {
        for (int i = 0; i < grid->width_padded; i++) {
            *grid_get_cell_padded(grid, i, j) *= factor;
        }
    }

    return 0;

fail_exit:
    return -1;
}

double grid_max(grid_t* grid) {
    if (grid == NULL) {
        LOG_ERROR("NULL pointer received");
        goto fail_exit;
    }

    double current_max = 0.0;
    for (int j = 0; j < grid->height_padded; j++) {
        for (int i = 0; i < grid->width_padded; i++) {
            double value = *grid_get_cell_padded(grid, i, j);
            if (value > current_max) {
                current_max = value;
            }
        }
    }

    return current_max;

fail_exit:
    return NAN;
}

int grid_fdump(grid_t* grid, char* prefix, FILE* file) {
    if (grid == NULL) {
        LOG_ERROR("NULL pointer received");
        goto fail_exit;
    }

    for (int j = 0; j < grid->height_padded; j++) {
        if (prefix != NULL) {
            printf("%s ", prefix);
        }
        for (int i = 0; i < grid->width_padded; i++) {
            double value = *grid_get_cell_padded(grid, i, j);
            fprintf(file, "%#6.2f ", value);
        }
        fprintf(file, "\n");
    }

    return 0;

fail_exit:
    return -1;
}