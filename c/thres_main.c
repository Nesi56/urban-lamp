/*
 * thres - run the "Threshold" Fourier filter on an image file.
 *
 *     thres input.pgm output.pgm [rad] [thres] [spectrum.pgm]
 *
 * rad (default 24) is the filter radius in pixels and thres (default 1) the threshold, as in the
 * dialogs of the original program.  Images are PGM files (8 or 16 bit): ImageJ, GIMP, IrfanView
 * and OpenCV read and write them.  spectrum.pgm (optional) receives the filtered spectrum.
 */
#include "threshold.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>

static int read_number(FILE *f, int *value)
{
    int c = fgetc(f);
    while (c != EOF && (isspace(c) || c == '#')) {
        if (c == '#')
            while (c != EOF && c != '\n')
                c = fgetc(f);
        c = fgetc(f);
    }
    if (c == EOF || !isdigit(c))
        return -1;
    *value = 0;
    while (c != EOF && isdigit(c)) {
        *value = *value * 10 + (c - '0');
        c = fgetc(f);
    }
    return 0; /* the single white-space character after the number has been read */
}

/* Read a binary PGM (P5) into floats; returns NULL on error. */
static float *read_pgm(const char *path, int *ny, int *nx)
{
    FILE *f = fopen(path, "rb");
    float *img = NULL;
    int maxval, i, n;
    if (f == NULL) {
        perror(path);
        return NULL;
    }
    if (fgetc(f) != 'P' || fgetc(f) != '5' || read_number(f, nx) || read_number(f, ny) ||
        read_number(f, &maxval) || *nx <= 0 || *ny <= 0 || maxval <= 0 || maxval > 65535) {
        fprintf(stderr, "%s: not a binary PGM image (P5)\n", path);
        fclose(f);
        return NULL;
    }
    n = *nx * *ny;
    img = (float *)malloc((size_t)n * sizeof(float));
    for (i = 0; img != NULL && i < n; i++) {
        int hi = fgetc(f), lo = maxval > 255 ? fgetc(f) : 0;
        if (hi == EOF || lo == EOF) {
            fprintf(stderr, "%s: file is shorter than its %d x %d pixels\n", path, *nx, *ny);
            free(img);
            img = NULL;
            break;
        }
        img[i] = (float)(maxval > 255 ? (hi << 8) | lo : hi);
    }
    fclose(f);
    return img;
}

static int write_pgm(const char *path, const unsigned char *img, int ny, int nx)
{
    FILE *f = fopen(path, "wb");
    size_t n = (size_t)nx * ny;
    if (f == NULL) {
        perror(path);
        return -1;
    }
    fprintf(f, "P5\n%d %d\n255\n", nx, ny);
    if (fwrite(img, 1, n, f) != n || fclose(f) != 0) {
        fprintf(stderr, "%s: could not write the image\n", path);
        return -1;
    }
    return 0;
}

int main(int argc, char **argv)
{
    int ny, nx, i, n, status = 1;
    float rad = 24.0f, thres = 1.0f, *img, *out;
    unsigned char *bytes, *spectrum;
    if (argc < 3 || argc > 6) {
        fprintf(stderr, "usage: %s input.pgm output.pgm [rad=24] [thres=1] [spectrum.pgm]\n", argv[0]);
        return 2;
    }
    if (argc > 3)
        rad = (float)atof(argv[3]);
    if (argc > 4)
        thres = (float)atof(argv[4]);
    img = read_pgm(argv[1], &ny, &nx);
    if (img == NULL)
        return 1;
    n = nx * ny;
    out = (float *)malloc((size_t)n * sizeof(float));
    bytes = (unsigned char *)malloc((size_t)n);
    spectrum = argc > 5 ? (unsigned char *)malloc((size_t)n) : NULL;
    if (out == NULL || bytes == NULL || (argc > 5 && spectrum == NULL)) {
        fprintf(stderr, "out of memory\n");
    } else if (ThresF(img, out, spectrum, ny, nx, rad, thres) != 0) {
        fprintf(stderr, "the filter failed (out of memory?)\n");
    } else {
        for (i = 0; i < n; i++)
            bytes[i] = (unsigned char)(out[i] + 0.5f);
        status = write_pgm(argv[2], bytes, ny, nx) != 0;
        if (status == 0 && spectrum != NULL)
            status = write_pgm(argv[5], spectrum, ny, nx) != 0;
        if (status == 0)
            printf("%s: %d x %d, radius %g px, threshold %g -> %s\n", argv[1], nx, ny, rad, thres, argv[2]);
    }
    free(img);
    free(out);
    free(bytes);
    free(spectrum);
    return status;
}
