/*
 * threshold.h - the "Threshold" Fourier filter (Ventana::CmThres) as portable C.
 *
 *     out = Thres(Im, rad, thres)
 *
 *   1. two-dimensional Fourier transform            fft2dd(b)
 *   2. circular filter with threshold               filtcirct(imf, b, thres, rad, rad + 10); filtcirc(b, rad)
 *   3. inverse Fourier transform                    fft2di(b)
 *   4. intensity |b|^2, scaled to 0..255            getintens(b); findmamime(...); floattobyte(...)
 *
 * Plain C99 with no dependencies (only <math.h>): works for any image size, not just powers of two.
 * The step functions keep the names and arguments of the original program, so any of them can be
 * replaced by the original version.
 */
#ifndef THRESHOLD_H
#define THRESHOLD_H

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32) && defined(THRESHOLD_BUILD_DLL)
#define THRES_API __declspec(dllexport)
#else
#define THRES_API
#endif

/* Complex image: re[i][j], im[i][j] with i = row (0..ny-1), j = column (0..nx-1). */
struct compleja {
    int ny, nx;
    float **re, **im;
};

/* ---- the filter --------------------------------------------------------------------------------
 * im:  input image, ny rows x nx columns, row by row
 * out: output image (same size): intensity of the filtered image scaled to 0..255
 * rad: filter radius in pixels of the centred spectrum
 * thres: threshold on log10(1 + |F|) of the spectrum (see filtcirct)
 * Returns 0, or -1 for bad arguments or when memory runs out. */
THRES_API int Thres(const unsigned char *im, unsigned char *out, int ny, int nx, float rad, float thres);

/* The same for images of any bit depth (12/16-bit cameras): float in, float 0..255 out.
 * spectrum: if not NULL, receives the filtered spectrum, log-scaled to 0..255 (what imagfoumem shows). */
THRES_API int ThresF(const float *im, float *out, unsigned char *spectrum, int ny, int nx, float rad,
                     float thres);

/* ---- the steps (same names as the original program) --------------------------------------------- */
THRES_API struct compleja memoriac(int ny, int nx);   /* complex image, zeroed; re == NULL if no memory */
THRES_API void freecomp(struct compleja b);
THRES_API unsigned char **tablab(int ny, int nx);     /* byte table t[i][j] */
THRES_API float **tablaf(int ny, int nx);             /* float table t[i][j] */
THRES_API void liberab(unsigned char **t);
THRES_API void liberaf(float **t);

THRES_API void bytetofloat(unsigned char **im, struct compleja b);   /* b.re = im, b.im = 0 */
THRES_API int fft2dd(struct compleja b);   /* direct transform, zero frequency moved to (ny/2, nx/2) */
THRES_API int fft2di(struct compleja b);   /* inverse of fft2dd (includes the 1/(nx*ny) factor) */
THRES_API void imagfouf(float **imf, struct compleja b);             /* imf = log10(1 + |b|) */
THRES_API void filtcirct(float **imf, struct compleja b, float thres, float r1, float r2);
THRES_API void filtcirc(struct compleja b, float rad);               /* zero everything beyond rad */
THRES_API void imagfoumem(unsigned char **im, struct compleja b);    /* log spectrum as 0..255 */
THRES_API void getintens(struct compleja b);                        /* b.re = |b|^2, b.im = 0 */
THRES_API void findmamime(struct compleja b, float *max, float *min, float *mean);  /* of b.re */
THRES_API void floattobyte(struct compleja b, unsigned char **im, float max, float min);

#ifdef __cplusplus
}
#endif

#endif /* THRESHOLD_H */
