/*
 * threshold.c - the "Threshold" Fourier filter (Ventana::CmThres) as portable C99.
 *
 * The original program used the frame grabber (MbufGet2d / MbufPut2d) and Windows dialogs
 * (TInputDialog) for input and output, and image sizes fixed at compile time (NMAXJ x NMAXI).
 * Here the image, its size, the radius and the threshold are arguments, and the image may have
 * any size.  The filter itself follows the original step by step; see threshold.h.
 *
 * Build:  make            (program "thres" and the shared library used by PhaseScope)
 */
#include "threshold.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define PI 3.14159265358979323846

/* ======================================================================================
 * Memory (tablab / tablaf / memoriac): one block per table plus row pointers, so that
 * t[i][j] works as in the original program.
 * ====================================================================================== */
static void **rows(size_t ny, size_t nx, size_t size)
{
    void **t;
    char *block;
    size_t i;
    if (ny == 0 || nx == 0)
        return NULL;
    t = (void **)malloc(ny * sizeof(void *));
    block = (char *)calloc(ny * nx, size);
    if (t == NULL || block == NULL) {
        free(t);
        free(block);
        return NULL;
    }
    for (i = 0; i < ny; i++)
        t[i] = block + i * nx * size;
    return t;
}

static void free_rows(void **t)
{
    if (t != NULL) {
        free(t[0]);
        free(t);
    }
}

unsigned char **tablab(int ny, int nx) { return ny > 0 && nx > 0 ? (unsigned char **)rows(ny, nx, 1) : NULL; }
float **tablaf(int ny, int nx) { return ny > 0 && nx > 0 ? (float **)rows(ny, nx, sizeof(float)) : NULL; }
void liberab(unsigned char **t) { free_rows((void **)t); }
void liberaf(float **t) { free_rows((void **)t); }

struct compleja memoriac(int ny, int nx)
{
    struct compleja b;
    b.ny = ny;
    b.nx = nx;
    b.re = tablaf(ny, nx);
    b.im = tablaf(ny, nx);
    if (b.re == NULL || b.im == NULL) {
        liberaf(b.re);
        liberaf(b.im);
        b.re = b.im = NULL;
        b.ny = b.nx = 0;
    }
    return b;
}

void freecomp(struct compleja b)
{
    liberaf(b.re);
    liberaf(b.im);
}

void bytetofloat(unsigned char **im, struct compleja b)
{
    int i, j;
    for (i = 0; i < b.ny; i++)
        for (j = 0; j < b.nx; j++) {
            b.re[i][j] = (float)im[i][j];
            b.im[i][j] = 0.0f;
        }
}

/* ======================================================================================
 * Fourier transform.  Powers of two use the radix-2 FFT; other lengths (640, 480, 2448 …)
 * use Bluestein's algorithm on a power-of-two FFT, so every size gives the exact DFT.
 * Calculations are in double precision; the images are stored as float.
 * ====================================================================================== */
typedef struct {
    int n;           /* transform length */
    int m;           /* length of the power-of-two FFT that does the work (n itself, or >= 2n-1) */
    double *cs, *sn; /* twiddle factors cos/sin(2 pi k / m), k < m/2 */
    double *wr, *wi; /* Bluestein chirp exp(-i pi k^2 / n), k < n */
    double *br, *bi; /* FFT of the chirp filter (length m) */
    double *xr, *xi; /* work buffers (length m) */
} plan_t;

static int is_pow2(int n) { return n > 0 && (n & (n - 1)) == 0; }

static void fft_pow2(const plan_t *p, double *re, double *im, int inverse)
{
    int n = p->m, i, j, k, len, half, step;
    for (i = 1, j = 0; i < n; i++) { /* bit-reversed order */
        int bit = n >> 1;
        for (; j & bit; bit >>= 1)
            j ^= bit;
        j ^= bit;
        if (i < j) {
            double t = re[i]; re[i] = re[j]; re[j] = t;
            t = im[i]; im[i] = im[j]; im[j] = t;
        }
    }
    for (len = 2; len <= n; len <<= 1) {
        half = len >> 1;
        step = n / len;
        for (i = 0; i < n; i += len)
            for (k = 0; k < half; k++) {
                double c = p->cs[k * step], s = inverse ? p->sn[k * step] : -p->sn[k * step];
                int a = i + k, b = a + half;
                double vr = re[b] * c - im[b] * s, vi = re[b] * s + im[b] * c;
                re[b] = re[a] - vr;
                im[b] = im[a] - vi;
                re[a] += vr;
                im[a] += vi;
            }
    }
}

static void plan_free(plan_t *p)
{
    free(p->cs); free(p->sn); free(p->wr); free(p->wi);
    free(p->br); free(p->bi); free(p->xr); free(p->xi);
}

static int plan_init(plan_t *p, int n)
{
    int k;
    memset(p, 0, sizeof(*p));
    p->n = n;
    p->m = n;
    if (!is_pow2(n)) {
        p->m = 1;
        while (p->m < 2 * n - 1)
            p->m <<= 1;
    }
    p->cs = (double *)malloc((p->m / 2 + 1) * sizeof(double));
    p->sn = (double *)malloc((p->m / 2 + 1) * sizeof(double));
    p->xr = (double *)calloc(p->m, sizeof(double));
    p->xi = (double *)calloc(p->m, sizeof(double));
    if (!p->cs || !p->sn || !p->xr || !p->xi)
        goto fail;
    for (k = 0; k <= p->m / 2; k++) {
        p->cs[k] = cos(2.0 * PI * k / p->m);
        p->sn[k] = sin(2.0 * PI * k / p->m);
    }
    if (p->m == n)
        return 0;
    /* Bluestein: X_k = w_k * sum_j (x_j w_j) conj(w_{k-j}),  w_k = exp(-i pi k^2 / n) */
    p->wr = (double *)malloc(n * sizeof(double));
    p->wi = (double *)malloc(n * sizeof(double));
    p->br = (double *)calloc(p->m, sizeof(double));
    p->bi = (double *)calloc(p->m, sizeof(double));
    if (!p->wr || !p->wi || !p->br || !p->bi)
        goto fail;
    p->wr[0] = p->br[0] = 1.0; /* k = 0 */
    p->wi[0] = p->bi[0] = 0.0;
    for (k = 1; k < n; k++) {
        double a = PI * (double)(((long long)k * k) % (2LL * n)) / n; /* k^2 mod 2n keeps it exact */
        p->wr[k] = cos(a);
        p->wi[k] = -sin(a);
    }
    for (k = 1; k < n; k++) {
        p->br[k] = p->br[p->m - k] = p->wr[k];
        p->bi[k] = p->bi[p->m - k] = -p->wi[k];
    }
    fft_pow2(p, p->br, p->bi, 0);
    return 0;
fail:
    plan_free(p);
    return -1;
}

/* in-place transform of re/im (length n); unscaled in both directions */
static void fft1d(plan_t *p, double *re, double *im, int inverse)
{
    int n = p->n, m = p->m, k;
    if (m == n) {
        fft_pow2(p, re, im, inverse);
        return;
    }
    for (k = 0; k < n; k++) { /* inverse = conj(forward(conj(x))) */
        double xr = re[k], xi = inverse ? -im[k] : im[k];
        p->xr[k] = xr * p->wr[k] - xi * p->wi[k];
        p->xi[k] = xr * p->wi[k] + xi * p->wr[k];
    }
    for (k = n; k < m; k++)
        p->xr[k] = p->xi[k] = 0.0;
    fft_pow2(p, p->xr, p->xi, 0);
    for (k = 0; k < m; k++) {
        double r = p->xr[k] * p->br[k] - p->xi[k] * p->bi[k];
        double i = p->xr[k] * p->bi[k] + p->xi[k] * p->br[k];
        p->xr[k] = r;
        p->xi[k] = i;
    }
    fft_pow2(p, p->xr, p->xi, 1);
    for (k = 0; k < n; k++) {
        double r = p->xr[k] / m, i = p->xi[k] / m;
        re[k] = r * p->wr[k] - i * p->wi[k];
        im[k] = r * p->wi[k] + i * p->wr[k];
        if (inverse)
            im[k] = -im[k];
    }
}

/* circular shift by (dy, dx): the centred spectrum of fft2dd and its undoing in fft2di */
static int shift(struct compleja b, int dy, int dx)
{
    float **t = tablaf(b.ny, b.nx);
    float **parts[2];
    int p, i, j;
    if (t == NULL)
        return -1;
    parts[0] = b.re;
    parts[1] = b.im;
    for (p = 0; p < 2; p++) {
        memcpy(t[0], parts[p][0], (size_t)b.ny * b.nx * sizeof(float));
        for (i = 0; i < b.ny; i++)
            for (j = 0; j < b.nx; j++)
                parts[p][(i + dy) % b.ny][(j + dx) % b.nx] = t[i][j];
    }
    liberaf(t);
    return 0;
}

static int fft2d(struct compleja b, int inverse)
{
    plan_t pr, pc;
    double *re, *im;
    int i, j, n = b.nx > b.ny ? b.nx : b.ny, ok = -1;
    if (b.re == NULL || b.ny <= 0 || b.nx <= 0)
        return -1;
    if (plan_init(&pr, b.nx) != 0)
        return -1;
    if (plan_init(&pc, b.ny) != 0) {
        plan_free(&pr);
        return -1;
    }
    re = (double *)malloc(n * sizeof(double));
    im = (double *)malloc(n * sizeof(double));
    if (re != NULL && im != NULL) {
        double scale = inverse ? 1.0 / ((double)b.nx * b.ny) : 1.0;
        for (i = 0; i < b.ny; i++) { /* rows */
            for (j = 0; j < b.nx; j++) {
                re[j] = b.re[i][j];
                im[j] = b.im[i][j];
            }
            fft1d(&pr, re, im, inverse);
            for (j = 0; j < b.nx; j++) {
                b.re[i][j] = (float)re[j];
                b.im[i][j] = (float)im[j];
            }
        }
        for (j = 0; j < b.nx; j++) { /* columns */
            for (i = 0; i < b.ny; i++) {
                re[i] = b.re[i][j];
                im[i] = b.im[i][j];
            }
            fft1d(&pc, re, im, inverse);
            for (i = 0; i < b.ny; i++) {
                b.re[i][j] = (float)(re[i] * scale);
                b.im[i][j] = (float)(im[i] * scale);
            }
        }
        ok = 0;
    }
    free(re);
    free(im);
    plan_free(&pr);
    plan_free(&pc);
    return ok;
}

int fft2dd(struct compleja b)
{
    if (fft2d(b, 0) != 0)
        return -1;
    return shift(b, b.ny / 2, b.nx / 2); /* zero frequency to the centre (ny/2, nx/2) */
}

int fft2di(struct compleja b)
{
    if (shift(b, b.ny - b.ny / 2, b.nx - b.nx / 2) != 0) /* back from the centre */
        return -1;
    return fft2d(b, 1);
}

/* ======================================================================================
 * Operations in the frequency domain.  The centre of the spectrum is (ny/2, nx/2); r is the
 * distance from it in pixels.
 * ====================================================================================== */
static double radius(struct compleja b, int i, int j)
{
    double dy = i - b.ny / 2, dx = j - b.nx / 2;
    return sqrt(dy * dy + dx * dx);
}

void imagfouf(float **imf, struct compleja b)
{
    int i, j;
    for (i = 0; i < b.ny; i++)
        for (j = 0; j < b.nx; j++)
            imf[i][j] = (float)log10(1.0 + sqrt((double)b.re[i][j] * b.re[i][j] + (double)b.im[i][j] * b.im[i][j]));
}

/* Circular filter with threshold:
 *   - a component weaker than the threshold (imf < thres, imf = log10(1 + |F|)) is removed; the zero
 *     frequency (mean brightness) is always kept;
 *   - within r1 of the centre the other components pass unchanged, from r1 to r2 they fade out
 *     smoothly (cosine), beyond r2 everything is removed. */
void filtcirct(float **imf, struct compleja b, float thres, float r1, float r2)
{
    int i, j;
    for (i = 0; i < b.ny; i++)
        for (j = 0; j < b.nx; j++) {
            double r = radius(b, i, j), w;
            if (r > r2)
                w = 0.0;
            else if (r > 0.0 && imf[i][j] < thres)
                w = 0.0;
            else if (r <= r1 || r2 <= r1)
                w = 1.0;
            else
                w = 0.5 * (1.0 + cos(PI * (r - r1) / (r2 - r1)));
            b.re[i][j] = (float)(b.re[i][j] * w);
            b.im[i][j] = (float)(b.im[i][j] * w);
        }
}

/* Circular filter: keep the frequencies within rad of the centre, remove the others. */
void filtcirc(struct compleja b, float rad)
{
    int i, j;
    for (i = 0; i < b.ny; i++)
        for (j = 0; j < b.nx; j++)
            if (radius(b, i, j) > rad)
                b.re[i][j] = b.im[i][j] = 0.0f;
}

/* The (filtered) spectrum as an 8-bit picture: log10(1 + |F|) scaled to 0..255. */
void imagfoumem(unsigned char **im, struct compleja b)
{
    int i, j;
    float **t = tablaf(b.ny, b.nx);
    double max = 0.0;
    if (t == NULL)
        return;
    imagfouf(t, b);
    for (i = 0; i < b.ny; i++)
        for (j = 0; j < b.nx; j++)
            if (t[i][j] > max)
                max = t[i][j];
    for (i = 0; i < b.ny; i++)
        for (j = 0; j < b.nx; j++)
            im[i][j] = (unsigned char)(max > 0.0 ? 255.0 * t[i][j] / max + 0.5 : 0.0);
    liberaf(t);
}

/* ======================================================================================
 * Intensity and conversion back to bytes
 * ====================================================================================== */
void getintens(struct compleja b)
{
    int i, j;
    for (i = 0; i < b.ny; i++)
        for (j = 0; j < b.nx; j++) {
            b.re[i][j] = b.re[i][j] * b.re[i][j] + b.im[i][j] * b.im[i][j];
            b.im[i][j] = 0.0f;
        }
}

void findmamime(struct compleja b, float *max, float *min, float *mean)
{
    int i, j;
    double sum = 0.0, hi = b.re[0][0], lo = b.re[0][0];
    for (i = 0; i < b.ny; i++)
        for (j = 0; j < b.nx; j++) {
            double v = b.re[i][j];
            if (v > hi) hi = v;
            if (v < lo) lo = v;
            sum += v;
        }
    *max = (float)hi;
    *min = (float)lo;
    *mean = (float)(sum / ((double)b.ny * b.nx));
}

/* max - min, or 0 when the image is flat apart from rounding noise (which would otherwise be
 * stretched to 0..255) */
static double span(float max, float min)
{
    double range = (double)max - min;
    return range > 1e-5 * fabs((double)max) ? range : 0.0;
}

void floattobyte(struct compleja b, unsigned char **im, float max, float min)
{
    int i, j;
    double range = span(max, min);
    for (i = 0; i < b.ny; i++)
        for (j = 0; j < b.nx; j++) {
            double v = range > 0.0 ? 255.0 * (b.re[i][j] - min) / range : 0.0;
            im[i][j] = (unsigned char)(v < 0.0 ? 0 : v > 255.0 ? 255 : (int)(v + 0.5));
        }
}

/* ======================================================================================
 * The filter: out = Thres(Im, rad, thres)
 * ====================================================================================== */

/* Steps 1-4 on b (which holds the image in b.re); leaves the intensity in b.re. */
static int filter(struct compleja b, unsigned char *spectrum, float rad, float thres,
                  float *max, float *min, float *mean)
{
    float **imf = tablaf(b.ny, b.nx);
    if (imf == NULL)
        return -1;
    if (rad < 0.0f)
        rad = 0.0f;
    if (fft2dd(b) != 0) {                        /* direct transform */
        liberaf(imf);
        return -1;
    }
    imagfouf(imf, b);                            /* operations in the frequency domain */
    filtcirct(imf, b, thres, rad, rad + 10.0f);
    filtcirc(b, rad);
    liberaf(imf);
    if (spectrum != NULL) {                      /* what the original shows before transforming back */
        unsigned char **s = tablab(b.ny, b.nx);
        int i;
        if (s == NULL)
            return -1;
        imagfoumem(s, b);
        for (i = 0; i < b.ny; i++)
            memcpy(spectrum + (size_t)i * b.nx, s[i], (size_t)b.nx);
        liberab(s);
    }
    if (fft2di(b) != 0)                          /* inverse transform */
        return -1;
    getintens(b);                                /* intensity |b|^2 in b.re */
    findmamime(b, max, min, mean);
    return 0;
}

int Thres(const unsigned char *im, unsigned char *out, int ny, int nx, float rad, float thres)
{
    struct compleja b;
    unsigned char **t;
    float max, min, mean;
    int i, j, ok;
    if (im == NULL || out == NULL || ny <= 0 || nx <= 0)
        return -1;
    b = memoriac(ny, nx);
    t = tablab(ny, nx);
    if (b.re == NULL || t == NULL) {
        freecomp(b);
        liberab(t);
        return -1;
    }
    for (i = 0; i < ny; i++)
        for (j = 0; j < nx; j++)
            t[i][j] = im[(size_t)i * nx + j];
    bytetofloat(t, b);                           /* conversion to float */
    ok = filter(b, NULL, rad, thres, &max, &min, &mean);
    if (ok == 0) {
        floattobyte(b, t, max, min);             /* normalise to 0..255 */
        for (i = 0; i < ny; i++)
            memcpy(out + (size_t)i * nx, t[i], (size_t)nx);
    }
    liberab(t);
    freecomp(b);
    return ok;
}

int ThresF(const float *im, float *out, unsigned char *spectrum, int ny, int nx, float rad, float thres)
{
    struct compleja b;
    float max, min, mean;
    int i, j, ok;
    if (im == NULL || out == NULL || ny <= 0 || nx <= 0)
        return -1;
    b = memoriac(ny, nx);
    if (b.re == NULL)
        return -1;
    for (i = 0; i < ny; i++)
        for (j = 0; j < nx; j++)
            b.re[i][j] = im[(size_t)i * nx + j];    /* b.im is already zero */
    ok = filter(b, spectrum, rad, thres, &max, &min, &mean);
    if (ok == 0) {
        double range = span(max, min);
        for (i = 0; i < ny; i++)
            for (j = 0; j < nx; j++)
                out[(size_t)i * nx + j] = range > 0.0 ? (float)(255.0 * (b.re[i][j] - min) / range) : 0.0f;
    }
    freecomp(b);
    return ok;
}
