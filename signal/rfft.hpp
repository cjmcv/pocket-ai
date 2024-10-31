/*!
* \brief rfft / irfft
*/

#ifndef POCKET_AI_SIGNAL_RFFT_HPP_
#define POCKET_AI_SIGNAL_RFFT_HPP_

#include <iostream>
#include <vector>

#include "pocket-ai/util/logger.hpp"
#include "kissfft/kiss_fft.h"
#include "kissfft/kiss_fftr.h"

#include "kissfft/kiss_fft_c.h"
#include "kissfft/kiss_fftr_c.h"

namespace pai {
namespace signal {

// typedef struct {
//     kiss_fft_scalar r;
//     kiss_fft_scalar i;
// }kiss_fft_cpx;
typedef struct {
    float r;
    float i;
} Complex;

inline void GenHanningWindow(uint32_t length, bool sym, float *output) {
    double double_pi = 2 * 3.14159265358979323846264338327;
    uint32_t num_intervals = sym ? length - 1 : length;
    float interval = double_pi / num_intervals;
    for (uint32_t i=0; i<length; i++)
        output[i] = i*interval;
    for (uint32_t i=0; i<length; i++)
        output[i] = 0.5 - 0.5 * cos(output[i]);

}

class Rfft {
public:
    void Init(uint32_t fft_length) {
        fft_length_ = fft_length;
        state_ = kiss_fftr_alloc(fft_length_,0,0,0);

        rin_ = (float *)malloc((fft_length_ + 2) * sizeof(float));
        sout_ = (kiss_fft_cpx *)malloc(fft_length_ * sizeof(kiss_fft_cpx));
    }
    void Deinit() {
        free(rin_);
        free(sout_);
        free(state_);
    }

    void GetIo(float **real_in, uint32_t *real_size, Complex **cpx_out, uint32_t *cpxout_size) {
        *real_in = rin_;
        *real_size = fft_length_ * sizeof(float); // Only need to fill size fft_length_ * sizeof(float).
        *cpx_out = (Complex*)sout_; // real0, img0, real1, img1, real2, img2...
        *cpxout_size = (fft_length_ / 2 + 1) * sizeof(kiss_fft_cpx);

        // Example input
        // for (uint32_t i=0; i<fft_length_; i++)
        //     real_in[i] = rin[i];
        //
        // Example output
        // for (uint32_t i=0; i<fft_length_/2+1; i++)
        //     printf("(%f, %f), ", (float)cpx_out[i].r , (float)cpx_out[i].i);
    }
    void Run() {
        kiss_fftr(state_, rin_, sout_);
    }

private:
    uint32_t fft_length_;
    kiss_fftr_cfg state_;

    float *rin_;
    kiss_fft_cpx *sout_;
};

class Irfft {
public:
    void Init(uint32_t fft_length) {
        fft_length_ = fft_length;
        state_ = kiss_fftr_alloc(fft_length_,1,0,0);

        cin_ = (kiss_fft_cpx *)malloc(fft_length_ * sizeof(kiss_fft_cpx));
        rout_ = (float *)malloc((fft_length_ + 2) * sizeof(float));
        
        printf("fft_length_: %d, state: %p \n", fft_length_, state_);
    }

    void Deinit() {
        free(cin_);
        free(rout_);
        free(state_);
    }

    void GetIo(Complex **cpx_in, uint32_t *cpxin_size, float **real_out, uint32_t *real_size) {
        *cpx_in = (Complex *)cin_; // cpx_in: real0, img0, real1, img1, real2, img2...
        *cpxin_size = (fft_length_ / 2 + 1) * sizeof(kiss_fft_cpx);

        *real_out = rout_;
        *real_size = fft_length_ * sizeof(float);
    }

    void Run() {
        for (uint32_t i=0; i<fft_length_ / 2 + 1; i++) {
            cin_[i].r = cin_[i].r / fft_length_;
            cin_[i].i = cin_[i].i / fft_length_;
        }
        kiss_fftri(state_, cin_, rout_);

        // Example input, 
        // You can take the output of kiss_fftr divided by fft_length_ as the input parameter to the kiss_fftri function.
        // for (uint32_t i=0; i<fft_length_/2+1; i++)
        //     (float)cin_[i].r , (float)cin_[i].i
        //
        // Example output
        // for (uint32_t i=0; i<fft_length_; i++)
        //     printf("%f, ", rout_[i]);
    }
    
private:
    uint32_t fft_length_;
    kiss_fftr_cfg state_;

    kiss_fft_cpx *cin_;
    float *rout_;
};

//// python demo
// import sys
// import numpy as np
// import scipy

// def fft(x, frame_len = 36):
//     window =scipy.signal.windows.hann(frame_len, sym=False)
//     x_win = x * window
//     y = np.fft.rfft(x_win, n=frame_len)
//     r = y.real
//     i = y.imag
    
//     np.set_printoptions(precision=6,  suppress=True)
//     print("r", r)
//     print("i", i)
    
// if __name__ == "__main__":
//     x = []
//     for i in range(36):
//         x.append(i)
//     fft(x, 36)

} // prof.
} // pai.
#endif //POCKET_AI_PROF_TIMER_HPP_
