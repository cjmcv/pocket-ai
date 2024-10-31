/*!
* \brief . 
*/

#include "pocket-ai/signal/rfft.hpp"

#include <time.h>
#include <unistd.h>

#include "gtest/gtest.h"

namespace {

using namespace pai::signal;

float real_in_target36[] = {0.000000, 0.007596, 0.060307, 0.200962, 0.467911, 0.893031, 1.500000, 2.302929, 3.305407, 4.500000, 5.868240, 7.381111, 9.000000, 10.678120, 12.362310, 13.995191, 15.517541, 16.870865, 18.000000, 18.855673, 19.396927, 19.593267, 19.426489, 18.892061, 17.999998, 16.775249, 15.257425, 13.500000, 11.568926, 9.540709, 7.500002, 5.536794, 3.743291, 2.210582, 1.025227, 0.265865};
float cpx_out_target36[] = {324.000000, 0.000000, -162.000000, 77.349716, 0.000004, -17.187920, -0.000001, -4.295948, -0.000016, -1.717218, 0.000007, -0.857317, 0.000002, -0.488485, -0.000005, -0.303790, 0.000002, -0.200886, -0.000000, -0.138832, -0.000008, -0.099041, 0.000008, -0.072154, -0.000001, -0.053166, -0.000004, -0.039173, 0.000008, -0.028425, -0.000009, -0.019789, 0.000004, -0.012534, -0.000008, -0.006069, 0.000000, 0.000000};

void RfftTest() {
    float limit = 0.00001f;
    uint32_t nfft = 36;
    Rfft fft;
    fft.Init(nfft);

    float win[36];
    GenHanningWindow(nfft, false, win);
    // for (uint32_t i=0; i<nfft; i++)
    //     printf("%f, ", win[i]);

    float *real_in;
    Complex *fft_out;
    uint32_t realin_size, fft_out_size;
    fft.GetIo(&real_in, &realin_size, &fft_out, &fft_out_size);

    for (uint32_t i = 0; i < realin_size / sizeof(float); i++) {
        real_in[i] = i*win[i];       
    }
    fft.Run();

    for (uint32_t i = 0; i < realin_size / sizeof(float); i++) {
        EXPECT_LE(real_in[i]-limit, real_in_target36[i]);
        EXPECT_GE(real_in[i]+limit, real_in_target36[i]);
    }

    for (uint32_t i=0; i<fft_out_size / sizeof(Complex); i++) {
        // if (i%3==0) printf("\n");
        // printf("<%d>(%f, %f), ", i, fft_out[i].r, fft_out[i].i);
        EXPECT_LE(fft_out[i].r-limit, cpx_out_target36[i*2]);
        EXPECT_GE(fft_out[i].r+limit, cpx_out_target36[i*2]);
        EXPECT_LE(fft_out[i].i-limit, cpx_out_target36[i*2+1]);
        EXPECT_GE(fft_out[i].i+limit, cpx_out_target36[i*2+1]);
    }
    // printf("\n");

    /////////////
    Irfft ifft;
    ifft.Init(nfft);
    Complex *cpx_in;
    float *ifft_out;
    uint32_t cpx_size, ifft_out_size;
    ifft.GetIo(&cpx_in, &cpx_size, &ifft_out, &ifft_out_size);

    memcpy(cpx_in, fft_out, cpx_size);
    ifft.Run();
    for (uint32_t i=0; i<ifft_out_size / sizeof(float); i++) {
        // if (i%3==0) printf("\n");
        // printf("%f, ", ifft_out[i]);
        EXPECT_LE(ifft_out[i]-limit, real_in_target36[i]);
        EXPECT_GE(ifft_out[i]+limit, real_in_target36[i]);
    }

    fft.Deinit();
    ifft.Deinit();
}

TEST(SignalTest, Rfft) {
    RfftTest();
}

}  // end of namespace.