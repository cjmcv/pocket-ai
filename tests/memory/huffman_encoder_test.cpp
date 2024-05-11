/*!
* \brief . 
*/

#include "memory/huffman_encoder.hpp"

#include <cstdlib>
#include <ctime>

#include "gtest/gtest.h"

namespace {

using namespace pai::memory;

void HuffmanEncoderTest() {
    srand(time(0));

    HuffmanEncoder encoder;
    uint32_t size = 10000;
    uint8_t *raw = (uint8_t *)malloc(size);
    for (uint32_t i=0; i<size; i++) {
        if (i%3 == 0)
            raw[i] = 100;
        else
            raw[i] = rand();
    }
    // uint8_t c = 15;
    // for (int i=0; i<8; i++)
    //     printf("%d, ", get_bit(&c, i));

    // for (uint32_t i=0; i<size; i++)
    //     printf("%d, ", raw[i]);
    
    uint32_t encoded_size = 0;
    uint8_t *encoded_data = nullptr;
    encoder.Encode(raw, size, &encoded_data, &encoded_size);

    uint32_t decoded_size = 0;
    uint8_t *decoded_data = nullptr;
    encoder.Decode(encoded_data, encoded_size, &decoded_data, &decoded_size);
    printf("encoded size: %d, %d, %d.\n", size, encoded_size, decoded_size);

    EXPECT_LT(encoded_size, decoded_size*0.90f);
    EXPECT_EQ(size, decoded_size);
    for (uint32_t i=0; i<size; i++) {
        EXPECT_EQ(raw[i], decoded_data[i]);
    }

    encoder.Release();
    free(raw);
}

TEST(MemoryTest, HuffmanEncoder) {
    HuffmanEncoderTest();
}

}  // end of namespace.