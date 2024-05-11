#ifndef POCKET_AI_UTIL_BMP_READER_HPP_
#define POCKET_AI_UTIL_BMP_READER_HPP_

#include <fstream>
#include <vector>
#include <stdexcept>
#include <iostream>

#include "util/logger.hpp"

namespace pai {
namespace util {

#pragma pack(push, 1)
struct BmpFileHeader {
    uint16_t file_type{ 0x4D42 };          // File type always BM which is 0x4D42 (stored as hex uint16_t in little endian)
    uint32_t file_size{ 0 };               // Size of the file (in bytes)
    uint16_t reserved1{ 0 };               // <Not used here> Reserved, always 0
    uint16_t reserved2{ 0 };               // <Not used here> Reserved, always 0
    uint32_t offset_data{ 0 };             // Start position of pixel data (bytes from the beginning of the file)
};

struct BmpInfoHeader {
    uint32_t size{ 0 };                      // Size of this header (in bytes)
    int32_t width{ 0 };                      // width of bitmap in pixels
    int32_t height{ 0 };                     // width of bitmap in pixels
                                             //       (if positive, bottom-up, with origin in lower left corner)
                                             //       (if negative, top-down, with origin in upper left corner)
    uint16_t planes{ 1 };                    // <Not used here> No. of planes for the target device, this is always 1
    uint16_t bit_count{ 0 };                 // No. of bits per pixel
    uint32_t compression{ 0 };               // <Not used here> 0 or 3 - for uncompressed images.
    uint32_t size_image{ 0 };                // <Not used here> 0 - for uncompressed images
    int32_t x_pixels_per_meter{ 0 };         // <Not used here>
    int32_t y_pixels_per_meter{ 0 };         // <Not used here>
    uint32_t colors_used{ 0 };               // <Not used here> No. color indexes in the color table. Use 0 for the max number of colors allowed by bit_count
    uint32_t colors_important{ 0 };          // <Not used here> No. of colors used for displaying the bitmap. If 0 all colors are required
};

struct BmpColorHeader {
    uint32_t red_mask{ 0x00ff0000 };         // Bit mask for the red channel
    uint32_t green_mask{ 0x0000ff00 };       // Bit mask for the green channel
    uint32_t blue_mask{ 0x000000ff };        // Bit mask for the blue channel
    uint32_t alpha_mask{ 0xff000000 };       // Bit mask for the alpha channel
    uint32_t color_space_type{ 0x73524742 }; // Default "sRGB" (0x73524742)
    uint32_t unused[16]{ 0 };                // Unused data for sRGB color space
};
#pragma pack(pop)

class BmpReader {
public:
    BmpReader(uint32_t width, uint32_t height, uint32_t channels) {
        if (channels != 3 && channels != 4)
            PAI_LOGE("BmpReader only supports 3 or 4 channel inputs for now.\n");

        bmp_info_header_.width = width;
        bmp_info_header_.height = height;
        bmp_info_header_.bit_count = channels * 8;

        NormalizeHeader();
    }

    BmpReader(const char *fname) {
        std::ifstream inp{fname, std::ios_base::binary};
        if (!inp) {
            PAI_LOGE("Unable to open the input image file: %s \n.", fname);
        }

        ReadHeaders(inp);
        if (bmp_info_header_.height < 0) {
            PAI_LOGE("The program can treat only BmpReader images with the origin in the bottom left corner!");
        }

        // Jump to the pixel data location
        inp.seekg(file_header_.offset_data, inp.beg);

        // Some editors will put extra info in the image file, we only save the headers and the data.
        NormalizeHeader();

        // Here we check if we need to take into account row padding
        if (bmp_info_header_.width % 4 == 0) { // 32bit其channels为4，一定能被4整除；只有24的需要判断，因为24的channel为3
            inp.read((char *)data_.data(), data_.size());
        }
        else {
            std::vector<uint8_t> padding_row(new_stride_ - row_stride_);
            for (int y = 0; y < bmp_info_header_.height; ++y) {
                inp.read((char *)(data_.data() + row_stride_ * y), row_stride_);
                inp.read((char *)padding_row.data(), padding_row.size());
            }
        }
    }

    void Write(const char *fname) {
        std::ofstream of{fname, std::ios_base::binary};
        if (!of) {
            PAI_LOGE("Unable to open the output image file: %s \n.", fname);
        }

        WriteHeaders(of);

        if (bmp_info_header_.width % 4 == 0) {
            of.write((const char *)data_.data(), data_.size());
        }
        else {
            std::vector<uint8_t> padding_row(new_stride_ - row_stride_);
            for (int y = 0; y < bmp_info_header_.height; ++y) {
                of.write((const char *)(data_.data() + row_stride_ * y), row_stride_);
                of.write((const char *)padding_row.data(), padding_row.size());
            }
        }
    }

    inline uint8_t *data() { return &data_[0]; };

private:
    void NormalizeHeader() {
        row_stride_ = bmp_info_header_.width * bmp_info_header_.bit_count / 8;
        if (bmp_info_header_.bit_count == 32) {
            bmp_info_header_.size = sizeof(BmpInfoHeader) + sizeof(BmpColorHeader);
            file_header_.offset_data = sizeof(BmpFileHeader) + sizeof(BmpInfoHeader) + sizeof(BmpColorHeader);

            bmp_info_header_.compression = 3;
            file_header_.file_size = file_header_.offset_data + data_.size();
        }
        else if (bmp_info_header_.bit_count == 24) {
            bmp_info_header_.size = sizeof(BmpInfoHeader);
            file_header_.offset_data = sizeof(BmpFileHeader) + sizeof(BmpInfoHeader);

            bmp_info_header_.compression = 0;
            new_stride_ = row_stride_;
            while (new_stride_ % 4 != 0) {
                new_stride_++;
            }
            file_header_.file_size = file_header_.offset_data + data_.size() + bmp_info_header_.height * (new_stride_ - row_stride_);
        }
        data_.resize(bmp_info_header_.width * bmp_info_header_.height * bmp_info_header_.bit_count / 8);
    }

    void ReadHeaders(std::ifstream &inf) {
        inf.read((char *)&file_header_, sizeof(file_header_));
        if (file_header_.file_type != 0x4D42) {
            PAI_LOGE("Error! Unrecognized file format.");
        }
        inf.read((char *)&bmp_info_header_, sizeof(bmp_info_header_));

        // The BmpColorHeader is used only for transparent images
        if (bmp_info_header_.bit_count == 32) {
            // Check if the file has bit mask color information
            if (bmp_info_header_.size >= (sizeof(BmpInfoHeader) + sizeof(BmpColorHeader))) {
                inf.read((char *)&bmp_color_header_, sizeof(bmp_color_header_));
                // Check if the pixel data is stored as BGRA and if the color space type is sRGB
                CheckColorHeader(bmp_color_header_);
            }
            else {
                PAI_LOGE("Error! Unrecognized file format: This file does not seem to contain bit mask information.");
            }
        }
    }
    
    void WriteHeaders(std::ofstream &of) {
        of.write((const char *)&file_header_, sizeof(file_header_));
        of.write((const char *)&bmp_info_header_, sizeof(bmp_info_header_));
        if (bmp_info_header_.bit_count == 32) {
            of.write((const char *)&bmp_color_header_, sizeof(bmp_color_header_));
        }
    }

    // Check if the pixel data is stored as BGRA and if the color space type is sRGB
    void CheckColorHeader(BmpColorHeader &bmp_color_header){
        BmpColorHeader expected_color_header;
        if (expected_color_header.red_mask != bmp_color_header.red_mask ||
            expected_color_header.blue_mask != bmp_color_header.blue_mask ||
            expected_color_header.green_mask != bmp_color_header.green_mask ||
            expected_color_header.alpha_mask != bmp_color_header.alpha_mask) {
            PAI_LOGE("Unexpected color mask format! The program expects the pixel data to be in the BGRA format");
        }
        if (expected_color_header.color_space_type != bmp_color_header.color_space_type) {
            PAI_LOGE("Unexpected color space type! The program expects sRGB values");
        }
    }

private:
    BmpFileHeader file_header_;
    BmpInfoHeader bmp_info_header_;
    BmpColorHeader bmp_color_header_;
    std::vector<uint8_t> data_;

    uint32_t row_stride_;
    uint32_t new_stride_;
};

} // namespace util
} // namespace pai

#endif // POCKET_AI_UTIL_BMP_READER_HPP_