#include <iostream>
#include <fstream>
#include <cstring>
#include <array>

#include <string>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <algorithm>
#include <cstdint>
#include <opencv2/opencv.hpp>
#include <sys/stat.h>


bool checkFileFormat(std::string &filename, std::string &extension) ;

// #include "main.h"
using namespace std::string_literals;

//#define NDEBUG

struct Resolution {
    uint16_t width;
    uint16_t height;
    uint16_t depth;
};

struct ColorCompression {
    uint8_t field1;
    uint8_t field2;
};

struct MipHeader {
    std::array<uint8_t, 12> magicNumber;
    time_t lastModified;
    uint32_t width;
    uint32_t height;
    uint32_t depth;
    Resolution resolution;
    uint32_t pageCount;
    uint32_t pageSize;
    uint32_t layerCount;
    uint8_t mipLevelCount;
    uint8_t mipTailStartLevel;
    uint32_t mipTailOffset;
    uint32_t mipTailSize;
    std::array<uint32_t, 16> pageIndexes;
    std::string format;
    uint8_t flags;
    ColorCompression colorCompression;
    uint8_t pageCompressionType;
    uint8_t pageCompressionLevel;
    uint32_t dataOffset;

    MipHeader() {
        magicNumber = {0xAB, 'L', 'T', 'X', 1, 0, 0, 0xBB, '\r', '\n', '\x1A', '\n'};
        lastModified = 0; //  время последнего изменения файла исходного изображения
        width = 0; // ширина исходного изображения
        height = 0; // высота исходного изображения
        depth = 1; // Глубина всегда равна 1
        resolution = {0, 0, 1}; //  разрешение страниц
        pageCount = 0; //  общие количество страниц
        pageSize = 65536; // Размер страницы всегда равен 65536
        layerCount = 1; // Количество слоев всегда равно 1
        mipLevelCount = 0; //  число MIP уровней изображения
        mipTailStartLevel = 0; //  номер MIP уровня начиная с которого остальные MIP уровни занимают не более одной страницы
        mipTailOffset = 0; //  смещение в байтах относительно конца заголовка с которого начинаются данные MIP tail
        mipTailSize = 0; //  число байт данных всех MIP tail уровней
        pageIndexes.fill(0); // индекс первой страницы для каждого MIP уровня
        format = ""; //  формат исходных данных (число каналов и тип)
        flags = 0; // Флаги не используются в задании
        colorCompression = {0, 0}; // Цветовая компрессия не используется в задании
        pageCompressionType = 0; // Тип компрессии страниц не используется в задании. Выставляем в 0.
        pageCompressionLevel = 0; // Степень компрессии страниц не используется в задании.
        dataOffset = sizeof(MipHeader); // Смещение на начало данных в файле. Если не используем компрессию равно размеру заголовка.
    }
};


time_t getLastModifiedTime(const char *filename) {
    struct stat fileInfo;
    if (stat(filename, &fileInfo) != 0) {
        // Обработка ошибки
        return 0;
    }
    return fileInfo.st_mtime;
}

Resolution fillResolution(uint32_t height, uint32_t width) {
    Resolution resolution;
    resolution.height = height;
    resolution.width = width;
    resolution.depth = 1;
    return resolution;
}

template<typename T>
std::vector<cv::Mat_<T>> createFixedMipLevels(cv::Mat_<T> image, int numLevels = 16) {
    std::vector<cv::Mat_<T>> mipLevels(numLevels);
    mipLevels[0] = image;
    for (int i = 1; i < numLevels; i++) {
        if (mipLevels[i - 1].cols < 2 || mipLevels[i - 1].rows < 2) {
            mipLevels[i] = cv::Mat_<T>(2, 2);
        } else {
            cv::resize(mipLevels[i - 1], mipLevels[i], cv::Size(), 0.5, 0.5);
        }
    }
    return mipLevels;
}


bool checkFileFormat(std::string &filename, std::string &extension) {
    extension = filename.substr(filename.find_last_of("."s) + 1);
    if (extension == "exr"s || extension == "png"s || extension == "jpg" || extension == "jpeg"s) return true;
    return false;
}

#ifdef NDEBUG

void printOutputFile();

#endif

int main(int argc, char **argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " image" << std::endl;
        return 1;
    }

    std::string filename = argv[1];
    std::string extension;
    cv::Mat_<cv::Vec4b> image = cv::imread(filename);
    if (checkFileFormat(filename, extension)) {

        if (image.empty()) {
            std::cerr << "Failed to read image: " << argv[1] << std::endl;
            return 1;
        }
    }

    auto MIPLevels = createFixedMipLevels(image);
    MipHeader header;
    header.lastModified = getLastModifiedTime(argv[1]);
    header.width = image.cols;
    header.height = image.rows;
    header.resolution = fillResolution(image.cols, image.rows);
    size_t dataSize = 0;
    for (const auto &mipLevel: MIPLevels) {
        dataSize += mipLevel.total() * mipLevel.elemSize();
    }
    header.pageCount = (dataSize + header.pageSize - 1) / header.pageSize;
    header.mipLevelCount = MIPLevels.size();


    size_t pageSize = header.pageSize;
    for (size_t i = 0; i < MIPLevels.size(); i++) {
        size_t mipSize = MIPLevels[i].total() * MIPLevels[i].elemSize();
        if (mipSize <= pageSize) {
            header.mipTailStartLevel = i;
            break;
        }
    }
    header.mipTailOffset = sizeof(MipHeader);
    size_t mipTailSize = 0;
    for (size_t i = header.mipTailStartLevel; i < MIPLevels.size(); i++) {
        mipTailSize += MIPLevels[i].total() * MIPLevels[i].elemSize();
    }
    header.mipTailSize = mipTailSize;

    size_t pageIndex = 0;
    for (size_t i = 0; i < MIPLevels.size(); i++) {
        header.pageIndexes[i] = pageIndex;
        size_t mipSize = MIPLevels[i].total() * MIPLevels[i].elemSize();
        pageIndex += (mipSize + pageSize - 1) / pageSize;
    }
    header.format = extension;

    std::ofstream file("output.ltx", std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open output file" << std::endl;
        return 1;
    }

    file.write(reinterpret_cast<const char *>(&header), sizeof(header));
    for (const auto &mipLevel: MIPLevels) {
        file.write(reinterpret_cast<const char *>(mipLevel.data), mipLevel.total() * mipLevel.elemSize());
    }


#ifdef NDEBUG
    std::cout << "\n =========Done!=========== \n\n";
              printOutputFile();
#endif

}

#ifdef NDEBUG

void printOutputFile() {
    std::ifstream file("../output.ltx", std::ios::binary);
    if (!file) {
        std::cerr << "Не удалось открыть файл output.ltx" << std::endl;
        return;
    }

    MipHeader header;
    file.read(reinterpret_cast<char *>(&header), sizeof(header));

    std::cout << "Ширина: " << header.width << std::endl;
    std::cout << "Высота: " << header.height << std::endl;
    std::cout << "Глубина: " << header.depth << std::endl;
    std::cout << "Разрешение: " << header.resolution.width << "x" << header.resolution.height << "x"
              << header.resolution.depth << std::endl;
    std::cout << "Количество страниц: " << header.pageCount << std::endl;
    std::cout << "Размер страницы: " << header.pageSize << std::endl;
    std::cout << "Количество слоев: " << header.layerCount << std::endl;
    std::cout << "Количество уровней MIP: " << static_cast<int>(header.mipLevelCount) << std::endl;
    std::cout << "Начальный уровень MIP tail: " << static_cast<int>(header.mipTailStartLevel) << std::endl;
    std::cout << "Смещение MIP tail: " << header.mipTailOffset << std::endl;
    std::cout << "Размер MIP tail: " << header.mipTailSize << std::endl;

    for (int i = 0; i < 16; i++) {
        if (header.pageIndexes[i] != 0) {
            std::cout << "Индекс первой страницы для уровня MIP " << i + 1
                      << ": " << header.pageIndexes[i] + 1
                      << std::endl;
        }
    }


}

#endif





