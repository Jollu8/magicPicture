#include <iostream>
#include <fstream>
#include <cstring>
#include <array>
#include <chrono>
#include <string>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <algorithm>
#include <opencv2/opencv.hpp>


//#define NDEBUG



using tm_clock = std::chrono::system_clock;
namespace fs = std::filesystem;
using uintType = int;
const int PAGE_SIZE = 65536;
const std::string path_image = "../lena.png";
const std::string path_outputLTX = "../output.ltx";

struct Header {
    std::array<int, 12> magicNumber{};
    std::chrono::system_clock::time_point lastModified;
    int width{};
    int height{};
    int depth{};
    int pageResolution[3]{};
    uint32_t pageCount{};
    uint32_t pageSize{};
    uint32_t layerCount{};
    uint8_t mipLevelCount{};
    uint8_t mipTailStart{};
    uint32_t mipTailOffset{};
    uint32_t mipTailSize{};
    uint32_t mipLevelPageIndex[16]{};
    uint32_t format{};
    uint8_t flags{};
    uint8_t colorCompression[2]{};
    uint8_t compressionType{};
    uint8_t compressionLevel{};
    uint32_t dataOffset{};
};

class MIP {
public:
    virtual ~MIP() = default;

#ifdef NDEBUG
    virtual void print() const = 0;
#endif
};

class MIPLevel : public MIP {
    std::vector<cv::Mat> levels_;
    int width_;
    int channels_;
    u_long depth_;
public:
    int height_;
// MIPLevel() = delete;

    MIPLevel(cv::Mat &&image) {
        width_ = image.cols;
        height_ = image.rows;
        channels_ = image.channels();
        depth_ = image.elemSize();
        levels_.emplace_back(std::move(image));

    }

    void insertImage(cv::Mat &image) {
        levels_.emplace_back(std::move(image));
    }

    const cv::Mat &getFirstMat() const {
        return levels_.at(0);
    }
     const cv::Mat &getFromIndex(u_long i) const {
        return levels_[i];
    }

    decltype(auto) level_size() const {
        return levels_.size();
    }

     const cv::Mat &level_top() const {
        return levels_.back();
    }

    decltype(auto) getWidth() const {
        return width_;
    }

    decltype(auto) getHeight() const {
        return height_;
    }

    decltype(auto) getChannels() const {
        return channels_;
    }

    decltype(auto) getDepth() const {
        return depth_;
    }
#ifdef NDEBUG

    void print() const override {
        for (auto i = 0ul; i < levels_.size(); ++i)
            std::cout << "MIP уровень " << i << ": " << levels_[i].cols << "x" << levels_[i].rows << std::endl;

    }

#endif
};

class MIPTail : public MIP {
    uintType tailStart_;
    u_long tailOfSet_;
    uintType tailSize_;

public:
    MIPTail() : tailStart_(0), tailOfSet_(0), tailSize_(0) {}

    decltype(auto) get_tail_start() const {
        return tailStart_;
    }
    decltype(auto) get_tailsOfSet() const {
        return tailOfSet_;
    }
    decltype(auto) get_tail_size() const {
        return tailSize_;
    }
    void set_tail_size(uintType i) {
        tailSize_ = i;
    }
    void set_tailOfSet(uintType i) {
        tailOfSet_ = i;
    }
    void set_tail_start(u_long start) {
        tailStart_ = start;
    }
#ifdef NDEBUG
    void print() const override {
        std::cout << "MIP Tail size= "<< tailSize_ <<std::endl;
    }
#endif

};

class PagesData : public MIP {
     // Размер страницы в байтах
    uintType pixelSize_; // Размер пикселя в байтах
    uintType pixelPerPage_; // Количество пикселей на странице
    int pageWidth_; // Ширина страницы в пикселях
    int pageHeight_; // Высота страницы в пикселях

    std::vector<std::vector<uchar>> data_;
public:
    PagesData() = delete;

    PagesData(const cv::Mat &image)  {
        pixelSize_ = (image.channels() * image.elemSize());
        this->pixelPerPage_ = PAGE_SIZE / pixelSize_;
        this->pageWidth_ = std::sqrt(pixelPerPage_);
        this->pageHeight_ = pageWidth_;


    }
    const auto &getData() const {
        return data_;
    }
    void insert(std::vector<uchar> &data) {
        data_.emplace_back(std::move(data));
    }

    auto getPixelSize() const {
        return pixelSize_;
    }

    auto getPixelPerPage() const {
        return pixelPerPage_;
    }

    auto getPageHeight() const {
        return pageHeight_;
    }

    auto getPageWidth() const {
        return pageWidth_;
    }

    auto getPageDataSize() const {
        return data_.size();
    }
#ifdef NDEBUG
    void print() const override {
        std::cout << "Page Data size = " << PAGE_SIZE << std::endl;

    }
#endif
};

tm_clock::time_point time_lastModified(const std::filesystem::path &filePath) {

    auto ftime = fs::last_write_time(filePath);
    return std::chrono::time_point_cast<tm_clock::duration>(
            ftime - fs::file_time_type::clock::now() + tm_clock::now());

}

void fillHeader(Header &header, MIPLevel &mipLevel_, PagesData &pagesData_, MIPTail &mipTail) {
    header.magicNumber[0] = 0xAB;
    header.magicNumber[1] = 'L';
    header.magicNumber[2] = 'T';
    header.magicNumber[3] = 'X';
    header.magicNumber[4] = 1; // major
    header.magicNumber[5] = 0; // minor
    header.magicNumber[6] = 0; // build
    header.magicNumber[7] = 0xBB;
    header.magicNumber[8] = '\r';
    header.magicNumber[9] = '\n';
    header.magicNumber[10] = '\x1A';
    header.magicNumber[11] = '\n';

// Заполнение остальных полей заголовка
    header.lastModified = time_lastModified(path_image);
    header.width = mipLevel_.getFirstMat().cols;
    header.height = mipLevel_.getFirstMat().rows;
    header.depth = 1; // В нашем случае всегда равна 1
    header.pageResolution[0] = pagesData_.getPageWidth();
    header.pageResolution[1] = pagesData_.getPageHeight();
    header.pageResolution[2] = 1; // Глубина всегда равна 1
    header.pageCount = pagesData_.getPageDataSize();
    header.pageSize = PAGE_SIZE;
// Вычисление количества MIP-уровней и заполнение соответствующего поля заголовка:
    header.mipLevelCount =
            std::log2(std::max(mipLevel_.getFirstMat().cols, mipLevel_.getFirstMat().rows)) + 1;
// Вычисление индекса первой страницы для каждого MIP-уровня и заполнение соответствующего поля заголовка:
    int pageIndex = 0;
    for (auto i = 0ul; i < mipLevel_.level_size(); i++) {
        header.mipLevelPageIndex[i] = pageIndex;
        pageIndex += std::ceil(static_cast<double>(mipLevel_.getFromIndex(i).cols) / pagesData_.getPageWidth()) *
                     std::ceil(static_cast<double>(mipLevel_.getFromIndex(i).rows) / pagesData_.getPageHeight());
    }

    for (auto i = 0ul; i < mipLevel_.level_size(); i++) {
        auto mipWidth = mipLevel_.getFromIndex(i).cols;
        auto mipHeight = mipLevel_.getFromIndex(i).rows;
        auto pageCount =
                std::ceil(static_cast<double>(mipWidth) / pagesData_.getPageWidth()) *
                std::ceil(static_cast<double>(mipHeight) / pagesData_.getPageHeight());
        if (pageCount <= 1) {
            mipTail.set_tail_start(i);
            break;
        }
    }
    header.mipTailStart = mipTail.get_tail_start();
    mipTail.set_tailOfSet(sizeof(Header) +
                          (header.mipLevelPageIndex[mipTail.get_tail_start()] * PAGE_SIZE));
    header.mipTailOffset = mipTail.get_tailsOfSet();

    for (u_long i = mipTail.get_tail_start(); i < mipLevel_.level_size(); i++) {
        auto mipWidth = mipLevel_.getFromIndex(i).cols;
        auto mipHeight = mipLevel_.height_ = mipLevel_.getFromIndex(i).rows;
        auto pageCount =
                std::ceil(static_cast<double>(mipWidth) / pagesData_.getPageWidth()) *
                std::ceil(static_cast<double>(mipHeight) / pagesData_.getPageHeight());
        mipTail.set_tail_size(pageCount * PAGE_SIZE);
    }
    header.mipTailSize = mipTail.get_tail_size();

    uint32_t format = 0;
    if (mipLevel_.getFirstMat().channels() == 1 && mipLevel_.getFirstMat().elemSize() == CV_8U)
        format = 0; // R8
    else if (mipLevel_.getFirstMat().channels() == 2 && mipLevel_.getFirstMat().elemSize() == CV_8U) {
        format = 1; // RG8
    } else if (mipLevel_.getFirstMat().channels() == 3 && mipLevel_.getFirstMat().elemSize() == CV_8U) {
        format = 2; // RGB8
    } else if (mipLevel_.getFirstMat().channels() == 4 && mipLevel_.getFirstMat().elemSize() == CV_8U) {
        format = 3; // RGBA8
    }

    header.format = format;
    header.flags = 0;
    header.colorCompression[0] = 0;
    header.colorCompression[1] = 0;
    header.compressionType = 0;
    header.compressionLevel = 0;
    header.dataOffset = sizeof(Header);
}




int main() {
    cv::Mat image = cv::imread(path_image, cv::IMREAD_UNCHANGED);
    if (image.empty()) {
        std::cerr << "Ошибка при загрузке изображения" << std::endl;
        return -1;
    }

// Получение информации об исходном изображении
    MIPLevel mipLevel_(std::move(image));

    auto currentWidth = mipLevel_.getWidth();
    auto currentHeight = mipLevel_.getHeight();

    while (currentWidth > 2 && currentHeight > 2) {
        currentWidth /= 2;
        currentHeight /= 2;

        cv::Mat mipLevel;
        cv::resize(mipLevel_.level_top(), mipLevel, cv::Size(currentWidth, currentHeight), 0, 0, cv::INTER_AREA);
        mipLevel_.insertImage(mipLevel);
    }

// Разбиение MIP уровней на страницы по 64кб
    PagesData pagesData_(mipLevel_.getFirstMat());

    for (auto i = 0ul; i < mipLevel_.level_size(); i++) {

        const cv::Mat &mipLevel = mipLevel_.getFromIndex(i);
        auto mipWidth = mipLevel.cols;
        auto mipHeight = mipLevel.rows;

        for (int y = 0; y < mipHeight; y += pagesData_.getPageHeight()) {

            for (int x = 0; x < mipWidth; x += pagesData_.getPageWidth()) {

                auto roiWidth = std::min(pagesData_.getPageWidth(), mipWidth - x);
                auto roiHeight = std::min(pagesData_.getPageHeight(), mipHeight - y);
                cv::Rect roi(x, y, roiWidth, roiHeight);

                cv::Mat page = mipLevel(roi);

                std::vector<uchar> pageData(PAGE_SIZE, 0);
                std::memcpy(pageData.data(), page.data, page.total() * pagesData_.getPixelSize());
                pagesData_.insert(pageData);
            }
        }
    }

    Header header = {};
    MIPTail mipTail;

    fillHeader(header, mipLevel_, pagesData_, mipTail);

    std::ofstream outputFile(path_outputLTX, std::ios::binary);

    if (!outputFile) {
        std::cerr << "Ошибка при открытии выходного файла" << std::endl;
        return -1;
    }

    outputFile.write(reinterpret_cast<const char *>(&header), sizeof(header));
    for (const auto &pageData : pagesData_.getData()) {
        outputFile.write(reinterpret_cast<const char *>(pageData.data()), pageData.size());
    }

    outputFile.close();
}
