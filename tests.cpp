#include <gtest/gtest.h>
#include "main.cpp"
#include <opencv2/opencv.hpp>


// Тест для проверки корректности работы конструктора класса MIPLevel
TEST(MIPLevelTest, ConstructorTest) {
    cv::Mat image = cv::imread(path_image, cv::IMREAD_UNCHANGED);
    MIPLevel mipLevel(std::move(image));

    EXPECT_EQ(mipLevel.getWidth(), image.cols);
    EXPECT_EQ(mipLevel.getHeight(), image.rows);
    EXPECT_EQ(mipLevel.getChannels(), image.channels());
    EXPECT_EQ(mipLevel.getDepth(), image.elemSize());
}

// Тест для проверки корректности работы метода insertImage класса MIPLevel
TEST(MIPLevelTest, InsertImageTest) {
    cv::Mat image = cv::imread(path_image, cv::IMREAD_UNCHANGED);
    MIPLevel mipLevel(std::move(image));

    cv::Mat newImage;
    cv::resize(image, newImage, cv::Size(image.cols / 2, image.rows / 2), 0, 0, cv::INTER_AREA);
    mipLevel.insertImage(newImage);

    EXPECT_EQ(mipLevel.level_size(), 2);
}

// Тест для проверки корректности работы методов класса MIPTail
TEST(MIPTailTest, MethodsTest) {
    MIPTail mipTail;
    mipTail.set_tail_start(3);
    mipTail.set_tailOfSet(100);
    mipTail.set_tail_size(50);

    EXPECT_EQ(mipTail.get_tail_start(), 3);
    EXPECT_EQ(mipTail.get_tailsOfSet(), 100);
    EXPECT_EQ(mipTail.get_tail_size(), 50);
}

// Тест для проверки корректности работы конструктора класса PagesData
TEST(PagesDataTest, ConstructorTest) {
    cv::Mat image = cv::imread(path_image, cv::IMREAD_UNCHANGED);
    PagesData pagesData(image);

    EXPECT_EQ(pagesData.getPixelSize(), image.channels() * image.elemSize());
}


