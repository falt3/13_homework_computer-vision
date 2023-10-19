#include <fstream>

#include <gtest/gtest.h>

#include <mnist/tf_classifier.h>

#include <helpers.h>

using namespace mnist;

const size_t width = 28;
const size_t height = 28;
const size_t output_dim = 10;


TEST(TfClassifier, predict_class) {
     auto clf = TfClassifier{"model", width, height};

    auto features = TfClassifier::features_t{};


    std::ifstream test_data{"test/test_data_cnn.txt"};
    ASSERT_TRUE(test_data.is_open());
    for (;;) {
        size_t y_true;
        test_data >> y_true;
        if (!read_features(test_data, features)) {
            break;
        }
        auto y_pred = clf.predict(features);
        ASSERT_EQ(y_true, y_pred);
    }
}