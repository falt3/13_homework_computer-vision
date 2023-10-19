#include <iostream>
#include <fstream>
#include <string>

#include <mnist/tf_classifier.h>

#include <helpers.h>

using namespace mnist;

const size_t width = 28;
const size_t height = 28;
const size_t output_dim = 10;

int main(int argc, const char* argv[]) 
{
    if (argc < 3) return 1;
    std::string fn_test = argv[1];
    std::string dn_model = argv[2];

    auto clf = TfClassifier{dn_model, width, height};

    auto features = TfClassifier::features_t{};   

    std::ifstream test_data{fn_test}; 
    if (test_data.is_open()) {
        size_t countAll = 0;
        size_t countTrue = 0;

        for (;;) {
            size_t y_true;
            test_data >> y_true;
            if (!read_features_csv(test_data, features)) {
                break;
            }
            auto y_pred = clf.predict(features);
            countAll++;
            if (y_true == y_pred)
                countTrue++;
        }
        std::cout << static_cast<double>(countTrue)/countAll << "\n";
    }

    return 0;
}