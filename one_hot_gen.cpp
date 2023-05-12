#include <iostream>
#include <fstream>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <DBoW3/DBoW3.h>
#include <experimental/filesystem>
#include <regex>

using namespace std;
using namespace cv;

vector<string> getFilesInDirectory(const string &dirName, const string &extension) {
    vector<string> allFiles;
    if (dirName.empty() || extension.empty())
        return allFiles;

    if (!experimental::filesystem::exists(dirName) || !experimental::filesystem::is_directory(dirName))
        return allFiles;

    for (auto &file: experimental::filesystem::directory_iterator(dirName)) {
        if (file.path().extension() == extension)
            allFiles.push_back(file.path().string());
    }
    return allFiles;
}

int main(int argc, char **argv) {
    // check input arguments
    if (argc != 4) {
        cerr << "Usage: " << argv[0] << " <vocabulary_file> <database_dir> <one_hot_file>" << endl;
        return 1;
    }

    // load vocabulary from file
    const string vocab_file = argv[1];
    // ifstream in(vocab_file);
    DBoW3::Vocabulary vocab(vocab_file);
    cout << "Vocabulary information: " << endl << vocab << endl;

    auto train_images_dir =  argv[2];
    vector<string> train_images = getFilesInDirectory(train_images_dir, ".png");
    cout << "Number of database images in " << train_images_dir<< ":" << train_images.size() << endl;

    cv::Ptr<cv::Feature2D> fdetector;
    fdetector = cv::ORB::create();
    fstream one_hot_file;
    one_hot_file.open(argv[3], ios::out);
    vector<string> splits;
    auto pos = 0;
    auto path = train_images[0];
    while  ((pos = path.find('/')) != string::npos) {
        auto split = path.substr(0, pos);
        splits.push_back(split);
        path.erase(0, pos + 1);
    }

    for (const auto &train_image_file: train_images) {
        DBoW3::BowVector bow_vector;
        vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
        const Mat image = imread(train_image_file);
        if (image.empty())throw std::runtime_error("Could not open image" + train_image_file);
        fdetector->detectAndCompute(image, cv::Mat(), keypoints, descriptors);
        vocab.transform(descriptors, bow_vector);
        auto image_name = train_image_file.substr(train_image_file.find_last_of('/') + 1);
        one_hot_file << splits[5]+"/"+splits[6]+"/"+image_name << " ";
        auto one_hot_vector = vector<int> (vocab.size(), 0);
        for (auto &item: bow_vector) {
            one_hot_vector[item.first] = 1;
        }
        for (auto &item: one_hot_vector) {
            one_hot_file << item;
        }
        one_hot_file << endl;
        cout << "image: " << train_image_file << " has " << bow_vector.size() << "/" << vocab.size() << " words" << endl;
    }
    return 0;
}