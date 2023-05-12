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
    if (argc != 3) {
        cerr << "Usage: " << argv[0] << " <vocabulary_file> <database_dir>" << endl;
        return 1;
    }

    // load vocabulary from file
    const string vocab_file = argv[1];
    // ifstream in(vocab_file);
    DBoW3::Vocabulary vocab(vocab_file);
    cout << "Vocabulary information: " << endl << vocab << endl;
    DBoW3::Database db(vocab, true, 3);

    auto train_images_dir =  argv[2];
    vector<string> train_images = getFilesInDirectory(train_images_dir, ".png");
    cout << "Number of database images in " << train_images_dir<< ":" << train_images.size() << endl;

    cv::Ptr<cv::Feature2D> fdetector;
    fdetector = cv::ORB::create();
    for (const auto &train_image_file: train_images) {
        cout << "processing image: " << train_image_file << endl;
        vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
        const Mat image = imread(train_image_file);
        if (image.empty())throw std::runtime_error("Could not open image" + train_image_file);
        fdetector->detectAndCompute(image, cv::Mat(), keypoints, descriptors);
        db.add(descriptors);
    }

    // receive test image path from command line
    string test_image_file;
    while (true) {
        std::cout << "please input image full path('exit' to quit): ";
        std::getline(std::cin, test_image_file);
        if (test_image_file == "exit") {
            break;
        }

        const Mat test_image = imread(test_image_file);

        // extract features from test image
        vector<KeyPoint> keypoints;
        Mat descriptors;
        fdetector->detectAndCompute(test_image, Mat(), keypoints, descriptors);

        // query vocabulary with test image descriptors
        DBoW3::BowVector bow_vector;
//        bow_vector
        vocab.transform(descriptors, bow_vector);
        cout << "bow_vector size: " << bow_vector.size()<< endl << bow_vector << endl;

        // get ranked list of similar images
        DBoW3::QueryResults ret;
        db.query(descriptors, ret, 4); // max result=4

        // display ranked list
        cout << "Rank  Image  Score" << endl;
        for (int i = 0; i < ret.size(); i++) {
            cout << i << "     " << ret[i].Id << "     " << ret[i].Score  << "     " << train_images[ret[i].Id] << endl;
        }


        cv::startWindowThread();
        cv::imshow("query image", test_image);
        for(int i = 0; i < ret.size(); i++) {
            auto results_image = imread(train_images[ret[i].Id]);
            cv::imshow("retrieval "+ to_string(i) + " image", results_image);
        }
        cv::waitKey(0);
        cv::destroyAllWindows();
        waitKey(1);
        db.query(bow_vector, ret, 4);
        cout << "Rank  Image  Score" << endl;
        for (int i = 0; i < ret.size(); i++) {
            cout << i << "     " << ret[i].Id << "     " << ret[i].Score  << "     " << train_images[ret[i].Id] << endl;
        }
    }
}