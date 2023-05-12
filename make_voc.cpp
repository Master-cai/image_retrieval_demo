#include <iostream>
#include <fstream>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <DBoW3/DBoW3.h>
#include <regex>
#include <experimental/filesystem>
#include <chrono>
#include <thread>
#include <mutex>

using namespace std;
using namespace cv;
using namespace DBoW3;

/*!
     * @brief get Files In Directory
     * @param dirName directory name containing images
     * @param extension file extension of images(ex: .png, .jpg)
     * @return all files in the directory with the specified extension
     * @note if dirName or extension is empty, empty vector will be returned
     * @note if dirName is not a directory or does not exist, empty vector will be returned
     */
vector<string> getFilesInDirectory(const string& dirName, const string& extension)
{
    vector<string> allFiles;
    if (dirName.empty() || extension.empty())
        return allFiles;

    if (!experimental::filesystem::exists(dirName) || !experimental::filesystem::is_directory(dirName))
        return allFiles;

    for (auto &file : experimental::filesystem::directory_iterator(dirName))
    {
        if (file.path().extension() == extension)
            allFiles.push_back(file.path().string());
    }
    return allFiles;
}
/**
 * @brief get Files In Txt
 * @param Text_path: path to txt file containing image paths (one image path per line)
 * @return all files in the txt file
 * @note if Text_path is empty, empty vector will be returned
 */
vector<string> getFilesInTxt(const string& Text_path)
{
    vector<string> allFiles;
    if (Text_path.empty())
        return allFiles;

    ifstream in(Text_path);
    string line;
    while (getline(in, line))
    {
        allFiles.push_back(line);
    }
    return allFiles;
}

/**
 * @brief Create a vocabulary from a set of images features
 * @param features vector of images features
 * @param k branching factor
 * @param L depth levels
 * @return vocabulary
 */
DBoW3::Vocabulary VocCreation(const vector<cv::Mat> &features, const int k=9, const int L=3)
{
    // branching factor and depth levels
    const WeightingType weight = TF_IDF;
    const ScoringType score = L1_NORM;

    DBoW3::Vocabulary voc(k, L, weight, score);

    cout << "Creating a small " << k << "^" << L << " vocabulary..." << endl;
    voc.create(features);
    cout << "... done!" << endl;
    // return the vocabulary
    return voc;
}

int main(int argc, char** argv) {
    // check input arguments
    if (argc != 3) {
        cerr << "Usage: " << argv[0] << " <images_txt/images_dir> <vocabulary_output_file>" << endl;
        return 1;
    }

    // set vocabulary parameters
    const int k = 10; // branching factor
    const int L = 4;  // depth levels

    vector<cv::Mat> features;
    cv::Ptr<cv::Feature2D> fdetector;
    fdetector = cv::ORB::create();
    vector<string> train_images;

    // read training images
    train_images = getFilesInTxt(argv[1]);
    cout << "Number of training images: " << train_images.size() << endl;

    auto start_time = clock();
    std::mutex mtx; // mutex
    int num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads(num_threads);
    int image_per_thread = (int)std::ceil((float)train_images.size() / num_threads);
    for (int t = 0; t < num_threads; t++) {
        threads[t] = std::thread([&, t]() {
            int start = t * image_per_thread;
            int end = std::min(start + image_per_thread, (int)train_images.size());
            for (int i = start; i < end; i++) {
                vector<cv::KeyPoint> keypoints;
                cv::Mat descriptors;
                const Mat image = imread(train_images[i]);
                if (image.empty())throw std::runtime_error("Could not open image" + train_images[i]);
                fdetector->detectAndCompute(image, cv::Mat(), keypoints, descriptors);
                features.push_back(descriptors);
                cout << "Extracted " << train_images[i] << "[" << i << "/" << train_images.size() << "]" << endl;
                std::lock_guard<std::mutex> lock(mtx); // lock feature vector
                features.emplace_back(descriptors);
            }
        });
    }
    for (auto& t : threads) t.join();
    auto end_time = clock();
    cout << "Time taken for feature extraction: " << (end_time - start_time) / (double) CLOCKS_PER_SEC << "s" << endl;

    // create vocabulary
    start_time = clock();
    DBoW3::Vocabulary vocab = VocCreation(features, k, L);
    end_time = clock();
    cout << "Time taken for vocabulary creation: " << (end_time - start_time) / (double) CLOCKS_PER_SEC << "s" << endl;
    // save to disk
    const string vocab_output_file = argv[2];
    cout << endl << "Saving vocabulary..." << endl;
    vocab.save(vocab_output_file);
    cout << "Done" << endl;
    return 0;
}