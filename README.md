# image retrieval demo

This repo is a minimal demo for image retrieval/loop closure detection. It is based on the [DBoW3](https://github.com/rmsalinas/DBow3) project.

## Prerequisites
The code has been tested on MacOS(Apple silicon) with clang-1316.0.21.2.5. Ubuntu should also work.  You need to install the following libraries:
### OpenCV 3.4.16 with contrib modules
We use [OpenCV](http://opencv.org/) to extract keypoints and descriptors. The [contrib modules](https://github.com/opencv/opencv_contrib) are required. Just follow the [official guide](https://docs.opencv.org/3.4.16/d7/d9f/tutorial_linux_install.html) to install OpenCV with contrib modules. 

```
Create a temporary directory, which we denote as <cmake_build_dir>, where you want to put the generated Makefiles, project files as well the object files and output binaries and enter there.

For example

cd ~/opencv
mkdir build
cd build
Configuring. Run cmake [<some optional parameters>] <path to the OpenCV source directory>

For example

cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr/local ..

to build with modules from opencv_contrib set OPENCV_EXTRA_MODULES_PATH to <path to opencv_contrib/modules/>
```

### DBoW3

We use [DBoW3](https://github.com/rmsalinas/DBow3) to build the bag-of-words vocabulary, encode images and perform image retrieval.

DBoW3 requires OpenCV only.  As described in official guide, we install DBoW3 as follow: 

```
git clone https://github.com/rmsalinas/DBow3.git

cd DBow3
mkdir build
cd build/
cmake ..
make
sudo make install
```

## Build
Clone the repository and use the build script to build the project.
```shell
sh build.sh
```
## How to use
There are two main programs in the project: `make_voc` and `query`.
### make_voc
This program builds the bag-of-words vocabulary from a set of images. The vocabulary is saved to a file. The program takes two arguments: the path to a text file containing the paths of images and the path to the output voc file.
```shell
make_voc <images_txt> <vocabulary_output_file>
```
### query
This program performs image retrieval based on a given vocabulary. It takes Two arguments: the path to the vocabulary file and the path to the database images directory. The program will load all images in that directory to build a database and you can query any image in the database. The program will show the top 4 retrieved images and their scores(the top 1 is the query image itself and the score is 1). You can input `exit` to quit the program.
```shell
query <vocabulary_file> <database_dir>
```
## data
We provide a small vocabulary trained on 10 sequences of [KITTI](https://www.cvlibs.net/datasets/kitti/) dataset named `KITTI_voc.yml.gz` in `data/`. You can use it to test the program. There are also a large vocabulary `orbvoc.dbow3` provided by [DBoW3](https://github.com/rmsalinas/DBow3).

We also use `one_hot_gen` to  transform the images in sequence 02 and 05 of KITTI dataset to one-hot encoding using the vocabulary `KITTI_voc.yml.gz`. The one-hot encoding images are saved in `data/02.txt` and `data/05.txt`. Each line in the text file is the image path and the one-hot encoding(10000 dimension) of the image.