#include <stdio.h>
#include <math.h>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/photo/photo.hpp"
#include "opencv2/stitching/stitcher.hpp"

#define DFUNCTION //printf("%s()\n", __FUNCTION__);

#define error(args ...) fprintf(stderr, args); \
    exit(0);

const int THRESHOLD1 = 100;
const int THRESHOLD2 = 240;

bool is_file_exists(char* name) {
    if (FILE *file = fopen(name, "r")) {
        fclose(file);
        return true;
    } else {
        return false;
    }
}

void imshow_resized(const cv::string& winname, const cv::Mat& img)
{
    cv::namedWindow(winname);
    cv::Mat smallImg;
    cv::resize(img, smallImg, cv::Size(), 0.5, 0.5);
    cv::imshow(winname, smallImg);
}

void print_help(char* name) {
    printf("Usage: ./%s <source image> <image traits> [ -o <output>] [-s transparency] [-h]\n", name);
}

void print_mat_info(const cv::Mat i, const char* name) {
    printf("%s\n", name);
    printf("Cols: %d,\nRows %d\n", i.cols, i.rows);
}

void check_and_get_args(int argc, char* argv[], std::string& src_img, std::string& mask_img, std::string& out_img, float& how_strong) {
    if (argc < 3) {
        print_help(argv[0]);
        exit(0);
    }
    if (!is_file_exists(argv[1])) {
        error("File does not exist: %s\n", argv[1]);
    }
    if (!is_file_exists(argv[2])) {
        error("File does not exist: %s\n", argv[2]);
    }

    src_img = argv[1];
    mask_img = argv[2];
    how_strong = 0.3;
    out_img = "result.jpg";


    for(int i = 3; i < argc; i++) {
        if (std::string(argv[i]) == "-h") {
            print_help(argv[0]);
            exit(0);
        } else if (std::string(argv[i]) == "-o") {
            if (i == argc - 1) {
                error("There is no output file\n");
            }

            out_img = std::string(argv[++i]) + std::string(".jpg");

            continue;
        } else if (std::string(argv[i]) == "-s") {
            if (i == argc - 1) {
                error("There is no transparency parametr!\n");
            }

            int transp = atoi(argv[++i]);
            if (transp > 100 || transp < 0) {
                error("Please enter valid transparency (from 0 to 100)\n");
            }
            how_strong = static_cast<float>(transp) / 100.0;

            continue;
        } else {
            error("No such option: %s\n", argv[i]);
        }
    }

}

cv::Mat transform_mask(const cv::Mat& mask) {
    DFUNCTION;
    cv::Mat gray(mask);
    cv::cvtColor(mask, gray, CV_BGR2GRAY);

    cv::Mat canny(gray);
    cv::Canny(gray, canny, THRESHOLD1, THRESHOLD2);

    cv::Mat canny_inv(canny);
    cv::bitwise_not (canny, canny_inv);
//    imshow_resized("im Canny inverted", canny_inv); // IF NEEDED

    cv::Mat dist(gray);
    cv::distanceTransform(canny_inv, dist, CV_DIST_L2, 3);

    return dist;
}

cv::Mat merge(const cv::Mat& src, const cv::Mat& mask, const cv::Mat& dist_mask, float how_strong) {
    DFUNCTION;
    cv::Mat res(src);

    cv::Mat fit_mask, fit_dist_mask;
    cv::resize(mask, fit_mask, cv::Size(src.cols, src.rows), 1, 1);
    cv::resize(dist_mask, fit_dist_mask, cv::Size(src.cols, src.rows), 1, 1);

    double min;
    double max;
    cv::minMaxIdx(dist_mask, &min, &max);

    for(int chanel = 0; chanel < 3; chanel++) {
        for(int i = 0; i < src.cols; i++) {
            for(int j = 0; j < src.rows; j++) {
                float distance = fit_dist_mask.at<float>(j, i);
                float coeff = pow((max - distance) / max, 3);
                coeff *= how_strong;
                res.at<cv::Vec3b>(j, i)[chanel] = (1 - coeff) * src.at<cv::Vec3b>(j, i)[chanel]
                    + fit_mask.at<cv::Vec3b>(j, i)[chanel] * coeff;
            }
        }
    }
    return res;
}

int main(int argc, char* argv[]) {
    std::string src_img, mask_img, out_img;
    float how_strong;

    check_and_get_args(argc, argv, src_img, mask_img, out_img, how_strong);

    cv::Mat src = cv::imread(src_img.c_str());
    cv::Mat mask = cv::imread(mask_img.c_str());

    cv::Mat cont_mask = transform_mask(mask);
    cv::Mat after_merge = merge(src, mask, cont_mask, how_strong);

    if (out_img == std::string("")) {
        imshow_resized("Result", after_merge);
        cv::waitKey();
    } else {
        printf("Writing file %s\n", out_img.c_str());
        cv::imwrite( out_img.c_str(), after_merge);
    }
    
    return 0;
}
