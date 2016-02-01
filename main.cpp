//
//  main.cpp
//  part3
//
//  Created by YoshimuraKazumi on 2015/10/14.
//  Copyright © 2015年 YoshimuraKazumi. All rights reserved.
//

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
//#include "Labeling.h"

int main() {
    // 顔認識の準備
    cv::CascadeClassifier cascade;
    cascade.load("haarcascade_frontalface_alt.xml"); // 顔認識のXMLファイル
    
    cv::Mat sample = cv::imread("img5.png"); // 理想の髪型になっている画像ファイル
    cv::Mat sample_hsv;
    cv::Mat org;    // カメラ画像
    cv::Mat cam;
    cv::Mat hsv;    // HSV変換後の画像
    cv::namedWindow("back");
    cv::namedWindow("video");
    
    // カメラの設定
    cv::VideoCapture capture(0);
    if (!capture.isOpened()) {
        return 0;
    }
    
    // 特徴点検出の準備
    cv::Ptr<cv::AKAZE> detector = cv::AKAZE::create();
    // 特徴点の対応付けを行うマッチャーの準備
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce");
    
    // カメラの画像サイズを調べる
    capture >> org;
    cv::resize(org, cam, cv::Size(), 0.5, 0.5);
    
    const int cam_h = cam.rows;
    const int cam_w = cam.cols;
    const int sam_h = sample.rows;
    const int sam_w = sample.cols;
    
    cv::Mat result(cam_h, cam_w, CV_8UC1);
    cv::Mat sam_result(sam_h, sam_w, CV_8UC1);
    
    sam_result = cv::Mat::zeros(sam_h, sam_w, CV_8UC1);
    cvtColor(sample, sample_hsv, CV_BGR2HSV);
    
    // 髪の毛部分の検出
    // 髪色の範囲をHSVで範囲指定
    for(int y=0; y<sam_h;y++){
        for(int x=0; x<sam_w; x++){
            int index_sam = sample_hsv.step*y+(x*3);
            if((sample_hsv.data[index_sam] >=130 || sample_hsv.data[index_sam] <=40)&& //Hの範囲指定
               //hsv.data[index+1] >= 70 &&                   // Sの範囲指定
               sample_hsv.data[index_sam+2] <= 50 ) {               //Vの範囲指定
                sam_result.data[sam_result.step*y+x] = 255;              // 髪色の該当色範囲を白くする
            }
        }
    }
    
    std::vector<cv::KeyPoint> keypoints_sam;
    detector->detect(sam_result, keypoints_sam);
    cv::Mat descriptor_sam;
    detector->compute(sam_result, keypoints_sam, descriptor_sam);
    
    while(true){
        // カメラ画像の取得
        capture >> org;
        cv::resize(org, cam, cv::Size(), 0.5, 0.5);
        
        //結果配列の初期化
        result = cv::Mat::zeros(cam_h, cam_w, CV_8UC1);
        
        //HSV画像への変換
        cvtColor(cam, hsv, CV_BGR2HSV);
        if(cam.empty()) return -1;
        cv::Mat dst_img = cam.clone();
        
        // 顔検出の処理
        std::vector<cv::Rect> faces_cam;
        cascade.detectMultiScale(dst_img, faces_cam, 1.1, 3, 0, cv::Size(20, 20));
        
        
        // 以下のコメント内では、髪の毛に似た色だけれど髪の毛ではない物体が出てきたときに無視するための処理を書いていますが、
        // 実行したら処理が重く、度々止まってしまうことがあったのでコメントアウトしました。
        // この処理を実行してプログラムを動かす時には、ここのコメントアウトを外した後、このさらに下の部分をコメントアウトしてから実行して下さい。
        
//        for (int i = 0; i < faces_cam.size(); i++){
//            int roi_x = faces_cam[i].x-((faces_cam[i].width * 1.5-faces_cam[i].width)/2);
//            int roi_y = (faces_cam[i].y-(faces_cam[i].height * 1.8-faces_cam[i].height))+(faces_cam[i].height * 1.8-faces_cam[i].height)/8;
//            int roi_w = faces_cam[i].width*1.5;
//            int roi_h = faces_cam[i].height*1.8;
//            if(roi_x<0){
//                roi_w += roi_x;
//                roi_x = 0;
//            }
//            if(roi_y<0){
//                roi_h += roi_y;
//                roi_y = 0;
//            }
//            if (roi_x + roi_w>cam_w){
//                roi_w = cam_w-roi_x;
//            }
//            if (roi_y + roi_h>cam_h){
//                roi_h = cam_h-roi_y;
//            }
//            //cv::Rect roi_rect(faces[i].x,faces[i].y,faces[i].width,faces[i].height);
//            cv::Rect roi_rect(roi_x,roi_y,roi_w,roi_h);
//            cv::Mat src_roi = cam(roi_rect);
//            cv::Mat dst_roi = dst_img(roi_rect);
//        
//            for(int y=roi_y; y<roi_y + roi_h;y++){
//                for(int x=roi_x; x<roi_x + roi_w; x++){
//                    int index = hsv.step*y+(x*3);
//                    if((hsv.data[index] >=130 || hsv.data[index] <=40)&&  //Hの範囲指定
//                       //hsv.data[index+1] >= 70 &&                   //Sの範囲指定
//                       hsv.data[index+2] <= 50 ) {               //Vの範囲指定
//                        result.data[result.step*y+x] = 255;              //該当色範囲を白くする
//                    }
//                }
//            }
//        }
        
        // ↑を実行する時は、下の範囲をコメントアウトして下さい。
        
        //// --- ここから ---
        for(int y=0; y<cam_h;y++){
            for(int x=0; x<cam_w; x++){
                int index = hsv.step*y+(x*3);
                if((hsv.data[index] >=130 || hsv.data[index] <=40)&& //Hの範囲指定
                   //hsv.data[index+1] >= 70 &&                   //Sの範囲指定
                   hsv.data[index+2] <= 50 ) {               //Vの範囲指定
                    result.data[result.step*y+x] = 255;              //該当色範囲を白くする
                }
            }
        }
        ////  --- ここまで ---
        
        
        // 特徴点の検出
        std::vector<cv::KeyPoint> keypoints_cam;
        detector->detect(result, keypoints_cam);
        // 特徴点の算出
        cv::Mat descriptor_cam;
        detector->compute(result, keypoints_cam, descriptor_cam);
    
        // 特徴点の対応付け
        std::vector<cv::DMatch> matches;
        matcher->match(descriptor_sam, descriptor_cam, matches);
        
        int N=100;
        nth_element(matches.begin(), matches.begin()+N-1, matches.end());
        matches.erase(matches.begin()+N, matches.end());
        
        // 特徴点による類似度を計算
        int votes=0;
        for(int i = 0; i < matches.size(); i++){
            if(matches[i].distance < 80){
                votes++;
            }
        }
        //int v = votes/matches.size();
        std::cout << votes <<"¥n";
        
        // 結果の描画と表示
        cv::Mat imgResult;
        cv::drawMatches(sam_result, keypoints_sam, result, keypoints_cam, matches, imgResult);
        cv::imshow("back", imgResult);
//        cv::imshow("back", result);
        int font = cv::FONT_HERSHEY_PLAIN;
        
        // 髪型がOKならキャプチャを保存 & 画面に"Saved"と表示
        if(faces_cam.size()>0 && votes > 55){
            cv::imwrite("file01.png", cam);
            cv::putText(cam, "Saved", cv::Point(50,100), font, 5, cv::Scalar(0,200,0), 2, CV_AA);
        }
        cv::imshow("video", cam);
        
        int key = cv::waitKey(30);
        if (key == 'q') {
            // キャプチャから画像データを読み込む // 画像をウィンドウに表示
            // キー入力待ち(30ms)
            // 入力されたキーが「q」だったら
            // ループを抜ける
        }
        
    }
}
