#pragma once

/*
Author: Leonel Ocsa S�nchez
Masther Thesis 3D reconstruction

3D reconstruction is one of my dreams since I was 14 years old, by that time I got my first console "the one and only second handed PSX"
and I discover one of the greastest games I've ever played: the Resident Evil Saga, at that momment I started playing RE3
and I falled in love. By that time I wonder how a great master piece game was made, how? I didn't know! but someday 
when me and my cousin went to buy some new games I started a conversation saying things like "Hey Manuel, can you imagine a zombie game
set in Arequipa City? Wow that would be awesome! Yes I know, I wish one day I can made a game like Resident Evil but set in Arequipa =)"
I remmenber walking down the street in front of the Santa Catalina Monastery... just imagining a resident evil game set in that part of
the city was a fantastic imagination experience for me, so I guess at that momment I wondered how can I model all the enviroment,
but now I have the oportunity of doing that, that's one part of my dream: doing a 3D reconstruction from my city!!! I know I'm not the best, and I KNOW this master thesis 
research involves a lot A LOT of knowledge, sometimes it looks impossible, sometimes it looks good, sometimes it makes me give up
but at the end I'm here to accomplish my dream, being a master on 3D reconstruction and a game developer... that's all
This lines of code are dedicated to all the people that always try to raise my spirit when is down and to all the people that didn't believe
what I can do, to all the people that always look at me and thinks I'm a rookie... always being humble!
*/

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2\calib3d\calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2\features2d\features2d.hpp>
#include <opencv2\xfeatures2d\nonfree.hpp>
//#include <opencv2/flann/miniflann.hpp> //FLANN
#include <opencv2/dnn/dict.hpp>

#include <string>
#include <array>

#include <Eigenvalues>
#include <Eigen>

#include "EssentialMatrixEstimator.h"
//#include "types.h"

using namespace std;
using namespace cv;

// Define non-copyable or non-movable classes.
#define NON_COPYABLE(class_name)          \
  class_name(class_name const&) = delete; \
  void operator=(class_name const& obj) = delete;
#define NON_MOVABLE(class_name) class_name(class_name&&) = delete;

namespace Eigen {

	typedef Eigen::Matrix<double, 3, 4> Matrix3x4d;
	typedef Eigen::Matrix<uint8_t, 3, 1> Vector3ub;
	typedef Eigen::Matrix<uint8_t, 4, 1> Vector4ub;
	typedef Eigen::Matrix<double, 6, 1> Vector6d;

}

class Pair3DReconstruction{
private:
	string source1; //source image 1
	string source2; //source image 2
	Ptr<xfeatures2d::SIFT> sift; //SIFT
	Mat RGB_im1; //RGB color image 1
	Mat RGB_im2; //RGB color image 2
	Mat GRAY_im1; //gray scale image 1
	Mat GRAY_im2; //gray scale image 2
	vector<KeyPoint> kp1; //keypoints from image 1
	vector<KeyPoint> kp2; //keypoint from image 2
	Mat descriptors1; //descriptors from image 1 
	Mat descriptors2; //descriptors from image 2
	Ptr<DescriptorMatcher> matcher; //matcher of points
	vector<DMatch> good_matches; //good matches after filtering
	vector<Point2f> points1; //matched points from image 1
	vector<Point2f> points2; //matched points from image 2
	vector<Eigen::Vector2d> eigen_points1;
	vector<Eigen::Vector2d> eigen_points2;
	Mat fundamentalMatrix; //fundamental matrix F 
	Mat essentialMatrix; //essential matrix E
	Mat cameraMatrix; //camera Matrix K
	void InitRGBPairs(); //RGB pair image initialization
	void InitGRAYPairs(); //GRAY SCALE pait image initialization
public:
	Pair3DReconstruction();
	Pair3DReconstruction(string src1, string src2);
	Mat getRGBImage1();
	Mat getRGBImage2();
	Mat getGRAYImage1();
	Mat getGRAYImage2();
	Mat getFundamentalMatrix();
	Mat getEssentialMatrix();
	Mat getCameraMatrix();
	vector<KeyPoint> getKeyPoints1();
	vector<KeyPoint> getKeyPoints2();
	vector<Eigen::Vector2d> getPoints1();
	vector<Eigen::Vector2d> getPoints2();
	vector<Point2f> getOPoints1();
	vector<Point2f> getOPoints2();
	void calculateKeyPoints();
	void calculateDescriptors();
	void calculateMatching();
	void drawKeyPoints(Mat img, vector<KeyPoint> kp, string im_name);
	void drawMatching(bool stepByStep); //if true draws the matching step by step
	void doPointsFiltering(); //filter the points obtained after matching giving us only the inlier points, for this I use the fundamental Matrix  calculation from OpenCV
	void calculateFundamentalMatrix();
	void calculateEssentialMatrix();
	//Descomposition of E getting 4 new values R1, R2, t1 and t2
	void DecomposeEssentialMatrix(const Eigen::Matrix3d& E, Eigen::Matrix3d* R1, Eigen::Matrix3d* R2, Eigen::Vector3d* t);
	//pose calculation from Essential Matrix
	void PoseFromEssentialMatrix(const Eigen::Matrix3d& E,
		const std::vector<Eigen::Vector2d>& points1,
		const std::vector<Eigen::Vector2d>& points2,
		Eigen::Matrix3d* R, Eigen::Vector3d* t,
		std::vector<Eigen::Vector3d>* points3D);

	bool CheckCheirality(const Eigen::Matrix3d& R, const Eigen::Vector3d& t,
		const std::vector<Eigen::Vector2d>& points1,
		const std::vector<Eigen::Vector2d>& points2,
		std::vector<Eigen::Vector3d>* points3D);
	
	Eigen::Matrix3x4d ComposeProjectionMatrix(const Eigen::Matrix3d& R, const Eigen::Vector3d& T);
	Eigen::Vector3d TriangulatePoint(const Eigen::Matrix3x4d& proj_matrix1,
		const Eigen::Matrix3x4d& proj_matrix2,
		const Eigen::Vector2d& point1,
		const Eigen::Vector2d& point2);
	double CalculateDepth(const Eigen::Matrix3x4d& proj_matrix, const Eigen::Vector3d& point3D);

	//Eigen::Matrix3d QuaternionToRotationMatrix(const Eigen::Vector4d& qvec);
	//Eigen::Vector4d NormalizeQuaternion(const Eigen::Vector4d& qvec);
	~Pair3DReconstruction();
};

