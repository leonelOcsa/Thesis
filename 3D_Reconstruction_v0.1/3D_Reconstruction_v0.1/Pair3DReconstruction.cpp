#include "stdafx.h"
#include "Pair3DReconstruction.h"

//FUNDAMENTAL MATRIX ESTIMATOR

class FundamentalMatrixSevenPointEstimator {
public:
	typedef Eigen::Vector2d X_t;
	typedef Eigen::Vector2d Y_t;
	typedef Eigen::Matrix3d M_t;

	// The minimum number of samples needed to estimate a model.
	static const int kMinNumSamples = 7;

	// Estimate either 1 or 3 possible fundamental matrix solutions from a set of
	// corresponding points.
	//
	// The number of corresponding points must be exactly 7.
	//
	// @param points1  First set of corresponding points.
	// @param points2  Second set of corresponding points
	//
	// @return         Up to 4 solutions as a vector of 3x3 fundamental matrices.
	static std::vector<M_t> Estimate(const std::vector<X_t>& points1,
		const std::vector<Y_t>& points2);

	// Calculate the residuals of a set of corresponding points and a given
	// fundamental matrix.
	//
	// Residuals are defined as the squared Sampson error.
	//
	// @param points1    First set of corresponding points as Nx2 matrix.
	// @param points2    Second set of corresponding points as Nx2 matrix.
	// @param F          3x3 fundamental matrix.
	// @param residuals  Output vector of residuals.
	static void Residuals(const std::vector<X_t>& points1,
		const std::vector<Y_t>& points2, const M_t& F,
		std::vector<double>* residuals) {
		ComputeSquaredSampsonError_(points1, points2, F, residuals);
	}
};

std::vector<FundamentalMatrixSevenPointEstimator::M_t>
FundamentalMatrixSevenPointEstimator::Estimate(
	const std::vector<X_t>& points1, const std::vector<Y_t>& points2) {
	//CHECK_EQ(points1.size(), 7);
	//CHECK_EQ(points2.size(), 7);

	// Note that no normalization of the points is necessary here.

	// Setup system of equations: [points2(i,:), 1]' * F * [points1(i,:), 1]'.
	Eigen::Matrix<double, 7, 9> A;
	for (size_t i = 0; i < 7; ++i) {
		const double x0 = points1[i](0);
		const double y0 = points1[i](1);
		const double x1 = points2[i](0);
		const double y1 = points2[i](1);
		A(i, 0) = x1 * x0;
		A(i, 1) = x1 * y0;
		A(i, 2) = x1;
		A(i, 3) = y1 * x0;
		A(i, 4) = y1 * y0;
		A(i, 5) = y1;
		A(i, 6) = x0;
		A(i, 7) = y0;
		A(i, 8) = 1;
	}

	// 9 unknowns with 7 equations, so we have 2D null space.
	Eigen::JacobiSVD<Eigen::Matrix<double, 7, 9>> svd(A, Eigen::ComputeFullV);
	const Eigen::Matrix<double, 9, 9> f = svd.matrixV();
	Eigen::Matrix<double, 1, 9> f1 = f.col(7);
	Eigen::Matrix<double, 1, 9> f2 = f.col(8);

	f1 -= f2;

	// Normalize, such that lambda + mu = 1
	// and add constraint det(F) = det(lambda * f1 + (1 - lambda) * f2).

	const double t0 = f1(4) * f1(8) - f1(5) * f1(7);
	const double t1 = f1(3) * f1(8) - f1(5) * f1(6);
	const double t2 = f1(3) * f1(7) - f1(4) * f1(6);
	const double t3 = f2(4) * f2(8) - f2(5) * f2(7);
	const double t4 = f2(3) * f2(8) - f2(5) * f2(6);
	const double t5 = f2(3) * f2(7) - f2(4) * f2(6);

	Eigen::Vector4d coeffs;
	coeffs(0) = f1(0) * t0 - f1(1) * t1 + f1(2) * t2;
	coeffs(1) = f2(0) * t0 - f2(1) * t1 + f2(2) * t2 -
		f2(3) * (f1(1) * f1(8) - f1(2) * f1(7)) +
		f2(4) * (f1(0) * f1(8) - f1(2) * f1(6)) -
		f2(5) * (f1(0) * f1(7) - f1(1) * f1(6)) +
		f2(6) * (f1(1) * f1(5) - f1(2) * f1(4)) -
		f2(7) * (f1(0) * f1(5) - f1(2) * f1(3)) +
		f2(8) * (f1(0) * f1(4) - f1(1) * f1(3));
	coeffs(2) = f1(0) * t3 - f1(1) * t4 + f1(2) * t5 -
		f1(3) * (f2(1) * f2(8) - f2(2) * f2(7)) +
		f1(4) * (f2(0) * f2(8) - f2(2) * f2(6)) -
		f1(5) * (f2(0) * f2(7) - f2(1) * f2(6)) +
		f1(6) * (f2(1) * f2(5) - f2(2) * f2(4)) -
		f1(7) * (f2(0) * f2(5) - f2(2) * f2(3)) +
		f1(8) * (f2(0) * f2(4) - f2(1) * f2(3));
	coeffs(3) = f2(0) * t3 - f2(1) * t4 + f2(2) * t5;

	Eigen::VectorXd roots_real;
	Eigen::VectorXd roots_imag;
	if (!FindPolynomialRootsCompanionMatrix_(coeffs, &roots_real, &roots_imag)) {
		return {};
	}

	std::vector<M_t> models;
	models.reserve(roots_real.size());

	for (Eigen::VectorXd::Index i = 0; i < roots_real.size(); ++i) {
		const double kMaxRootImag = 1e-10;
		if (std::abs(roots_imag(i)) > kMaxRootImag) {
			continue;
		}

		const double lambda = roots_real(i);
		const double mu = 1;

		Eigen::MatrixXd F = lambda * f1 + mu * f2;

		F.resize(3, 3);

		const double kEps = 1e-10;
		if (std::abs(F(2, 2)) < kEps) {
			continue;
		}

		F /= F(2, 2);

		models.push_back(F.transpose());
	}

	return models;
}

//****************************************************************************************************//


Pair3DReconstruction::Pair3DReconstruction()
{
}

Pair3DReconstruction::Pair3DReconstruction(string src1, string src2) {
	source1 = src1;
	source2 = src2;
	sift = xfeatures2d::SIFT::create();
	InitRGBPairs();
	InitGRAYPairs();
	matcher = DescriptorMatcher::create("FlannBased");
	//init with K from calibration, be careful because this calibration is for Logitech C525
	cameraMatrix = Mat::zeros(3, 3, CV_64FC1);
	cameraMatrix.at<double>(0, 0) = 7.3129070817560148e+02;
	cameraMatrix.at<double>(0, 1) = 0.;
	cameraMatrix.at<double>(0, 2) = 3.1950000000000000e+02;
	cameraMatrix.at<double>(1, 0) = 0.;
	cameraMatrix.at<double>(1, 1) = 7.3202468183635153e+02;
	cameraMatrix.at<double>(1, 2) = 2.3950000000000000e+02;
	cameraMatrix.at<double>(2, 0) = 0.;
	cameraMatrix.at<double>(2, 1) = 0.;
	cameraMatrix.at<double>(2, 2) = 1.;
}

void Pair3DReconstruction::InitRGBPairs() {
	RGB_im1 = imread(source1);
	RGB_im2 = imread(source2); 
}
void Pair3DReconstruction::InitGRAYPairs() {
	GRAY_im1 = imread(source1, CV_LOAD_IMAGE_GRAYSCALE); 
	GRAY_im2 = imread(source2, CV_LOAD_IMAGE_GRAYSCALE);
}

Mat Pair3DReconstruction::getRGBImage1() {
	return RGB_im1;
}
Mat Pair3DReconstruction::getRGBImage2() {
	return RGB_im2;
}
Mat Pair3DReconstruction::getGRAYImage1() {
	return GRAY_im1;
}
Mat Pair3DReconstruction::getGRAYImage2(){
	return GRAY_im2;
}

vector<KeyPoint> Pair3DReconstruction::getKeyPoints1() {
	return kp1;
}
vector<KeyPoint> Pair3DReconstruction::getKeyPoints2() {
	return kp2;
}

Mat Pair3DReconstruction::getFundamentalMatrix() {
	return fundamentalMatrix;
}

Mat Pair3DReconstruction::getEssentialMatrix() {
	return essentialMatrix;
}

Mat Pair3DReconstruction::getCameraMatrix() {
	return cameraMatrix;
}

void Pair3DReconstruction::calculateKeyPoints() {
	sift->detect(GRAY_im1, kp1);
	sift->detect(GRAY_im2, kp2);
}

void Pair3DReconstruction::calculateDescriptors() {
	sift->compute(GRAY_im1, kp1, descriptors1);
	sift->compute(GRAY_im2, kp2, descriptors2);
}

void Pair3DReconstruction::calculateMatching() {
	vector<vector<DMatch>> matches;
	vector< vector<DMatch> > matches12, matches21;
	matcher->knnMatch(descriptors1, descriptors2, matches12, 2);
	matcher->knnMatch(descriptors2, descriptors1, matches21, 2);
	// ratio test proposed by David Lowe paper = 0.8
	const float ratio = 0.8;
	for (int i = 0; i < matches12.size(); i++) {
		if (matches12[i][0].distance < ratio * matches12[i][1].distance) {
			good_matches.push_back(matches12[i][0]);
			points2.push_back(kp2[matches12[i][0].trainIdx].pt); //almacenamos los keypoints que hacen match en points
			points1.push_back(kp1[matches12[i][0].queryIdx].pt);
		}
	}
}

void Pair3DReconstruction::drawKeyPoints(Mat img, vector<KeyPoint> kp, string im_name) {
	Mat kp_im;
	drawKeypoints(img, kp, kp_im, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	imshow(im_name, kp_im);
}

void Pair3DReconstruction::drawMatching(bool stepByStep) {
	Mat img_matches;
	if (stepByStep == true) {
		for (int i = 0; i < good_matches.size(); i++) {
			vector<DMatch> sub_good_matches(good_matches.begin() + i, good_matches.begin() + i + 1);
			drawMatches(GRAY_im1, kp1, GRAY_im2, kp2, sub_good_matches, img_matches, Scalar::all(-1), Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
			imshow("SIFT MATCHER FLANN BASED", img_matches);
			waitKey(0);
		}
	}
	else {
		drawMatches(GRAY_im1, kp1, GRAY_im2, kp2, good_matches, img_matches, Scalar::all(-1), Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
		imshow("FLANN-Matcher SIFT Matches", img_matches);
	}
}

void Pair3DReconstruction::doPointsFiltering() {	
	Mat mask;
	fundamentalMatrix = findFundamentalMat(Mat(points1), Mat(points2), CV_FM_7POINT, 3, 0.99, mask);  // usando el método de 8 PUNTOS
	
	Mat mr;
	mask.convertTo(mr, CV_8UC1);
	//seleccionamos solo los puntos inliers
	std::vector<Point2f> inlierPoints1, inlierPoints2;
	mask.convertTo(mask, CV_8UC1);
	for (int i = 0; i < mask.rows; i++) {
		if (mask.at<uchar>(i, 0) == 1) {
			inlierPoints1.push_back(points1[i]);
			inlierPoints2.push_back(points2[i]);
			eigen_points1.push_back(Eigen::Vector2d(points1[i].x, points1[i].y));
			eigen_points2.push_back(Eigen::Vector2d(points2[i].x, points2[i].y));
		}
	}

	points1.clear();
	points2.clear();
	points1 = inlierPoints1;
	points2 = inlierPoints2;
}

void Pair3DReconstruction::calculateFundamentalMatrix() {
	vector<FundamentalMatrixSevenPointEstimator::M_t> models = FundamentalMatrixSevenPointEstimator::Estimate(eigen_points1, eigen_points2);
	for (int i = 0; i < models.size(); i++) {
		fundamentalMatrix.at<double>(i, 0) = (models[i](0));
		fundamentalMatrix.at<double>(i, 1) = (models[i](1));
		fundamentalMatrix.at<double>(i, 2) = (models[i](2));
	}
}

void Pair3DReconstruction::calculateEssentialMatrix() {
	//E = transpose(K)*F*K
	essentialMatrix = cameraMatrix.t()*fundamentalMatrix*cameraMatrix;
}

Pair3DReconstruction::~Pair3DReconstruction()
{
}
