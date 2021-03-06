#include "stdafx.h"

#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include <stdlib.h>
#include <time.h>

#include "EssentialMatrixEstimator.h"
#include "ransac.h"

#include "Pair3DReconstruction.h" //

#define FLANN_INDEX_KDTREE 0

using namespace cv;
using namespace std;

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
//
extern "C" void __declspec(dllexport) __stdcall Hola()
{
	cout << "Hola" << endl;
}
//

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


Mat skewMat(Mat ep) { //ep es de dimension 3 x 1
	Mat epT;
	transpose(ep, epT);
	cout << ep.size() << " ep " << ep.depth() << endl;
	cout << epT.size() << " epT " << epT.depth() << endl;
	Mat skew = Mat::zeros(3, 3, CV_64FC1);
	skew.at<double>(0, 1) = -epT.at<double>(0, 2);
	skew.at<double>(0, 2) = epT.at<double>(0, 1);
	skew.at<double>(1, 0) = epT.at<double>(0, 2);
	skew.at<double>(1, 2) = -epT.at<double>(0, 0);
	skew.at<double>(2, 0) = -epT.at<double>(0, 1);
	skew.at<double>(2, 1) = epT.at<double>(0, 0);
	cout << "sdsd" << endl;
	//cout << skew << endl;
	return skew;
}

int getRank(Mat M) {
	Mat1d w, u, vt;
	SVD::compute(M, w, u, vt);
	//w es la matriz de valores no singulares
	//Asi que se busca aquellos valores no singulares que no sean 0s
	//Para ello usamos un threshold pequeño 
	Mat1b nonZeroSingularesValues = w > 0.0001;
	//y contamos el numero de valores no nulos
	int rank = countNonZero(nonZeroSingularesValues);

	return rank;

}

Mat makeInvertible(Mat ninv) {
	//int dim = ninv.rows;
	//int rank = getRank(ninv);
	Mat Sm, U, V;
	SVD::compute(ninv, Sm, U, V, SVD::FULL_UV);
	transpose(V, V);
	/*
	cout << "S" << endl << endl;
	cout << Sm << endl;
	cout << "U" << endl << endl;
	cout << U << endl;
	cout << "V" << endl << endl;
	cout << V << endl;
	*/
	Mat S = Mat::eye(Sm.rows, Sm.rows, CV_64F);
	S.at<double>(0, 0) = Sm.at<double>(0, 0);
	S.at<double>(1, 1) = Sm.at<double>(1, 0);
	S.at<double>(2, 2) = Sm.at<double>(2, 0);
	//cout << "S" << endl << endl;
	//cout << S << endl;

	Mat Ss = Mat(S, Rect(0, 0, 2, 2));
	Mat Us = Mat(U, Rect(0, 0, 2, 3));
	Mat Vs = Mat(V, Rect(0, 0, 2, 3));
	/*
	cout << "S" << endl << endl;
	cout << Ss << endl;
	cout << "U" << endl << endl;
	cout << Us << endl;
	cout << "V" << endl << endl;
	cout << Vs << endl;
	*/
	Mat I = Mat::eye(ninv.rows, ninv.rows, CV_64F);
	Mat Ust;
	transpose(Us, Ust);

	Mat inv = ninv + (I - Us*Ust); //obtenemos la matriz inversa correcta a partir de la matriz no inversa

	return inv;

	/*
	cout << "inv" << endl;
	cout << inv << endl;
	cout << "ninv" << endl << endl;
	cout << ninv << endl;
	*/

	//a partir de aqui se hace un proceso de verificacion, falta concluir
	/*
	Eigen::Matrix<double, 3, 3> ninv_e;

	ninv_e.row(0).col(0).setConstant(ninv.at<double>(0, 0));
	ninv_e.row(0).col(1).setConstant(ninv.at<double>(0, 1));
	ninv_e.row(0).col(2).setConstant(ninv.at<double>(0, 2));

	ninv_e.row(1).col(0).setConstant(ninv.at<double>(1, 0));
	ninv_e.row(1).col(1).setConstant(ninv.at<double>(1, 1));
	ninv_e.row(1).col(2).setConstant(ninv.at<double>(1, 2));

	ninv_e.row(2).col(0).setConstant(ninv.at<double>(2, 0));
	ninv_e.row(2).col(1).setConstant(ninv.at<double>(2, 1));
	ninv_e.row(2).col(2).setConstant(ninv.at<double>(2, 2));

	Eigen::EigenSolver<Eigen::Matrix3d> es(ninv_e, true);

	Eigen::VectorXcd eigenvals = es.eigenvalues();
	Eigen::MatrixXcd eigenvecs = es.eigenvectors();

	Mat eigenvalues = Mat(ninv.rows, 1, CV_64F);
	Mat eigenvector = Mat(ninv.rows, ninv.rows, CV_64F);;
	Mat eigenvaluesD = Mat::eye(ninv.rows, ninv.rows, CV_64F);

	eigenvalues.at<double>(0) = real(eigenvals[0]); //
	eigenvalues.at<double>(1) = real(eigenvals[1]); //
	eigenvalues.at<double>(2) = real(eigenvals[2]); //

	eigenvaluesD.at<double>(0, 0) = real(eigenvals[0]);
	eigenvaluesD.at<double>(1, 1) = real(eigenvals[1]);
	eigenvaluesD.at<double>(2, 2) = real(eigenvals[2]);

	eigenvector.at<double>(0, 0) = real(eigenvecs(0, 0));
	eigenvector.at<double>(0, 1) = real(eigenvecs(0, 1));
	eigenvector.at<double>(0, 2) = real(eigenvecs(0, 2));

	eigenvector.at<double>(1, 0) = real(eigenvecs(1, 0));
	eigenvector.at<double>(1, 1) = real(eigenvecs(1, 1));
	eigenvector.at<double>(1, 2) = real(eigenvecs(1, 2));

	eigenvector.at<double>(2, 0) = real(eigenvecs(2, 0));
	eigenvector.at<double>(2, 1) = real(eigenvecs(2, 1));
	eigenvector.at<double>(2, 2) = real(eigenvecs(2, 2));

	for (int i = 0; i < ninv.rows; i++) {
	if (eigenvalues.at<double>(i) <= 0.00001) {
	cout << " + " << i << endl;
	}
	}


	cout << "eigenvalues de ninv" << endl << endl;
	cout << eigenvaluesD << endl;


	cout << "eigenvectors de ninv" << endl << endl;
	cout << eigenvector << endl;
	*/

	//https://mathoverflow.net/questions/251206/transforming-a-non-invertible-matrix-into-an-invertible-matrix
	//https://eigen.tuxfamily.org/dox/GettingStarted.html
	//http://ksimek.github.io/2012/08/14/decompose/


}

//
template <typename T>
static float distancePointLine(const cv::Point_<T> point, const cv::Vec<T, 3>& line)
{
	//Line is given as a*x + b*y + c = 0
	return std::fabsf(line(0)*point.x + line(1)*point.y + line(2))
		/ std::sqrt(line(0)*line(0) + line(1)*line(1));
}

void HouseHolderQR(const cv::Mat &A, cv::Mat &Q, cv::Mat &R)
{
	assert(A.channels() == 1);
	assert(A.rows >= A.cols);
	auto sign = [](double value) { return value >= 0 ? 1 : -1; };
	const auto totalRows = A.rows;
	const auto totalCols = A.cols;
	R = A.clone();
	Q = cv::Mat::eye(totalRows, totalRows, A.type());
	for (int col = 0; col < A.cols; ++col)
	{
		cv::Mat matAROI = cv::Mat(R, cv::Range(col, totalRows), cv::Range(col, totalCols));
		cv::Mat y = matAROI.col(0);
		auto yNorm = norm(y);
		cv::Mat e1 = cv::Mat::eye(y.rows, 1, A.type());
		cv::Mat w = y + sign(y.at<double>(0, 0)) *  yNorm * e1;
		cv::Mat v = w / norm(w);
		cv::Mat vT; cv::transpose(v, vT);
		cv::Mat I = cv::Mat::eye(matAROI.rows, matAROI.rows, A.type());
		cv::Mat I_2VVT = I - 2 * v * vT;
		cv::Mat matH = cv::Mat::eye(totalRows, totalRows, A.type());
		cv::Mat matHROI = cv::Mat(matH, cv::Range(col, totalRows), cv::Range(col, totalRows));
		I_2VVT.copyTo(matHROI);
		R = matH * R;
		Q = Q * matH;
	}
}


int main() {
	freopen("input_thesis.txt", "r", stdin);
	freopen("output_thesis.txt", "w", stdout);
	
	//String img1 = "images/pokeball/im18.jpg";
	//String img2 = "images/pokeball/im22.jpg";

	String img1 = "images/test/im1.jpg";
	String img2 = "images/test/im6.jpg";

	Pair3DReconstruction *pair3dRec;
	pair3dRec = new Pair3DReconstruction(img1, img2);
	pair3dRec->calculateKeyPoints();
	//pair3dRec->drawKeyPoints(pair3dRec->getRGBImage1(), pair3dRec->getKeyPoints1(), "Key Point from Image 1");
	//pair3dRec->drawKeyPoints(pair3dRec->getRGBImage2(), pair3dRec->getKeyPoints2(), "Key Point from Image 2");
	pair3dRec->calculateDescriptors();
	pair3dRec->calculateMatching();
	pair3dRec->drawMatching(false);
	//pair3dRec->doPointsFiltering(); //we keep only inliers points
	
	//cout << "FUNDAMENTAL MATRIX by OpenCV before inliers filtering" << endl; //not necessary anymore
	//cout << pair3dRec->getFundamentalMatrix() << endl;

	pair3dRec->calculateFundamentalMatrix(); //Fundamental Matrix Calculation
	Mat F = pair3dRec->getFundamentalMatrix();
	/*
	vector<KeyPoint> imgpts1_good;
	vector<KeyPoint> imgpts2_good;
	vector<DMatch> matches;
	Mat F = pair3dRec->GetFundamentalMat(pair3dRec->getKeyPoints1(),
		pair3dRec->getKeyPoints2(),
		imgpts1_good,
		imgpts2_good,
		matches,
		pair3dRec->getGRAYImage1(),
		pair3dRec->getGRAYImage2()
	);
	*/
	cout << "FUNDAMENTAL MATRIX by colmap" << endl;
	cout << F << endl;
	
	pair3dRec->calculateEssentialMatrix(); //Essential Matrix calculation
	Mat E = pair3dRec->getEssentialMatrix();

	cout << "ESSENTIAL MATRIX" << endl;
	cout << E << endl;

	Mat K = pair3dRec->getCameraMatrix();
	cout << "CAMERA MATRIX" << endl;
	cout << K << endl;

	//calculo de la matriz de proyeccion
	Eigen::Matrix3d* R = new Eigen::Matrix3d();
	Eigen::Vector3d* t = new Eigen::Vector3d();
	std::vector<Eigen::Vector3d>* points3D = new std::vector<Eigen::Vector3d>();

	Eigen::Matrix3d Es = Eigen::Matrix3d::Zero(3,3);
	Es(0, 0) = E.at<double>(0, 0);
	Es(0, 1) = E.at<double>(0, 1);
	Es(0, 2) = E.at<double>(0, 2);
	Es(1, 0) = E.at<double>(1, 0);
	Es(1, 1) = E.at<double>(1, 1);
	Es(1, 2) = E.at<double>(1, 2);
	Es(2, 0) = E.at<double>(2, 0);
	Es(2, 1) = E.at<double>(2, 1);
	Es(2, 2) = E.at<double>(2, 2);
	
	cout << "essential" << endl;
	cout << Es << endl;

	//cv::Mat_<float> a = Mat_<float>::ones(2, 2);
	//Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> b;
	//cv2eigen(a, b);

	pair3dRec->PoseFromEssentialMatrix(Es, pair3dRec->getPoints1(), pair3dRec->getPoints2(), R, t, points3D);


	ofstream outputPLY; //archivo de salida PLY
	outputPLY.open("output3D.ply");
	outputPLY << "ply" << endl;
	outputPLY << "format ascii 1.0" << endl;
	outputPLY << "comment written by Leonel Ocsa Sanchez" << endl;
	outputPLY << "element vertex " << points3D->size() << endl;
	outputPLY << "property float32 x" << endl;
	outputPLY << "property float32 y" << endl;
	outputPLY << "property float32 z" << endl;
	outputPLY << "property uchar red" << endl;
	outputPLY << "property uchar green" << endl;
	outputPLY << "property uchar blue" << endl;
	outputPLY << "end_header" << endl;
	outputPLY << endl;

	Mat img = pair3dRec->getRGBImage1();
	vector<Point2f> points = pair3dRec->getOPoints1();
	
	for (int i = 0; i < points3D->size(); i++) {
		cout << points3D->at(i) << endl;
		outputPLY << points3D->at(i).x()  << " ";
		outputPLY << points3D->at(i).y() << " ";
		outputPLY << points3D->at(i).z()  << " ";
		outputPLY << (int)(img.at<Vec3b>(Point(points[i].x, points[i].y))[0]) << " ";
		outputPLY << (int)(img.at<Vec3b>(Point(points[i].x, points[i].y))[1]) << " ";
		outputPLY << (int)(img.at<Vec3b>(Point(points[i].x, points[i].y))[2]) << endl;
	}
	outputPLY.close();

	cout << "points size" << endl;
	cout << points.size() << endl;
	
	waitKey(0);
	return 0;
}