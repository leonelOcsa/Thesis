#include "stdafx.h"
#include "EssentialMatrixEstimator.h"



std::vector<EssentialMatrixEstimator::M_t>
EssentialMatrixEstimator::Estimate(const std::vector<X_t>& points1,
	const std::vector<Y_t>& points2) {
	//CHECK_EQ(points1.size(), points2.size());

	// Step 1: Extraction of the nullspace x, y, z, w.

	Eigen::Matrix<double, Eigen::Dynamic, 9> Q(points1.size(), 9);
	for (size_t i = 0; i < points1.size(); ++i) {
		const double x1_0 = points1[i](0);
		const double x1_1 = points1[i](1);
		const double x2_0 = points2[i](0);
		const double x2_1 = points2[i](1);
		Q(i, 0) = x1_0 * x2_0;
		Q(i, 1) = x1_1 * x2_0;
		Q(i, 2) = x2_0;
		Q(i, 3) = x1_0 * x2_1;
		Q(i, 4) = x1_1 * x2_1;
		Q(i, 5) = x2_1;
		Q(i, 6) = x1_0;
		Q(i, 7) = x1_1;
		Q(i, 8) = 1;
	}

	// Extract the 4 Eigen vectors corresponding to the smallest singular values.
	const Eigen::JacobiSVD<Eigen::Matrix<double, Eigen::Dynamic, 9>> svd(
		Q, Eigen::ComputeFullV);
	const Eigen::Matrix<double, 9, 4> E = svd.matrixV().block<9, 4>(0, 5);

	// Step 3: Gauss-Jordan elimination with partial pivoting on A.

	Eigen::Matrix<double, 10, 20> A;
	//#include "estimators/essential_matrix_poly.h"
	Eigen::Matrix<double, 10, 10> AA =
		A.block<10, 10>(0, 0).partialPivLu().solve(A.block<10, 10>(0, 10));

	// Step 4: Expansion of the determinant polynomial of the 3x3 polynomial
	//         matrix B to obtain the tenth degree polynomial.

	Eigen::Matrix<double, 13, 3> B;
	for (size_t i = 0; i < 3; ++i) {
		B(0, i) = 0;
		B(4, i) = 0;
		B(8, i) = 0;
		B.block<3, 1>(1, i) = AA.block<1, 3>(i * 2 + 4, 0);
		B.block<3, 1>(5, i) = AA.block<1, 3>(i * 2 + 4, 3);
		B.block<4, 1>(9, i) = AA.block<1, 4>(i * 2 + 4, 6);
		B.block<3, 1>(0, i) -= AA.block<1, 3>(i * 2 + 5, 0);
		B.block<3, 1>(4, i) -= AA.block<1, 3>(i * 2 + 5, 3);
		B.block<4, 1>(8, i) -= AA.block<1, 4>(i * 2 + 5, 6);
	}

	// Step 5: Extraction of roots from the degree 10 polynomial.
	Eigen::Matrix<double, 11, 1> coeffs;
	//#include "estimators/essential_matrix_coeffs.h"

	Eigen::VectorXd roots_real;
	Eigen::VectorXd roots_imag;
	if (!FindPolynomialRootsCompanionMatrix(coeffs, &roots_real, &roots_imag)) {
		return {};
	}

	std::vector<M_t> models;
	models.reserve(roots_real.size());

	for (Eigen::VectorXd::Index i = 0; i < roots_imag.size(); ++i) {
		const double kMaxRootImag = 1e-10;
		if (std::abs(roots_imag(i)) > kMaxRootImag) {
			continue;
		}

		const double z1 = roots_real(i);
		const double z2 = z1 * z1;
		const double z3 = z2 * z1;
		const double z4 = z3 * z1;

		Eigen::Matrix3d Bz;
		for (size_t j = 0; j < 3; ++j) {
			Bz(j, 0) = B(0, j) * z3 + B(1, j) * z2 + B(2, j) * z1 + B(3, j);
			Bz(j, 1) = B(4, j) * z3 + B(5, j) * z2 + B(6, j) * z1 + B(7, j);
			Bz(j, 2) = B(8, j) * z4 + B(9, j) * z3 + B(10, j) * z2 + B(11, j) * z1 +
				B(12, j);
		}

		const Eigen::JacobiSVD<Eigen::Matrix3d> svd(Bz, Eigen::ComputeFullV);
		const Eigen::Vector3d X = svd.matrixV().block<3, 1>(0, 2);

		const double kMaxX3 = 1e-10;
		if (std::abs(X(2)) < kMaxX3) {
			continue;
		}

		Eigen::MatrixXd essential_vec = E.col(0) * (X(0) / X(2)) +
			E.col(1) * (X(1) / X(2)) + E.col(2) * z1 +
			E.col(3);
		essential_vec /= essential_vec.norm();

		const Eigen::Matrix3d essential_matrix =
			Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(
				essential_vec.data());
		models.push_back(essential_matrix);
	}

	return models;
}

void EssentialMatrixEstimator::Residuals(
	const std::vector<X_t>& points1, const std::vector<Y_t>& points2,
	const M_t& E, std::vector<double>* residuals) {
	ComputeSquaredSampsonError(points1, points2, E, residuals);
}


