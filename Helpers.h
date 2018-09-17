#pragma once
#include <iostream>
#include <fstream>
#include <string>
#include <vector>


#include <stdlib.h>
#include <time.h>

#include <glog\logging.h>
#include <Eigenvalues>
#include <Eigen>

void ComputeSquaredSampsonError(const std::vector<Eigen::Vector2d>& points1,
	const std::vector<Eigen::Vector2d>& points2,
	const Eigen::Matrix3d& E,
	std::vector<double>* residuals) {
	CHECK_EQ(points1.size(), points2.size());

	residuals->resize(points1.size());

	// Note that this code might not be as nice as Eigen expressions,
	// but it is significantly faster in various tests

	const double E_00 = E(0, 0);
	const double E_01 = E(0, 1);
	const double E_02 = E(0, 2);
	const double E_10 = E(1, 0);
	const double E_11 = E(1, 1);
	const double E_12 = E(1, 2);
	const double E_20 = E(2, 0);
	const double E_21 = E(2, 1);
	const double E_22 = E(2, 2);

	for (size_t i = 0; i < points1.size(); ++i) {
		const double x1_0 = points1[i](0);
		const double x1_1 = points1[i](1);
		const double x2_0 = points2[i](0);
		const double x2_1 = points2[i](1);

		// Ex1 = E * points1[i].homogeneous();
		const double Ex1_0 = E_00 * x1_0 + E_01 * x1_1 + E_02;
		const double Ex1_1 = E_10 * x1_0 + E_11 * x1_1 + E_12;
		const double Ex1_2 = E_20 * x1_0 + E_21 * x1_1 + E_22;

		// Etx2 = E.transpose() * points2[i].homogeneous();
		const double Etx2_0 = E_00 * x2_0 + E_10 * x2_1 + E_20;
		const double Etx2_1 = E_01 * x2_0 + E_11 * x2_1 + E_21;

		// x2tEx1 = points2[i].homogeneous().transpose() * Ex1;
		const double x2tEx1 = x2_0 * Ex1_0 + x2_1 * Ex1_1 + Ex1_2;

		// Sampson distance
		(*residuals)[i] =
			x2tEx1 * x2tEx1 /
			(Ex1_0 * Ex1_0 + Ex1_1 * Ex1_1 + Etx2_0 * Etx2_0 + Etx2_1 * Etx2_1);
	}
}

//pol...

// Remove leading zero coefficients.
Eigen::VectorXd RemoveLeadingZeros(const Eigen::VectorXd& coeffs) {
	Eigen::VectorXd::Index num_zeros = 0;
	for (; num_zeros < coeffs.size(); ++num_zeros) {
		if (coeffs(num_zeros) != 0) {
			break;
		}
	}
	return coeffs.tail(coeffs.size() - num_zeros);
}

// Remove trailing zero coefficients.
Eigen::VectorXd RemoveTrailingZeros(const Eigen::VectorXd& coeffs) {
	Eigen::VectorXd::Index num_zeros = 0;
	for (; num_zeros < coeffs.size(); ++num_zeros) {
		if (coeffs(coeffs.size() - 1 - num_zeros) != 0) {
			break;
		}
	}
	return coeffs.head(coeffs.size() - num_zeros);
}



bool FindLinearPolynomialRoots(const Eigen::VectorXd& coeffs,
	Eigen::VectorXd* real, Eigen::VectorXd* imag) {
	CHECK_EQ(coeffs.size(), 2);

	if (coeffs(0) == 0) {
		return false;
	}

	if (real != nullptr) {
		real->resize(1);
		(*real)(0) = -coeffs(1) / coeffs(0);
	}

	if (imag != nullptr) {
		imag->resize(1);
		(*imag)(0) = 0;
	}

	return true;
}

bool FindQuadraticPolynomialRoots(const Eigen::VectorXd& coeffs,
	Eigen::VectorXd* real,
	Eigen::VectorXd* imag) {
	CHECK_EQ(coeffs.size(), 3);

	const double a = coeffs(0);
	if (a == 0) {
		return FindLinearPolynomialRoots(coeffs.tail(2), real, imag);
	}

	const double b = coeffs(1);
	const double c = coeffs(2);
	if (b == 0 && c == 0) {
		if (real != nullptr) {
			real->resize(1);
			(*real)(0) = 0;
		}
		if (imag != nullptr) {
			imag->resize(1);
			(*imag)(0) = 0;
		}
		return true;
	}

	const double d = b * b - 4 * a * c;

	if (d >= 0) {
		const double sqrt_d = std::sqrt(d);
		if (real != nullptr) {
			real->resize(2);
			if (b >= 0) {
				(*real)(0) = (-b - sqrt_d) / (2 * a);
				(*real)(1) = (2 * c) / (-b - sqrt_d);
			}
			else {
				(*real)(0) = (2 * c) / (-b + sqrt_d);
				(*real)(1) = (-b + sqrt_d) / (2 * a);
			}
		}
		if (imag != nullptr) {
			imag->resize(2);
			imag->setZero();
		}
	}
	else {
		if (real != nullptr) {
			real->resize(2);
			real->setConstant(-b / (2 * a));
		}
		if (imag != nullptr) {
			imag->resize(2);
			(*imag)(0) = std::sqrt(-d) / (2 * a);
			(*imag)(1) = -(*imag)(0);
		}
	}

	return true;
}

bool FindPolynomialRootsDurandKerner(const Eigen::VectorXd& coeffs_all,
	Eigen::VectorXd* real,
	Eigen::VectorXd* imag) {
	CHECK_GE(coeffs_all.size(), 2);

	const Eigen::VectorXd coeffs = RemoveLeadingZeros(coeffs_all);

	const int degree = coeffs.size() - 1;

	if (degree <= 0) {
		return false;
	}
	else if (degree == 1) {
		return FindLinearPolynomialRoots(coeffs, real, imag);
	}
	else if (degree == 2) {
		return FindQuadraticPolynomialRoots(coeffs, real, imag);
	}

	// Initialize roots.
	Eigen::VectorXcd roots(degree);
	roots(degree - 1) = std::complex<double>(1, 0);
	for (int i = degree - 2; i >= 0; --i) {
		roots(i) = roots(i + 1) * std::complex<double>(1, 1);
	}

	// Iterative solver.
	const int kMaxNumIterations = 100;
	const double kMaxRootChange = 1e-10;
	for (int iter = 0; iter < kMaxNumIterations; ++iter) {
		double max_root_change = 0.0;
		for (int i = 0; i < degree; ++i) {
			const std::complex<double> root_i = roots(i);
			std::complex<double> numerator = coeffs[0];
			std::complex<double> denominator = coeffs[0];
			for (int j = 0; j < degree; ++j) {
				numerator = numerator * root_i + coeffs[j + 1];
				if (i != j) {
					denominator = denominator * (root_i - roots(j));
				}
			}
			const std::complex<double> root_i_change = numerator / denominator;
			roots(i) = root_i - root_i_change;
			max_root_change =
				std::max(max_root_change, std::abs(root_i_change.real()));
			max_root_change =
				std::max(max_root_change, std::abs(root_i_change.imag()));
		}

		// Break, if roots do not change anymore.
		if (max_root_change < kMaxRootChange) {
			break;
		}
	}

	if (real != nullptr) {
		real->resize(degree);
		*real = roots.real();
	}
	if (imag != nullptr) {
		imag->resize(degree);
		*imag = roots.imag();
	}

	return true;
}

bool FindPolynomialRootsCompanionMatrix(const Eigen::VectorXd& coeffs_all,
	Eigen::VectorXd* real,
	Eigen::VectorXd* imag) {
	CHECK_GE(coeffs_all.size(), 2);

	Eigen::VectorXd coeffs = RemoveLeadingZeros(coeffs_all);

	const int degree = coeffs.size() - 1;

	if (degree <= 0) {
		return false;
	}
	else if (degree == 1) {
		return FindLinearPolynomialRoots(coeffs, real, imag);
	}
	else if (degree == 2) {
		return FindQuadraticPolynomialRoots(coeffs, real, imag);
	}

	// Remove the coefficients where zero is a solution.
	coeffs = RemoveTrailingZeros(coeffs);

	// Check if only zero is a solution.
	if (coeffs.size() == 1) {
		if (real != nullptr) {
			real->resize(1);
			(*real)(0) = 0;
		}
		if (imag != nullptr) {
			imag->resize(1);
			(*imag)(0) = 0;
		}
		return true;
	}

	// Fill the companion matrix.
	Eigen::MatrixXd C(coeffs.size() - 1, coeffs.size() - 1);
	C.setZero();
	for (Eigen::MatrixXd::Index i = 1; i < C.rows(); ++i) {
		C(i, i - 1) = 1;
	}
	C.row(0) = -coeffs.tail(coeffs.size() - 1) / coeffs(0);

	// Solve for the roots of the polynomial.
	Eigen::EigenSolver<Eigen::MatrixXd> solver(C, false);
	if (solver.info() != Eigen::Success) {
		return false;
	}

	// If there are trailing zeros, we must add zero as a solution.
	const int effective_degree =
		coeffs.size() - 1 < degree ? coeffs.size() : coeffs.size() - 1;

	if (real != nullptr) {
		real->resize(effective_degree);
		real->head(coeffs.size() - 1) = solver.eigenvalues().real();
		if (effective_degree > coeffs.size() - 1) {
			(*real)(real->size() - 1) = 0;
		}
	}
	if (imag != nullptr) {
		imag->resize(effective_degree);
		imag->head(coeffs.size() - 1) = solver.eigenvalues().imag();
		if (effective_degree > coeffs.size() - 1) {
			(*imag)(imag->size() - 1) = 0;
		}
	}

	return true;
}