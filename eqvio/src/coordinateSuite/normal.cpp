/*
    This file is part of EqVIO.

    EqVIO is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    EqVIO is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with EqVIO.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "eqvio/EqFMatrices.h"

using namespace Eigen;
using namespace std;
using namespace liepp;

Eigen::MatrixXd EqFStateMatrixA_normal(const VIOGroup& X, const VIOState& xi0, const IMUVelocity& imuVel);
Eigen::MatrixXd EqFInputMatrixB_normal(const VIOGroup& X, const VIOState& xi0);
Eigen::Matrix<double, 2, 3> EqFoutputMatrixCiStar_normal(
    const Vector3d& q0, const SOT3d& QHat, const GIFT::GICameraPtr& camPtr, const Eigen::Vector2d& y);

VIOAlgebra liftInnovation_normal(const Eigen::VectorXd& baseInnovation, const VIOState& xi0);
VIOAlgebra liftInnovation_normal(const Eigen::VectorXd& totalInnovation, const VIOState& xi0);
VIOGroup liftInnovationDiscrete_normal(const Eigen::VectorXd& totalInnovation, const VIOState& xi0);

const EqFCoordinateSuite EqFCoordinateSuite_normal{VIOChart_normal,        EqFStateMatrixA_normal,
                                                   EqFInputMatrixB_normal, EqFoutputMatrixCiStar_normal,
                                                   liftInnovation_normal,  liftInnovationDiscrete_normal};
 
Eigen::MatrixXd EqFStateMatrixA_normal(const VIOGroup& X, const VIOState& xi0, const IMUVelocity& imuVel) {
    const MatrixXd& M = coordinateDifferential_normal_euclid(xi0);
    return M * EqFCoordinateSuite_euclid.stateMatrixA(X, xi0, imuVel) * M.inverse();
}

Eigen::MatrixXd EqFInputMatrixB_normal(const VIOGroup& X, const VIOState& xi0) {
    const MatrixXd& M = coordinateDifferential_normal_euclid(xi0);
    return M * EqFCoordinateSuite_euclid.inputMatrixB(X, xi0);
}

VIOAlgebra liftInnovation_normal(const Eigen::VectorXd& baseInnovation, const VIOState& xi0) {
    const MatrixXd& M = coordinateDifferential_normal_euclid(xi0);
    return EqFCoordinateSuite_euclid.liftInnovation(M.inverse() * baseInnovation, xi0);
}

VIOGroup liftInnovationDiscrete_normal(const Eigen::VectorXd& totalInnovation, const VIOState& xi0) {
    return EqFCoordinateSuite_euclid.liftInnovationDiscrete(
        VIOChart_euclid(VIOChart_normal.inv(totalInnovation, xi0), xi0), xi0);
}

Eigen::Matrix<double, 2, 3> EqFoutputMatrixCiStar_normal(
    const Vector3d& q0, const SOT3d& QHat, const GIFT::GICameraPtr& camPtr, const Eigen::Vector2d& y) {
    (void)y;
    const Vector3d& y0 = q0.normalized();
    const Vector3d& yHat = QHat.R.inverse() * y0;
    Eigen::Matrix<double, 2, 3> C0i = Eigen::Matrix<double, 2, 3>::Zero();
    C0i.block<2, 2>(0, 0) =
        camPtr->projectionJacobian(yHat) * QHat.R.asMatrix().transpose() * sphereChart_normal.chartInvDiff0(q0);
    return C0i;
}
