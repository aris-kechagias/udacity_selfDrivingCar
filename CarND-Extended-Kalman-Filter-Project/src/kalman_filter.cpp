#include "kalman_filter.h"
#include <iostream>
#include "tools.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::cout;
using std::endl;

/* 
 * Please note that the Eigen library does not initialize 
 *   VectorXd or MatrixXd objects with zeros upon creation.
 */

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  Q_ = Q_in;
  H_ = H_in;
  R_ = R_in;
}

//Linear
void KalmanFilter::Predict() {
  x_ = F_ * x_;  // n = 0 as zero mean is supposed
  MatrixXd Ft = F_.transpose();
  P_ = F_ * P_ * Ft + Q_;
}

//Laser
void KalmanFilter::Update(const VectorXd &z) {

  VectorXd y = z - H_ * x_;  //measurement update

  StateUpdate(y);
}

//Radar
void KalmanFilter::UpdateEKF(const VectorXd &z) {
  
  //calculate measurement function
  tools.ValueLimiter(&x_[0], "UpdateEKF () - Limiting divisor");
  tools.ValueLimiter(&x_[1], "UpdateEKF () - Limiting divisor");

  double c1 = sqrtf(pow(x_[0], 2) + pow(x_[1], 2));
  double c2 = atan2(x_[1], x_[0]);
  double c3 = (x_[0] * x_[2] + x_[1] * x_[3]) / c1;

  VectorXd h_ = VectorXd(3);
  h_ << c1, c2, c3;

  VectorXd y = z - h_;  //measurement update
  y[1] = NormalizeAngle(y[1]);


  StateUpdate(y);
}

void KalmanFilter::StateUpdate(VectorXd y)
{
  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd K = (P_ * Ht) * S.inverse();

  x_ = x_ + (K * y);
  P_ = (MatrixXd::Identity(x_.size(), x_.size()) - K * H_) * P_;
}

double KalmanFilter::NormalizeAngle(double angle) 
{
  //normalize angle in [-pi, pi]

  while (angle > M_PI) angle -= 2. * M_PI;
  while (angle < -M_PI) angle += 2. * M_PI;
  return angle;
}
