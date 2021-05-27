#include "FusionEKF.h"
#include <iostream>
#include "Eigen/Dense"
#include "tools.h"
#include <ctype.h>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::cout;
using std::endl;
using std::vector;

FusionEKF::FusionEKF() {
  is_initialized_ = false;

  previous_timestamp_ = 0;

  // initializing matrices
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  H_laser_ = MatrixXd(2, 4);
  Hj_ = MatrixXd(3, 4);

  //measurement covariance matrix - laser
  R_laser_ << 0.0225, 0,
              0, 0.0225;

  //measurement covariance matrix - radar
  R_radar_ << 0.09, 0, 0,
              0, 0.0009, 0,
              0, 0, 0.09;

  // measurement matrix - laser
  H_laser_ << 1, 0, 0, 0,
              0, 1, 0, 0;

  // measurement matrix - radar
  Hj_ << 1, 1, 0, 0,
         1, 1, 0, 0,
         1, 1, 1, 1;

  // process noise
  noise_ax_ = 9.0;
  noise_ay_ = 9.0;
}

FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {

  size_t sz = measurement_pack.raw_measurements_.size();
  for (size_t i = 0; i < sz; ++i)
  {
    if (std::isnan(measurement_pack.raw_measurements_[i])) return;
  }

  // Initialization

  if (!is_initialized_) {

    // first measurement
    cout << "EKF: " << endl;
    // state covariance matrix P
    MatrixXd P_init = MatrixXd(4, 4);
    P_init << 1, 0,    0, 0,
              0, 1,    0, 0,
              0, 0, 1000, 0,
              0, 0, 0, 1000;

    // the initial transition matrix F_
    MatrixXd F_init = MatrixXd(4, 4);
    F_init << 1, 0, 1, 0,
              0, 1, 0, 1,
              0, 0, 1, 0,
              0, 0, 0, 1;

    // process covariance matrix
    MatrixXd Q_init = MatrixXd(4, 4);
    Q_init << 1, 0, 1, 0,
              0, 1, 0, 1,
              1, 0, 1, 0,
              0, 1, 0, 1;

    VectorXd x_init = VectorXd(4);
    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      double ro = measurement_pack.raw_measurements_[0];
      double theta = measurement_pack.raw_measurements_[1];
      double ro_dot = measurement_pack.raw_measurements_[2];

      x_init << ro*cos(theta),
                ro*sin(theta),
                ro_dot*cos(theta),
                ro_dot*sin(theta);

      Hj_ = tools.CalculateJacobian(x_init);

      ekf_.Init(x_init, P_init, F_init, Hj_, R_radar_, Q_init);
    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      x_init << measurement_pack.raw_measurements_[0], 
        measurement_pack.raw_measurements_[1],
        0,
        0;

      ekf_.Init(x_init, P_init, F_init, H_laser_, R_laser_, Q_init);
    }
    
    previous_timestamp_ = measurement_pack.timestamp_;

    is_initialized_ = true; // done initializing, no need to predict or update
    return;
  }

  // Prediction

  double dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;
  previous_timestamp_ = measurement_pack.timestamp_;

  // Set F matrix with current time elapsed
  ekf_.F_(0, 2) = dt;
  ekf_.F_(1, 3) = dt;

  // Set the process covariance matrix Q
  double dt_2 = dt * dt;
  double dt_3 = dt_2 * dt;
  double dt_4 = dt_3 * dt;

  ekf_.Q_ << dt_4 / 4 * noise_ax_, 0, dt_3 / 2 * noise_ax_, 0,
             0, dt_4 / 4 * noise_ay_, 0, dt_3 / 2 * noise_ay_,
             dt_3 / 2 * noise_ax_, 0, dt_2* noise_ax_, 0,
             0, dt_3 / 2 * noise_ay_, 0, dt_2* noise_ay_;

  ekf_.Predict();

  // Update

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    ekf_.H_ = tools.CalculateJacobian(ekf_.x_);
    ekf_.R_ = R_radar_;

    ekf_.UpdateEKF(measurement_pack.raw_measurements_);
  } else {
    ekf_.H_ = H_laser_;
    ekf_.R_ = R_laser_;

    ekf_.Update(measurement_pack.raw_measurements_);
  }

  // print the output
  cout << "x_ = " << ekf_.x_ << endl;
  cout << "P_ = " << ekf_.P_ << endl;
}
