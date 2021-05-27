#include <numeric>
#include "PID.h"

PID::PID(double Kp_, double Ki_, double Kd_)
  : Kp(Kp_), Ki(Ki_), Kd(Kd_) {
  p_error = 0;
  i_error = 0;
  d_error = 0;
}

PID::~PID() {}

void PID::setCoeffs(std::vector<double> coeffs) {
  Kp = coeffs[0];
  Ki = coeffs[1];
  Kd = coeffs[2];
}

void PID::updateError(double cte) {
  i_error += cte;
  d_error = cte - p_error;
  p_error = cte;
}

double PID::totalError() {
  double error = + Kp*p_error + Ki*i_error + Kd*d_error;
	return limit_output(error, 1);
}

double PID::getKp() const
{
	return Kp;
}

double PID::getKi() const
{
	return Ki;
}

double PID::getKd() const
{
	return Kd;
}
