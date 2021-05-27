#ifndef PID_H
#define PID_H

#include <vector>

class PID {
 public:
  /**
   * Initialize PID.
   * @param (Kp_, Ki_, Kd_) The initial PID coefficients
   */
  PID(double Kp_ = 0.13, double Ki_ = 0.0005, double Kd_ = 3.5);
  virtual ~PID();

  void setCoeffs(std::vector<double> coeffs);

  /**
   * Update the PID error variables given cross track error.
   * @param cte The current cross track error
   */
  void updateError(double cte);

  /**
   * Calculate the total PID error.
   * @output The total PID error
   */
  double totalError();
  
  double getKp() const;
  double getKi() const;
  double getKd() const;

 private:
  /**
   * PID Errors
   */
  double p_error;
  double i_error;
  double d_error;

  /**
   * PID Coefficients
   */ 
  double Kp;
  double Ki;
  double Kd;

  // Calculate steering value here, remember the steering value is [-1, 1].
  double limit_output(double output, int limit) {
    return (output < -limit) ? -limit :
      (output > limit) ? limit :
      output;
  }
};

#endif  // PID_H