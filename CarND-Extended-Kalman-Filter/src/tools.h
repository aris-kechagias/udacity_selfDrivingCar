#ifndef TOOLS_H_
#define TOOLS_H_

#include <vector>
#include <string>
#include "Eigen/Dense"

class Tools {
 public:
  /**
   * Constructor.
   */
  Tools();

  /**
   * Destructor.
   */
  virtual ~Tools();

  /**
   * A helper method to calculate RMSE.
   */
  Eigen::VectorXd CalculateRMSE(const std::vector<Eigen::VectorXd> &estimations, 
                                const std::vector<Eigen::VectorXd> &ground_truth);

  /**
   * A helper method to calculate Jacobians.
   */
  Eigen::MatrixXd CalculateJacobian(const Eigen::VectorXd& x_state);
  
  /**
   * Constraints the value in approaching zero
   */
  void ValueLimiter(double *val, std::string txt);
};

#endif  // TOOLS_H_
