#ifndef TWIDDLE_H
#define TWIDDLE_H

#include "PID.h"

class Twiddle
{
public:
	Twiddle() = default;
	Twiddle(const PID& pid, double tol = 0.0001, int load = 100, int limit = 700);
	virtual ~Twiddle();

	void tune(double cte);

	std::vector<double> get_coeffs() {
		return pid_coeffs;
	}

private:
	std::vector<double> pid_coeffs;
	std::vector<double> delta_p;
	double tolerance;

	std::size_t idx = 0;
	void next_coeff() {
		idx = (idx + 1) % 3;
	}
	
	void calculate_error(double cte);

	double avg_error = 0;
	double best_error = 0;

	double quadratic_error = 0; // not need static as captured in lambda
	long unsigned current_run = 0;

	enum algo_state{
		increase,
		decrease
	} state;

	unsigned const load_algo_iterations;
	unsigned const cycle_limit;
};

#endif
