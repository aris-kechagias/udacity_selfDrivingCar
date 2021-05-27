#include <cmath>
#include <vector>
#include <numeric>
#include "Twiddle.h"

Twiddle::Twiddle(const PID& pid, double tol, int load, int limit)
	: tolerance(tol), load_algo_iterations(load), cycle_limit(limit)
{
	std::vector<double>  coeffs = { pid.getKp(), pid.getKi(), pid.getKd() };
	pid_coeffs = coeffs;
	
	std::vector<double> deltas = { pid.getKp() / 10, pid.getKi() / 10, pid.getKd() / 10 };
	delta_p = deltas;
}

Twiddle::~Twiddle()
{
}

void Twiddle::calculate_error(double cte)
{
	quadratic_error = (current_run < load_algo_iterations) ? 0 
		: quadratic_error + (cte * cte);

	// 0 for wrong values of denominator
	avg_error = quadratic_error / std::abs(current_run - load_algo_iterations);
}

void
Twiddle::tune(double cte) //cross track error
{

	// reset to avoid overflow
	if (current_run == cycle_limit) {
		current_run = 0;
		for (std::size_t i = 0; i < pid_coeffs.size(); i++)
		{
			pid_coeffs[i] += delta_p[i];
		}
		best_error = 0;
		avg_error = 0;
		state = increase;
	}

	current_run++;
	
	calculate_error(cte);

	if (current_run == load_algo_iterations)
	{
		best_error = avg_error;
	}
	if (current_run > load_algo_iterations)
	{
		if (std::accumulate(delta_p.begin(), delta_p.end(), 0) > tolerance)
		{
			if (state == increase)									// 2. run
			{
				if (avg_error < best_error)						// 3. check error
				{
					best_error = avg_error;
					delta_p[idx] *= 1.1;
					next_coeff();
					//keep state
				}
				else
				{
					pid_coeffs[idx] -= 2 * delta_p[idx]; // 4. decrease
					//keep coefficient
					state = decrease;
				}
			}
			else																		 // 5. run
			{ // state == decrease
				if (avg_error < best_error)
				{
					best_error = avg_error;
					delta_p[idx] *= 1.1;
				}
				else
				{
					pid_coeffs[idx] += delta_p[idx];
					delta_p[idx] *= 0.9;
				}
				next_coeff();
				state = increase;
			}

			pid_coeffs[idx] += delta_p[idx];				// 1. update coeffs
		}
	}
}
