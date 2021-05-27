/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"
#include <random>

using std::string;
using std::vector;
using std::normal_distribution;

using namespace std;

/**
 * TODO: Set the number of particles. Initialize all particles to
 *   first position (based on estimates of x, y, theta and their uncertainties
 *   from GPS) and all weights to 1.
 * TODO: Add random Gaussian noise to each particle.
 * NOTE: Consult particle_filter.h for more information about this method
 *   (and others in this file).
 */
void ParticleFilter::init(double x, double y, double theta, double std[]) 
{
  if (!is_initialized) 
  {
    unsigned num_particles = 50;  // TODO: Set the number of particles
    std::default_random_engine generator;

    particles.reserve(num_particles);
    weights.resize(num_particles, 1.0);

    normal_distribution<double> dist_x(x, std[0]);
    normal_distribution<double> dist_y(y, std[1]);
    normal_distribution<double> dist_theta(theta, std[2]);

    //Add random gaussian noise
    for (unsigned i = 0; i != num_particles; ++i)
    {
      Particle temp = { .id = (int)i,
        .x = dist_x(generator),
        .y = dist_y(generator),
        .theta = dist_theta(generator),
        .weight = 1 };

      particles.push_back(temp);
    }
    is_initialized = true;
  }
}

/**
 * TODO: Add measurements to each particle and add random Gaussian noise.
 * NOTE: When adding noise you may find std::normal_distribution
 *   and std::default_random_engine useful.
 *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
 *  http://www.cplusplus.com/reference/random/default_random_engine/
 */
void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) 
{
  std::default_random_engine generator;
  for (auto& part : particles)
  {
    if (fabs(yaw_rate) > 0.0001)
    {
      part.x += (velocity / yaw_rate) * (sin(part.theta + (yaw_rate * delta_t)) - sin(part.theta));
      part.y += (velocity / yaw_rate) * (cos(part.theta) - cos(part.theta + (yaw_rate * delta_t)));
      part.theta += yaw_rate * delta_t;
    }
    else
    {
      part.x += velocity * delta_t * cos(part.theta);
      part.y += velocity * delta_t * sin(part.theta);
      // part.theta same
    }

    normal_distribution<double> noise_x(0, std_pos[0]);
    normal_distribution<double> noise_y(0, std_pos[1]);
    normal_distribution<double> noise_theta(0, std_pos[2]);

    //Add random gaussian noise
    part.x += noise_x(generator);
    part.y += noise_y(generator);
    part.theta += noise_theta(generator);
  }
}

/**
  * TODO: Predict measurements to all map landmarks within sensor range for each particle
  *   and update the weights of each particle using a multi-variate Gaussian probability density function.
  * TODO: Update the weights of each particle using a multi-variate Gaussian 
  *   distribution. You can read more about this distribution here: 
  *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
  * NOTE: The observations are given in the VEHICLE'S coordinate system. 
  *   Your particles are located according to the MAP'S coordinate system. 
  *   You will need to transform between the two systems. Keep in mind that
  *   this transformation requires both rotation AND translation (but no scaling).
  *   The following is a good resource for the theory:
  *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
  *   and the following is a good resource for the actual equation to implement
  *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
  */
void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const vector<LandmarkObs>& observations, 
                                   const Map& map_landmarks)
{
  double weight_sum = 0;

  for (size_t i = 0; i < particles.size(); i++)
  {
    weights[i] = 1.0;
    particles[i].weight = 1.0;
    particles[i].associations.clear();
    particles[i].sense_x.clear();
    particles[i].sense_y.clear();

    vector<LandmarkObs> landmarks_inrange;
    for (auto map_landmark : map_landmarks.landmark_list)
    {
      if (dist(map_landmark.x_f, map_landmark.y_f, particles[i].x, particles[i].y) <= sensor_range)
      {
        LandmarkObs landmark{ .id = map_landmark.id_i, .x = map_landmark.x_f, .y = map_landmark.y_f };
        landmarks_inrange.push_back(landmark);
      }
    }

    vector<LandmarkObs> transformed_obs;
    for (auto& obs : observations)
    {
      double obs_weight;
      LandmarkObs obs_glob;
      obs_glob.x = particles[i].x + (cos(particles[i].theta) * obs.x) - (sin(particles[i].theta) * obs.y);
      obs_glob.y = particles[i].y + (sin(particles[i].theta) * obs.x) + (cos(particles[i].theta) * obs.y);
      obs_glob.id = obs.id;
      transformed_obs.push_back(obs_glob);

      if (landmarks_inrange.size() > 0)
      {
        auto landmarks_start = landmarks_inrange.begin();
        LandmarkObs best_landmark = *landmarks_start; // ok as distances > 0
        double best_dist = dist(landmarks_start->x, landmarks_start->y, obs_glob.x, obs_glob.y);

        for (auto it = ++landmarks_start; it != landmarks_inrange.end(); ++it)
        {
          double temp = dist(it->x, it->y, obs_glob.x, obs_glob.y);
          if (temp < best_dist)
          {
            best_dist = temp;
            best_landmark = *it;
          }
        }

        obs_glob.id = best_landmark.id;
        obs_weight = multiv_prob(std_landmark[0], std_landmark[1], obs_glob.x, obs_glob.y, best_landmark.x, best_landmark.y);
      }

      particles[i].associations.push_back(obs_glob.id);
      particles[i].sense_x.push_back(obs_glob.x);
      particles[i].sense_y.push_back(obs_glob.y);

      if (obs_weight > 0)
      {
        particles[i].weight = obs_weight;
        weights[i] = obs_weight;
      }
    }  // for observations
    weight_sum += particles[i].weight;
  } // for particles

  //Normalization
  for (size_t i = 0; i < particles.size(); i++)
  {
    particles[i].weight /= weight_sum;
    weights[i] /= weight_sum;
  }
}

/**
 * TODO: Resample particles with replacement with probability proportional
 *   to their weight.
 * NOTE: You may find std::discrete_distribution helpful here.
 *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
 */
void ParticleFilter::resample() 
{
  vector<Particle> resampled_particles;
  vector<double> resampled_weights;

  for (auto part : particles)
  {
    resampled_weights.push_back(part.weight);
  }

  std::default_random_engine generator;
  std::discrete_distribution<int> weight_distr(resampled_weights.begin(), resampled_weights.end());

  for (size_t i = 0; i < particles.size(); ++i)
  {
    int sample_idx = weight_distr(generator);
    resampled_particles.push_back(particles[sample_idx]);
    resampled_particles.back().id = i; 
  }
  particles = resampled_particles;
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x; 
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1);  // get rid of the trailing space
  return s;
}