#ifndef COST_H
#define COST_H

#include <map>
#include <vector>

#include "Vehicle.h"

using std::map;
using std::string;
using std::vector;


float goal_distance_cost(
  const Lane ego_lane,
  const Lane target_lane,
  const Vehicle ego_vehicle,
  const std::map<Lane, std::vector<Vehicle>>& traffic);

float inefficiency_cost(
  const Lane ego_lane,
  const Lane target_lane,
  const Vehicle& ego_vehicle,
  const std::map<Lane, std::vector<Vehicle>>& traffic);

float calculate_cost(
  const Lane ego_lane,
  const Lane target_lane,
  const Vehicle& ego_vehicle,
  const std::map<Lane, std::vector<Vehicle>>& traffic);


float lane_speed(
  const vector<Vehicle>& sorted_lane_traffic, 
  const Vehicle ego_vehicle);

#endif  // COST_H