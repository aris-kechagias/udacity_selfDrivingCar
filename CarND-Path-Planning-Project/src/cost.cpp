#include "cost.h"

#include <cmath>
#include <functional>
#include <iterator>
#include <string>

#include <algorithm>

using std::string;
//using std::vector;
//using std::map;

const float REACH_GOAL = pow(10, 6);
const float EFFICIENCY = pow(10, 5);

/*Python implemented costs*/
// efficiency_cost
// total_jerk_cost
// max_jerk_cost
// max_accel_cost
// total_accel_cost
// exceeds_speed_limit_cost -> Not implemented
// stays_on_road_cost -> Not implemented
// buffer_cost
// collision_cost
// d_diff_cost
// s_diff_cost
// time_diff_cost


/**
* Cost Functions are called only if we have an obstacle
* 
* The cost calculation will be called once for every possible direction change
*/


/**
* Only calculate the cost for an imidiate change but if the inteded lane
* (to be calculated in the next cycle) has no cars divide by 2 to force a change
* 
* Cost increases based on distance of intended lane (for planning a lane change) 
*  and final lane of trajectory.
* Cost of being out of goal lane also becomes larger as vehicle approaches goal distance.
*/
float goal_distance_cost(
  const Lane ego_lane,
  const Lane target_lane,
  const Vehicle ego_vehicle,
  const std::map<Lane, std::vector<Vehicle>>& traffic) 
{
  float cost = 0;

  // ego lane
  if (traffic.at(target_lane).size() == 0)
  {
    cost = 0;
  }
  else
  {
    Vehicle closest_veh_target_lane = *std::upper_bound(
      traffic.at(target_lane).begin(),
      traffic.at(target_lane).end(),
      ego_vehicle);

    // compare distance to the ego lane
    float distance = closest_veh_target_lane.s_current - ego_vehicle.s_current;
    if (distance < 0)
    {
      cost = 1;
    }
    else
    {
      cost = 1 - exp(-(abs(target_lane - ego_lane) / distance));
    }

    // intended lane is the one we want to end to
    Lane intended_lane = (ego_lane == Lane::left)? Lane::right : 
      (ego_lane == Lane::right)? Lane::left : Lane::center; // assuming 3 lane road

    if (intended_lane != Lane::center && traffic.at(intended_lane).size() == 0)
    { // we could also repeat for this lane but just ask for an empty one
      cost /= 2;
    }
  }

  return cost;
}

/*
* Cost becomes higher for trajectories with intended lane 
*  that have traffic slower than vehicle's target speed.
* You can use the lane_speed function to determine the speed for a lane. 
*/
float inefficiency_cost(
  const Lane ego_lane,
  const Lane target_lane,
  const Vehicle& ego_vehicle,
  const std::map<Lane, std::vector<Vehicle>>& traffic) 
{
  double target_lane_speed = lane_speed(traffic.at(target_lane), ego_vehicle);

  float cost = (ego_vehicle.velocity - target_lane_speed) / ego_vehicle.velocity;
  if (cost < 0)
  {
    cost = 0;
  }
  return cost;
}


float calculate_cost(
  const Lane ego_lane,
  const Lane target_lane,
  const Vehicle& ego_vehicle,
  const std::map<Lane, std::vector<Vehicle>>& traffic) 
{
  // Sum weighted cost functions to get total cost for trajectory.
  float cost = 0.0;

  // Add additional cost functions here.
  vector<std::function<float(
    const Lane,
    const Lane,
    const Vehicle&, 
    const std::map<Lane, std::vector<Vehicle>>&)>> cf_list = 
    { goal_distance_cost, inefficiency_cost };

  vector<float> weight_list = { REACH_GOAL, EFFICIENCY };

  for (int i = 0; i < cf_list.size(); ++i) {
    float new_cost = weight_list[i] * cf_list[i](ego_lane, target_lane, ego_vehicle, traffic);
    cost += new_cost;
  }

  return cost;
}

/* Helper */

/*
* To get the speed limit for a lane, we find the first vehicle in front of us in that lane.
*/
float lane_speed(const vector<Vehicle>& sorted_lane_traffic, const Vehicle ego_vehicle) 
{
  double speed_behind = 0;
  double speed_ahead = 0;

  auto it_first = sorted_lane_traffic.begin();
  auto it_last = sorted_lane_traffic.begin();

  if (sorted_lane_traffic.size() > 0)
  {
    while (it_first->s_current > ego_vehicle.s_current && it_first != it_last) {
      ++it_first;
    }
    if (it_last - it_first > 0) {
      it_last = it_first; // vehicle behind
      speed_behind = it_last->velocity;
    }
    --it_first; // vehicle ahead
    speed_ahead = it_first->velocity;
    return (speed_behind > speed_ahead) ? speed_behind : speed_ahead;
  }
  return velocity.max; // the lane is empty
}

