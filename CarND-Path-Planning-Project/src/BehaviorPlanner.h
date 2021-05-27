#ifndef BEHAVIORPLANNER_H
#define BEHAVIORPLANNER_H

#include <vector>
#include <map>
#include "Vehicle.h"

extern Velocity velocity;


class BehaviorPlanner
{
public:
  BehaviorPlanner() : available_turns({
    {Lane::left, {Lane::left, Lane::center}}, //, Lane::right
    {Lane::center, {Lane::center, Lane::left, Lane::right}},
    {Lane::right, {Lane::right, Lane::center}} //, Lane::left
  }), safety_distance(20){}

  ~BehaviorPlanner() {}

	Lane plan_path(const std::vector<std::vector<double>>& sensed_vehicles,
		int prev_path_size,
    Lane ego_lane,
    double ego_s,
    double ego_d,
    double ego_s_projected,
		float* ego_velocity);

private:
  // lanes start count from the leftmost point of the left lane
  Lane find_lane(float obj_displacement) {
    int out = Lane::left;

    if (obj_displacement < Lane::left ||
        obj_displacement > Lane::right * Lane::size + Lane::size)
    {
      out = ignore;
    }
    else
    {
      while (out != Lane::right)
      {
        if ((obj_displacement > out * Lane::size) &&
          (obj_displacement < out * Lane::size + Lane::size)) break;

        ++out;
      }
    }

    return (Lane)out;
  }

  int safety_distance;
  std::map<Lane, std::vector<Lane>> available_turns;
};

#endif // !BEHAVIORPLANNER_H