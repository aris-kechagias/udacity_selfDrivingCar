#include "BehaviorPlanner.h"

#include <cmath>

#include <vector>
#include <map>
#include <iterator>
#include <algorithm>

#include "cost.h"
#include <iostream>
#include <string>
Lane BehaviorPlanner::plan_path(
	const std::vector<std::vector<double>>& sensed_vehicles, 
	int prev_path_size,
	Lane ego_lane,
	double ego_s,
	double ego_d,
	double ego_s_projected,
	float* ego_velocity)
{
	Lane out = ego_lane;

	std::map<Lane, std::vector<Vehicle>> traffic = {
		{Lane::left, {}},
		{Lane::center, {}},
		{Lane::right, {}}
	};

	/* analyze traffic */
	for (auto const & vehicle : sensed_vehicles)
	{
		float displacement = vehicle[6];
		Lane lane = find_lane(displacement);

		if (lane != Lane::ignore)
		{
			double velocity = std::sqrt(vehicle[3] * vehicle[3] + vehicle[4] * vehicle[4]); // [3] = vx, [4] = vy
			double s_per_waypoint = 0.02 * velocity; // one waypoint every .02s
			double  s_current = vehicle[5];
			double s_projected = s_current + prev_path_size * s_per_waypoint;

			Vehicle temp_veh(s_current, s_projected, velocity);

			bool we_urge = (temp_veh.s_projected - ego_s_projected < safety_distance);
			bool they_urge = (ego_s_projected - temp_veh.s_projected < safety_distance);

			if (((ego_s < temp_veh.s_current) && we_urge) ||
				  ((ego_s > temp_veh.s_current) && they_urge && lane != ego_lane))
			{
				traffic[lane].push_back(temp_veh); // car on the way
			}
		}
	}
	// and sort traffic in each lane
	for (auto& elem: traffic) {
		auto it_first = elem.second.begin();
		auto it_last = elem.second.end();
		std::sort(it_first, it_last, compare_pose);
	}

	/* do planning */
	bool allow_turn = (ego_d > ego_lane * Lane::size + 1) && (ego_d < ego_lane* Lane::size + Lane::size - 1);

	if ( allow_turn && 
		*ego_velocity > velocity.max / 2 &&
		ego_lane != Lane::right &&
		traffic[ego_lane].size() == 0 && // cannot occure together with the next if
		traffic[(Lane)(ego_lane + 1)].size() == 0)
	{
		out = (Lane)(ego_lane + 1);
	}
	if (allow_turn && traffic[ego_lane].size() > 0) //  obstacle_inlane
	{
		Vehicle ego_vehicle(ego_s, ego_s_projected, *ego_velocity);
		std::vector<double> costs;
		for (auto& ln : available_turns.at(ego_lane))
		{
			costs.push_back(calculate_cost(ego_lane, ln, ego_vehicle, traffic));
		}
		int minElementIdx = std::min_element(costs.begin(), costs.end()) - costs.begin();
		out = available_turns.at(ego_lane)[minElementIdx];
	}

	/* adjust speed */
	if (traffic[ego_lane].size() > 0) // obstacle in lane
	{ 
		auto it = traffic[ego_lane].begin();
		while (it->s_current > ego_s) {
			++it;
		}
		Vehicle temp = *--it;
		double distance = temp.s_projected - ego_s_projected;
		
		if (distance > safety_distance) {
			*ego_velocity = temp.velocity;
		} else if (distance < safety_distance) { // dangerous zone
			*ego_velocity -= 2*velocity.step;
		} /*else {
			*ego_velocity -= 2*velocity.step/3;
		}*/
	} 
	else {
		if (*ego_velocity < velocity.max) {
			*ego_velocity += velocity.step;
		}
	}
	return out;
}
