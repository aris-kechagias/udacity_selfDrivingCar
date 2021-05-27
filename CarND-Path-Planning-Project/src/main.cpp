#include <uWS/uWS.h>
#include <fstream>
#include <iostream>
#include "Eigen-3.3/Eigen/Core"
#include "Eigen-3.3/Eigen/QR"
#include "json.hpp"
#include "spline.h"

#include "BehaviorPlanner.h"
#include "helpers.h"

using nlohmann::json;
using std::string;
using std::vector;

Velocity velocity = { .0f, 49.5f, .224f };

int main() {
  uWS::Hub h;

  // Load up map values for waypoint's x,y,s and d normalized normal vectors
  vector<double> map_waypoints_x;
  vector<double> map_waypoints_y;
  vector<double> map_waypoints_s;
  vector<double> map_waypoints_dx;
  vector<double> map_waypoints_dy;

  // Waypoint map to read from
  string map_file_ = "../data/highway_map.csv";
  // The max s value before wrapping around the track back to 0
  double max_s = 6945.554;

  std::ifstream in_map_(map_file_.c_str(), std::ifstream::in);

  string line;
  while (getline(in_map_, line)) {
    std::istringstream iss(line);
    double x;
    double y;
    float s;
    float d_x;
    float d_y;
    iss >> x;
    iss >> y;
    iss >> s;
    iss >> d_x;
    iss >> d_y;
    map_waypoints_x.push_back(x);
    map_waypoints_y.push_back(y);
    map_waypoints_s.push_back(s);
    map_waypoints_dx.push_back(d_x);
    map_waypoints_dy.push_back(d_y);
  }

  // init 
  BehaviorPlanner planner;
  Lane ego_lane = Lane::center; //  start lane
  float ego_velocity = velocity.min; // limit = 50

  //Vehicle ego_vehicle(s, 0, velocity.min, false);
  h.onMessage(
  [&map_waypoints_x,&map_waypoints_y,&map_waypoints_s,
    &map_waypoints_dx,&map_waypoints_dy, &ego_lane, &ego_velocity, &planner]
  (uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length, uWS::OpCode opCode)
  {
    // "42" at the start of the message means there's a websocket message event.
    // The 4 signifies a websocket message
    // The 2 signifies a websocket event
    if (length && length > 2 && data[0] == '4' && data[1] == '2') {

      auto s = hasData(data);

      if (s != "") {
        auto j = json::parse(s);
        
        string event = j[0].get<string>();
        
        if (event == "telemetry") {
          // Main car's localization Data;
          double car_x = j[1]["x"]; // j[1] is the data JSON object
          double car_y = j[1]["y"];
          double car_s = j[1]["s"];
          double car_d = j[1]["d"];
          double car_yaw = j[1]["yaw"];
          double car_speed = j[1]["speed"];
          // Previous path data given to the Planner
          auto previous_path_x = j[1]["previous_path_x"];
          auto previous_path_y = j[1]["previous_path_y"];
          // Previous path's end s and d values 
          double end_path_s = j[1]["end_path_s"];
          double end_path_d = j[1]["end_path_d"];
          // A list of all other cars on the same side of the road.
          auto cars_around = j[1]["sensor_fusion"];

          int prev_path_size = previous_path_x.size();
          double ego_s_projected = (prev_path_size > 0) ? end_path_s :  car_s;

          ego_lane = planner.plan_path(cars_around, prev_path_size, ego_lane, car_s, car_d, ego_s_projected, &ego_velocity);
          
          //reference x,y,yaw states
          //reference point is either the starting point where the car is
          //or the end point of the previous path
          double ref_x;
          double ref_y;
          double ref_yaw;
          double ref_x_prev;
          double ref_y_prev;

          if (prev_path_size < 3)
          { // previous path too short -> current point as reference
            ref_x = car_x;
            ref_y = car_y;
            ref_yaw = deg2rad(car_yaw);

            // using a projected point that make the path tangent to the car
            ref_x_prev = car_x - cos(car_yaw);
            ref_y_prev = car_y - sin(car_yaw);
          }
          else
          { // previous path end point as reference
            ref_x = previous_path_x[prev_path_size - 1];
            ref_y = previous_path_y[prev_path_size - 1];
            ref_yaw = atan2(ref_y - ref_y_prev, ref_x - ref_x_prev);

            ref_x_prev = previous_path_x[prev_path_size - 2];
            ref_y_prev = previous_path_y[prev_path_size - 2];
          }

          // Waypoints (x,y), evenly spaced every 30m; to be interpolated with a spline
          vector<double> sparse_ptsx{};
          vector<double> sparse_ptsy{};

          sparse_ptsx.push_back(ref_x_prev);
          sparse_ptsy.push_back(ref_y_prev);

          sparse_ptsx.push_back(ref_x);
          sparse_ptsy.push_back(ref_y);

          //In Frenet add evenly 30m points ahead of the starting reference // instead of 50 points
          double lateral_disp = Lane::size * ego_lane + 2; // 2 for the middle of lane

          vector<double> next_wp0 = getXY(ego_s_projected + 30, lateral_disp, map_waypoints_s, map_waypoints_x, map_waypoints_y);
          sparse_ptsx.push_back(next_wp0[0]);
          sparse_ptsy.push_back(next_wp0[1]);

          vector<double> next_wp1 = getXY(ego_s_projected + 60, lateral_disp, map_waypoints_s, map_waypoints_x, map_waypoints_y);
          sparse_ptsx.push_back(next_wp1[0]);
          sparse_ptsy.push_back(next_wp1[1]);
          
          vector<double> next_wp2 = getXY(ego_s_projected + 90, lateral_disp, map_waypoints_s, map_waypoints_x, map_waypoints_y);
          sparse_ptsx.push_back(next_wp2[0]);
          sparse_ptsy.push_back(next_wp2[1]); // 5 points into sparse_ptsx up to here

          // Convert the waypoints from map coordinates to vehicle coordinates 
          for (size_t i = 0; i < sparse_ptsx.size(); i++)
          {  
            // First shift; ensures that no functions are near vertircal
            double shift_x = sparse_ptsx[i] - ref_x;
            double shift_y = sparse_ptsy[i] - ref_y;

            // Then rotate the path to reach a car reference angle of 0 degrees
            sparse_ptsx[i] = shift_x * cos(0 - ref_yaw) - shift_y * sin(0 - ref_yaw);
            sparse_ptsy[i] = shift_x * sin(0 - ref_yaw) + shift_y * cos(0 - ref_yaw);
          }

          tk::spline s;
          s.set_points(sparse_ptsx, sparse_ptsy);

          vector<double> next_x_vals;
          vector<double> next_y_vals;

          for (size_t i = 0; i < prev_path_size; i++)
          { // adding all the previous path points for smoothnes
            next_x_vals.push_back(previous_path_x[i]);
            next_y_vals.push_back(previous_path_y[i]);
          }

          // Calculate how to break up spline points so that we travel at our desired reference velocity
          double target_x = 30.0; //horizon value
          double target_y = s(target_x);
          double target_dist = sqrt(target_x * target_x + target_y * target_y);

          double x_add_on = 0;

          // portion of last path not driven by the car
          for (size_t i = 1; i <= 50 - prev_path_size; i++)
          { // Fill up the rest of our path planner after filling it with the previous points
            double N = target_dist / (.02 * ego_velocity / 2.24); // 2.24 mph -> meterpersec
            double x_point = x_add_on + target_x / N;
            double y_point = s(x_point);

            x_add_on = x_point;

            // rotate back to normal after previous rotation (global coordinates)
            double x_ref = x_point;
            double y_ref = y_point;
            x_point = x_ref * cos(ref_yaw) - y_ref * sin(ref_yaw);
            y_point = x_ref * sin(ref_yaw) + y_ref * cos(ref_yaw);
            x_point += ref_x;
            y_point += ref_y;

            next_x_vals.push_back(x_point);
            next_y_vals.push_back(y_point);
          }

          json msgJson;

          msgJson["next_x"] = next_x_vals;
          msgJson["next_y"] = next_y_vals;

          auto msg = "42[\"control\","+ msgJson.dump()+"]";

          ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
        }  // end "telemetry" if
      } else {
        // Manual driving
        std::string msg = "42[\"manual\",{}]";
        ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
      }
    }  // end websocket if
  }); // end h.onMessage

  h.onConnection([&h](uWS::WebSocket<uWS::SERVER> ws, uWS::HttpRequest req) {
    std::cout << "Connected!!!" << std::endl;
  });

  h.onDisconnection([&h](uWS::WebSocket<uWS::SERVER> ws, int code,
                         char *message, size_t length) {
    ws.close();
    std::cout << "Disconnected" << std::endl;
  });

  int port = 4567;
  if (h.listen(port)) {
    std::cout << "Listening to port " << port << std::endl;
  } else {
    std::cerr << "Failed to listen to port" << std::endl;
    return -1;
  }
  
  h.run();
}