# **Path Planning**

---

## Rubric Points

### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/1971/view) individually and describe how I addressed each point in my implementation.


[//]: # (Image References)

[image1]: ./center_2021_02_02_10_42_06_349.jpg

---
### Compilation

#### 1. The code compiles correctly.

The compilation result is included in the /build folder. The source files are:
* BehaviorPlanner.cpp, BehaviorPlanner.h containing the plan algorithm
* Vehicle.h a class including some vehicle data
* cost.cpp, cost.h containing the cost functions
* main.cpp implementing the car movement
* helpers.h, json.hpp, spline.h as initialy included in the project

### Valid Trajectories

#### 1. The car is able to drive at least 4.32 miles without incident..

The top right screen of the simulator showing the current/best miles driven without incident counts at least 4.32 miles without incident (including exceeding acceleration/jerk/speed, collision, and driving outside of the lanes).

#### 2. The car drives according to the speed limit.

The car does not exceed a hard-coded speed limit of 49.5 mph, as .max member of the Velocity type. A global variable of this type is defined in main.cpp. This variable also defines a lower speed .min 0 mph and an advance/reduce step of .224 mph.

#### 3. Max Acceleration and Jerk are not Exceeded.

The car does not exceed a total acceleration of 10 m/s^2 and a jerk of 10 m/s^3. The limiting of max accelaration and jerk is reached by two means:
* Using the spline.h header we achieve a smooth change from lane to lane. The smoothness of the interpolated spline, is guaranteed by reusing two path points past the current position of the vehicle.
* Also in BehaviorPlanner.cpp (line 63) a limit for enabling turns form lane to lane is set, to ensure a gradual change when the vehicle moves to two lanes consequently.

#### 4. Car does not have collisions.

The car must not come into contact with any of the other cars on the road. This is achieved by:
* Not allowing a lane change when there is less than safety_distance available behind and ahead the ego vehicle, in the aimed lane.
* If a vehicle is sensed ahead in with projected s (value to achieve in the future) position more than the projected s of the ego vehicle, the ego vehicle follows its speed.
* If this difference in the projected s  positions is less than the safety_distance, the ego velocity is decrease appropriately.

#### 5. The car stays in its lane, except for the time between changing lanes.

The car doesn't spend more than a 3 second length out side the lane lanes during changing lanes, and every other time the car stays inside one of the 3 lanes on the right hand side of the road. This behavior is also achieved.

#### 6. The car is able to change lanes

The car is able to smoothly change lanes when it makes sense to do so, such as when behind a slower moving car and an adjacent lane is clear of other traffic. This behavior is also achieved, incorporating the evaluation of costs functions implemented in cost.cpp. The vehicle changes the lane, if a vehicle ahead, on the same lane is limiting the ego vehicles velocity, and when it is safe to do so.

### General Comments 

The implementation is base on the code provided on the prject's Q&A section. Additionall meaningfull variables where defined and magic numbers were eliminated. The sample code was also restructured to allow a quick understanding of how the vehicle moves along.

To add a realistic behavior, the vehicle choses to turn to a lane on the right of the road, if there is one available, and such a turn is safe.

#### Behavior planning

The behavior planning is implemented in the method path_plan of the class BehaviorPlanner. The progress is divided in 3 parts which could also be implemented as helper functions.
* First, the traffic on the revant side of the road is analysed, and the sensed vehicles, are sorted.
* Second, the move planning is done utilizing the cost functions.
* Finally, the vehicles velocity is adjusted accordingly.

#### Cost functions
To evaluate the traffic in each line, the sensed vehicles are sorted according to their s position in each lane. The cost functions are evaluating the same properties as in the course, and the weighting is kept the same. However the functions are reimplemented, as the vehicle does not use a state machine any more, but evaluates the traffic in the closest lanes.

To enable the vehicle movement, if a lane other than a next neighboring is empty, and if a turn to the interim lane is safe, the cost for moving to the interim lane is divided by 2.
 
### Outlook

Even if the velocity is limited, when obstacles are sensed, this process is not smooth, leading to the vehicle over using the brakes. Here there would be space for the implementation of a controller setting the vehicles speed.

The vehicle could sense also the space behind its current s position in the neighboring lanes, if it has repeatedly reduced speed for a consecutive couple of times. This could enable tracking the speed of a vehicle behind in the neighboring lane, and trying to get behind it, and change to the other available lane. The implementation of such a behavior however is more complex than the current one.


