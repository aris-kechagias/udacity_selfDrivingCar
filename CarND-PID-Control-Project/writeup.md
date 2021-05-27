# **PID Controller**

---

## Rubric Points

### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/1972/view) individually and describe how I addressed each point in my implementation.

---
### Compilation

#### 1. The code compiles correctly.
The compilation result is included in the /build folder. The source files are:
* Twiddle.cpp providing the implementation of the twiddle tunning algorithm
* PID.cpp providing the controller implementation
* main.cpp implementing the car movement

#### 2. Implementation

The PID procedure follows what was taught in the lessons. The PID implementation is kept minimal, updating the controller coefficients with every measurement.

Additionally, the output limiter is integrated as inline member function.

The Twiddle algorithm follows the implementation provided in the class, however it is modified to allow real-time operation.

Specifically, a coefficient is updated in every cycle. As in real-time it is not possible to update the measurements while the algorithm is running the following changes were made:
* The `while` loop is replaced with an `if` condition.
* The algorithm is divided in 2 "states" dictated from the points where a measurement update is needed.
* The `for` loop in the class example is removed, as the algorithm does not pause during its execution for receiving new samples. Instead an "index" variable is introduced (as shown in several posts in knowledge), which is changed when a full run of the `for` loop in the class finishes. This lasts 2 cycles at maximum, in case that the second algorihtm state is entered for a coefficient.
* The initial coefficient update step is moved to the end  to allow a new sample comming in, before the condition "sum of params > tolerance". This change does not make any difference, as the parameters are updated with every sample, and not every 200 points as in the class example.
* The error and step counting variables are reset every 700 cycles, to prevent an overflow. This step was initially in the `run` function in the lesson, but here is integrated in the twiddle algorithm.

#### 3. Reflection

##### Visual aids

A demo record without online tuning is here: ![](self_driving_car_nanodegree_program_noTwiddle.mp4), while another with twiddle is here: ![](self_driving_car_nanodegree_program_withTwiddle.mp4)

##### Effect of the P, I, D components in the implementation.

Moving the first parameter update to the end of the algorithm, and also the update of the coefficients in every cycle does not have a negative effect to the algorithm, as the first (100) samples are used to "load" the algorithm and update the quadratic error, without changing the controller coefficients.

I found that longer twiddle update cycles (say 700 samples) have a stabilizing effect during the algorithm run.

Trhough trial-and-error I chose a value for Kp 0.13 and for Ki 0.0005. Big values for Ki lead the vehicle quickly off the road. For Kd I chose the value 3.5 with a ratio to Kp about 2.5/1 to allow for quickier damping of  of oscilations, when the curve of the road changes. Specifically, I found that higher ratios, are beneficial in higher velocities, but the other parameters have to be updated appropriately. With those parameters, and twiddle tuning, the vehicle could drive up to 50 mph.

A controller for the velocity, is not implemented, but an idea would be to update it also considering the cross track error, and lower the speed when the vehicle drives off the road center. 

##### Final hyperparameter update

There is no filnal coefficient choice, as the coefficients are tuned online with the twiddle algorithm. As described in the implementation part.

