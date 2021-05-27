#ifndef VEHICLE_H
#define VEHICLE_H

class Vehicle
{
public:
	//Vehicle(double s_curr = 0, double s_proj = 0, double vel = 0, bool shift = false);
	Vehicle(double s_curr, double s_proj, double vel)
			: s_current(s_curr), s_projected(s_proj), velocity(vel) {}

	~Vehicle() {}

	void move(double s_curr, double s_proj, double vel) {
		s_current = s_curr;
		s_projected = s_proj;
		velocity = vel;
	}

	double s_current;
	double s_projected;
	double velocity;
	//bool shift_pending;
private:

};

// dont need a separate cpp just for one operator
inline bool operator<(const Vehicle& first, const Vehicle& second)
{
	return first.s_current < second.s_current;
}

// sorts with descending s value
auto compare_pose = [](Vehicle& ahead, Vehicle& behind) { return ahead.s_current > behind.s_current; };

typedef struct {
	float min;
	float max;
	float step;
} Velocity;

enum Lane { left = 0, center = 1, right = 2, ignore = 3, size = 4 };

extern Velocity velocity;

#endif // !VEHICLE_H
