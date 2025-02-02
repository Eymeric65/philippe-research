import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --- Simulation Parameters ---
box_width, box_height = 100, 100   # Dimensions of the box
timesteps = 1000                   # Total number of time steps
p_ray = 0.05                   # Probability of emitting a ray at each step
ray_length = 2                    # Length of each emitted ray

h, k, r = 50, 50, np.sqrt(2)*50  # Circle center and radius

# Particle parameters
init_position = np.array([50.0, 50.0])
init_speed = 0.5
init_angle = np.random.uniform(0, 2*np.pi)

# --- Particle Class ---
class Particle:
    def __init__(self, position, speed, angle,prob_ray):
        self.position = position.copy()
        self.speed = speed
        self.angle = angle  # current heading in radians
        self.prob_ray = prob_ray

    def update(self):
        # Update heading with a small random change (simulating smooth, car-like motion)
        self.angle += np.random.normal(scale=0.2)
        self.speed += np.random.normal(scale=0.01)

        dx = self.speed * np.cos(self.angle)
        dy = self.speed * np.sin(self.angle)
        new_position = self.position + np.array([dx, dy])
        
        # Reflect off the vertical walls
        if new_position[0] < 0 or new_position[0] > box_width:
            self.angle = np.pi - self.angle
            new_position[0] = np.clip(new_position[0], 0, box_width)
        # Reflect off the horizontal walls
        if new_position[1] < 0 or new_position[1] > box_height:
            self.angle = -self.angle
            new_position[1] = np.clip(new_position[1], 0, box_height)
        
        self.position = new_position

        ray=None
        if np.random.random()<self.prob_ray:
            
            u,v = self._generate_ray()

            amp = np.random.random()*6 + 4 # defform the output to blur the true position


            ray=[new_position[0]+u*amp,new_position[1]+v*amp,u,v]

        return self.position,ray
    
    def _generate_ray(self):

        raydir= np.random.random()*2*np.pi

        return np.cos(raydir),np.sin(raydir)

# --- Initialize Simulation ---
t=1

particle = Particle(init_position, init_speed, init_angle,p_ray)
positions = [init_position.copy()]  # Store trajectory positions
time=[t]
rays = []  # List to store emitted rays as tuples: (start_point, ray_direction)
rays_time=[]
for i in range(timesteps):

    t+=1

    new_pos,new_ray = particle.update()
    positions += [new_pos]
    time+= [t]

    if new_ray is not None:
        rays += [new_ray]
        rays_time+= [t]
    
#print(rays)

positions = np.array(positions)
rays = np.array(rays)
time = np.array(time)

# --- Create the optimiser ---

A=np.zeros((len(rays),)*2)
B=np.zeros((len(rays),1))

for i in range(len(rays)-1):
    A_P = np.array([[
        rays[i,2]**2+rays[i,3]**2,-(rays[i,3]*rays[i+1,3]+rays[i,2]*rays[i+1,2])
    ],[
        -(rays[i,3]*rays[i+1,3]+rays[i,2]*rays[i+1,2]),rays[i+1,2]**2+rays[i+1,3]**2
    ]])

    theta_x = rays[i,0] - rays[i+1,0]
    theta_y = rays[i,1] - rays[i+1,1]

    B_P = np.array([[(theta_x*rays[i,2]+theta_y*rays[i,3])],[-(theta_x*rays[i+1,2]+theta_y*rays[i+1,3])]])

    A[i:i+2,i:i+2]+=A_P
    B[i:i+2]+=B_P


X = -1/2*np.linalg.pinv(A) @ B / 2

## Create set of point

position_retrieved=[]

for i in range(len(rays)):
    position_retrieved += [[rays[i,0]+rays[i,2]*X[i,0],rays[i,1]+rays[i,3]*X[i,0]]]

position_retrieved=np.array(position_retrieved)

## Philippe method 

def line_circle_intersection(x0, y0, dx, dy, h, k, r):
    # Quadratic equation coefficients
    A = dx**2 + dy**2
    B = 2 * (dx * (x0 - h) + dy * (y0 - k))
    C = (x0 - h)**2 + (y0 - k)**2 - r**2

    # Solve quadratic equation A*t^2 + B*t + C = 0
    discriminant = B**2 - 4*A*C

    if discriminant < 0:
        return None  # No real intersections
    elif discriminant == 0:
        t = -B / (2 * A)
        return [(x0 + t * dx, y0 + t * dy)]  # One intersection (tangent)
    else:
        sqrt_disc = np.sqrt(discriminant)
        t1 = (-B + sqrt_disc) / (2 * A)
        t2 = (-B - sqrt_disc) / (2 * A)
        
        # Compute intersection points
        p1 = np.array([x0 + t1 * dx, y0 + t1 * dy])
        p2 = np.array([x0 + t2 * dx, y0 + t2 * dy])
        return (p1+p2)/2
    
def moving_average(arr, window_size):
    kernel = np.ones(window_size) / window_size
    return np.convolve(arr, kernel, mode='same')

position_philippe=[]

for i in range(len(rays)):

    position_philippe+=[line_circle_intersection(rays[i,0], rays[i,1], rays[i,2], rays[i,3], h, k, r)]

position_philippe = np.array(position_philippe)

window_size=int(len(rays)/20)

position_philippe[:,0] =moving_average(position_philippe[:,0],window_size)
position_philippe[:,1] =moving_average(position_philippe[:,1],window_size)


print(X)

# --- Set Up Plot ---

label_gt = "ground truth"
label_opt = "minimisation method"
label_mav = "center average method"

fig = plt.figure(figsize=(9, 12), constrained_layout=True)
gs = fig.add_gridspec(3, 1, height_ratios=[4, 1, 1])

ax = fig.add_subplot(gs[0])
ax.plot(positions[:,0],positions[:,1],label=label_gt)
ax.plot(position_retrieved[:,0],position_retrieved[:,1],label=label_opt)
ax.plot(position_philippe[:,0],position_philippe[:,1],label=label_mav)

# for ray in rays:
#     ray_start= ray[:2] -ray_length*ray[2:]
#     ray_end= ray[:2] +ray_length*ray[2:]
#     ax.plot([ray_start[0],ray_end[0]],[ray_start[1],ray_end[1]],"red")

ax.set_xlim(0, box_width)
ax.set_ylim(0, box_height)
ax.set_title('Particle Trajectory with Random Ray Emissions')
ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
ax.set_aspect("equal")

ax.legend()

ax = fig.add_subplot(gs[1])
ax.plot(time,positions[:,0],label=label_gt)
ax.plot(rays_time,position_retrieved[:,0],label=label_opt)
ax.plot(rays_time,position_philippe[:,0],label=label_mav)
ax.set_title("X coordinate of particle")
ax.set_ylabel('X Position')
ax.set_xlabel('Time')
ax.legend()

ax = fig.add_subplot(gs[2])
ax.plot(time,positions[:,1],label=label_gt)
ax.plot(rays_time,position_retrieved[:,1],label=label_opt)
ax.plot(rays_time,position_philippe[:,1],label=label_mav)
ax.set_title("Y coordinate of particle")
ax.set_ylabel('Y Position')
ax.set_xlabel('Time')
ax.legend()

plt.tight_layout()

plt.savefig("plot.png", dpi=300, bbox_inches="tight")

#plt.show()
