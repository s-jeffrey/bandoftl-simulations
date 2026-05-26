import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys

def dynamics(positions, velocities, params, time, perturbed_vehicle):
    alpha = params['alpha']
    beta = params['beta']
    if perturbed_vehicle is not None:
        alpha = params['alpha_perturbed']
        beta = params['beta_perturbed']
    vmax_desired = params['vmax_desired']
    L = params['circum']
    vehicle_length = params['vehicle_length']

    xl = np.roll(positions, 1)
    vl = np.roll(velocities, 1)

    delta_x = (xl - positions) % L
    delta_v = vl - velocities

    v_optimal = vmax_desired * (np.tanh((delta_x - vehicle_length) / 2.5 - 2) + np.tanh(2))/(1 + np.tanh(2))

    positions_dot = velocities
    velocities_dot = alpha * (v_optimal - velocities) + beta * delta_v / delta_x ** 2
    return positions_dot, velocities_dot

def rk4(init_positions, init_velocities, params, dt, T):
    num_steps = int(T / dt)
    num_vehicles = len(init_positions)

    positions_history = np.zeros((num_steps + 1, num_vehicles))
    velocities_history = np.zeros((num_steps + 1, num_vehicles))
    perturbed_history = np.full((num_steps + 1,), None, dtype=object)
    
    positions_history[0] = init_positions
    velocities_history[0] = init_velocities
    perturbed_history[0] = None

    positions_curr = np.array(init_positions)
    velocities_curr = np.array(init_velocities)

    perturbed_vehicle = None
    perturbed_time_left = 0.0
    has_perturbed = False
    cooldown_time_left = 0.0

    safe_gap = 5.0

    for i in range(num_steps):
        time_curr = i * dt

        # check headway
        if perturbed_vehicle is not None:
                leader_id = (perturbed_vehicle - 1) % num_vehicles
                space_headway = (positions_curr[leader_id] - positions_curr[perturbed_vehicle]) % params['circum']
                if space_headway < safe_gap:
                    perturbed_vehicle = None
                    perturbed_time_left = 0.0
                    cooldown_time_left = 5.0

        # decide perturbation
        if perturbed_time_left <= 0:
            if perturbed_vehicle is not None:
                perturbed_vehicle = None
                cooldown_time_left = 5.0
            if time_curr >= 5.0 and not has_perturbed and cooldown_time_left <= 0:
                if np.random.rand() < 0.5:
                    perturbed_vehicle = np.random.randint(0, num_vehicles)
                    perturbed_time_left = np.random.uniform(2.0, 5.0)
                    # has_perturbed = True
                    print(f'perturbed at time {time_curr} on vehicle {perturbed_vehicle} for {perturbed_time_left}s')
        
        perturbed_history[i] = perturbed_vehicle

        # K1
        k1_pos_dot, k1_vel_dot = dynamics(positions_curr, velocities_curr, params, time_curr, perturbed_vehicle)

        # K2
        k2_pos_dot, k2_vel_dot = dynamics(
            positions_curr + 0.5 * dt * k1_pos_dot,
            velocities_curr + 0.5 * dt * k1_vel_dot,
            params, time_curr + 0.5 * dt, perturbed_vehicle
        )

        # K3
        k3_pos_dot, k3_vel_dot = dynamics(
            positions_curr + 0.5 * dt * k2_pos_dot,
            velocities_curr + 0.5 * dt * k2_vel_dot,
            params, time_curr + 0.5 * dt, perturbed_vehicle
        )

        # K4
        k4_pos_dot, k4_vel_dot = dynamics(
            positions_curr + dt * k3_pos_dot,
            velocities_curr + dt * k3_vel_dot,
            params, time_curr + dt, perturbed_vehicle
        )

        positions_curr += (dt / 6) * (k1_pos_dot + 2 * k2_pos_dot + 2 * k3_pos_dot + k4_pos_dot)
        positions_curr = positions_curr % params['circum']
        velocities_curr += (dt / 6) * (k1_vel_dot + 2 * k2_vel_dot + 2 * k3_vel_dot + k4_vel_dot)

        if perturbed_vehicle is not None:
            perturbed_time_left -= dt
        if cooldown_time_left > 0:
            cooldown_time_left -= dt

        positions_history[i + 1] = positions_curr
        velocities_history[i + 1] = velocities_curr

    perturbed_history[num_steps] = perturbed_vehicle

    return np.array(positions_history), np.array(velocities_history), perturbed_history

# for stability
def run_simulation(alpha, beta, params):
    params['alpha'] = alpha
    params['beta'] = beta

    positions_history, velocities_history = rk4(init_positions, init_velocities, params, dt, T)
    stability_metric = np.std(velocities_history[-1, :])
    return stability_metric

def create_animation(positions_history, params, filename='traffic_flow.gif', frame_step=5):
    num_vehicles = positions_history.shape[1]
    L = params['circum']
    R = L / (2 * np.pi)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal')
    ax.set_axis_off()

    # Draw the ring road
    theta_road = np.linspace(0, 2 * np.pi, 200)
    x_road = R * np.cos(theta_road)
    y_road = R * np.sin(theta_road)
    ax.plot(x_road, y_road, color='gray', linewidth=4, linestyle='--')
    ax.set_title("Ring Road Vehicle Simulation")

    # Initialize
    initial_positions = positions_history[0]
    initial_angles = (initial_positions / L) * 2 * np.pi
    x_init = R * np.cos(initial_angles)
    y_init = R * np.sin(initial_angles)
    # The leader is red, the followers are blue
    colors = ['red'] + ['blue'] * (num_vehicles - 1)
    scatter = ax.scatter(x_init, y_init, c=colors, s=80, zorder=5)

    # Add a text box
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12, family='monospace')

    def init():
        time_text.set_text('')
        return scatter, time_text

    def update(frame):
        current_positions = positions_history[frame]
        angles = (current_positions / L) * 2 * np.pi
        x = R * np.cos(angles)
        y = R * np.sin(angles)
        scatter.set_offsets(np.c_[x, y])

        perturbed_id = perturbed_history[frame]

        if perturbed_id is not None:
            current_colors = ['red' if j == perturbed_id else 'blue' for j in range(num_vehicles)]
            scatter.set_color(current_colors)
            scatter.set_linewidths([2.0 if j == perturbed_id else 0.5 for j in range(num_vehicles)])
        else:
            scatter.set_color(colors)
            scatter.set_linewidths(0.5)

        current_time = time[frame]
        time_text.set_text(f'time: {current_time:.2f}')
        return scatter, time_text

    # frames
    frames_to_render = range(0, len(positions_history), frame_step)
    
    print(f"Generating animation...")
    ani = animation.FuncAnimation(
        fig, 
        update, 
        frames=frames_to_render,
        init_func=init, 
        blit=False, 
        interval=20
    )
    ani.save(filename, writer='pillow', fps=30)
    plt.close(fig)
    print(f"Animation saved as {filename}")








if __name__ == "__main__":
    num_vehicles = 22

    params = {
            'alpha': 0.5,
            'beta': 20.0,
            'alpha_perturbed': 1.0,
            'beta_perturbed': 10.45**2,
            'vmax_desired': 12.0, # 40 / 3600 * 1000,
            'circum': 230.0,
            'vehicle_length': 4.0,
            'perturbed_time': 2.0,
        }

    dt = 0.01
    T = 100.0
    
    init_positions = np.zeros(num_vehicles)
    init_velocities = np.zeros(num_vehicles)
    
    # init leader
    xl0 = 0.0
    vl0 = 30 / 3600 * 1000
    vl0 = 12*(np.tanh((10.455-4)/2.5-2)+np.tanh(2))/(1+np.tanh(2))
    # vl0 = 1.0
    init_positions[0] = xl0
    init_velocities[0] = vl0

    # init followers
    init_headway = float(params['circum'] / num_vehicles)
    v0 = 30 / 3600 * 1000
    v0 = 12*(np.tanh((10.455-4)/2.5-2)+np.tanh(2))/(1+np.tanh(2))
    # v0 = 0.0

    print(f"{num_vehicles} vehicles evenly spaced {init_headway:.2f}m apart on ring with c={params['circum']}m")
    print(f"vl0 = {vl0}, v0 = {v0}")

    for i in range(1, num_vehicles):
        init_positions[i] = (init_positions[i - 1] - init_headway) % params['circum']
        init_velocities[i] = v0
    

    # # Test alpha/beta for stability
    # alphas = np.linspace(0.0, 5.0, 10)
    # betas = np.linspace(0.0, 200.0, 20)
    # results = np.zeros((len(alphas), len(betas)))
    # for i, a in enumerate(alphas):
    #     for j, b in enumerate(betas):
    #         results[i, j] = run_simulation(a, b, params)
    # plt.figure(figsize=(10, 7))
    # plt.contourf(betas, alphas, results, cmap='viridis')
    # plt.colorbar(label='Velocity Variance')
    # plt.xlabel('beta')
    # plt.ylabel('alpha')
    # plt.title('Stability Heatmap')
    # plt.savefig('stability_map.png')

    # simulation
    positions_history, velocities_history, perturbed_history = rk4(init_positions, init_velocities, params, dt, T)
    time = np.linspace(0, T, len(positions_history))


    # Plot positions
    plt.figure(figsize=(10, 7))
    for i in range(num_vehicles):
        pos = positions_history[:, i]
        pos_diff = np.diff(pos)
        wrap = np.where(pos_diff < -params['circum'] * 0.5)[0]
        pos_masked = np.array(pos, dtype=float)
        pos_masked[wrap + 1] = np.nan
        plt.plot(time, pos_masked, label=f'v{i}')
    plt.title('Positions vs Time')
    plt.xlabel('Time')
    plt.ylabel('Position m')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('pos.png')

    # Plot velocities
    plt.figure(figsize=(10, 7))
    for i in range(num_vehicles):
        plt.plot(time, velocities_history[:, i], label=f'v{i}')
    plt.title('Velocities vs Time')
    plt.xlabel('Time')
    plt.ylabel('Velocity m/s')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('vel.png')

    # Plot relative position vs time
    plt.figure(figsize=(10, 7))
    for i in range(1, num_vehicles):
        relative_position = (positions_history[:, i - 1] - positions_history[:, i]) % params['circum']
        plt.plot(time, relative_position, label=f'v{i-1} - v{i}')
    plt.title('Relative Positions vs Time')
    plt.xlabel('Time')
    plt.ylabel('Relative Position')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('pos_rel.png')

    # Plot relative velocity vs time
    plt.figure(figsize=(10, 7))
    for i in range(1, num_vehicles):
        relative_velocity = (velocities_history[:, i - 1] - velocities_history[:, i])
        plt.plot(time, relative_velocity, label=f'v{i-1} - v{i}')
    plt.title('Relative Velocities vs Time')
    plt.xlabel('Time')
    plt.ylabel('Relative Velocity')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('vel_rel.png')

    # Create animation
    create_animation(positions_history, params, filename='traffic_flow.gif')