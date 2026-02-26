# © 2025 ETH Zurich, Robotic Systems Lab
# Author: Filip Bjelonic
# Licensed under the Apache License 2.0

"""Script to run an environment with zero action agent."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Pace agent for Isaac Lab environments.")
parser.add_argument("--num_envs", type=int, default=4096, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-Pace-Anymal-D-v0", help="Name of the task.")
parser.add_argument("--start_time", type=float, default=0.0, help="Skip scoring before this time (seconds). Still simulates through it.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import pace_sim2real.tasks  # noqa: F401
from pace_sim2real.utils import project_root
from pace_sim2real import CMAESOptimizer


def main():
    """Zero actions agent with Isaac Lab environment."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs
    )
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")

    # Create optimization
    bounds_params = env_cfg.sim2real.bounds_params.to(env.unwrapped.device)
    articulation = env.unwrapped.scene["robot"]
    joint_order = env_cfg.sim2real.joint_order
    sim_joint_ids = torch.tensor([articulation.joint_names.index(name) for name in joint_order], device=env.unwrapped.device)

    data_file = project_root() / "data" / env_cfg.sim2real.data_dir
    log_dir = project_root() / "logs" / "pace" / env_cfg.sim2real.robot_name

    data = torch.load(data_file)
    time_data = data["time"].to(env.unwrapped.device)
    measured_dof_pos = data["dof_pos"].to(env.unwrapped.device)

    # Detect torque vs position mode based on data keys
    torque_mode = "des_torque" in data
    if torque_mode:
        target_actions = data["des_torque"].to(env.unwrapped.device)
        if "dof_vel" in data:
            measured_dof_vel = data["dof_vel"].to(env.unwrapped.device)
            print("[INFO]: Torque-based sysid mode (velocity comparison, measured vel)")
        else:
            # Fallback: compute velocities via finite difference
            dt = time_data[1:] - time_data[:-1]
            measured_dof_vel = (measured_dof_pos[1:] - measured_dof_pos[:-1]) / dt.unsqueeze(-1)
            measured_dof_vel = torch.cat([torch.zeros(1, measured_dof_pos.shape[1], device=env.unwrapped.device), measured_dof_vel], dim=0)
            print("[INFO]: Torque-based sysid mode (velocity comparison, finite diff)")
    else:
        target_actions = data["des_dof_pos"].to(env.unwrapped.device)
        print("[INFO]: Position-based sysid mode (position comparison)")

    initial_dof_pos = measured_dof_pos[0, :].unsqueeze(0).repeat(env.unwrapped.num_envs, 1)

    time_steps = time_data.shape[0]
    sim_dt = env.unwrapped.sim.cfg.dt

    # Compute start index for scoring (skip early transient)
    score_start_idx = int((time_data >= args_cli.start_time).nonzero(as_tuple=True)[0][0]) if args_cli.start_time > 0 else 0
    if score_start_idx > 0:
        print(f"[INFO]: Scoring starts at t={args_cli.start_time}s (index {score_start_idx}/{time_steps})")

    opt = CMAESOptimizer(
        bounds=bounds_params,
        population_size=env.unwrapped.num_envs,
        log_dir=log_dir,
        joint_order=joint_order,
        max_iteration=env_cfg.sim2real.cmaes.max_iteration,
        data=data,
        device=env.unwrapped.device,
        epsilon=env_cfg.sim2real.cmaes.epsilon,
        sigma=env_cfg.sim2real.cmaes.sigma,
        save_interval=env_cfg.sim2real.cmaes.save_interval,
        save_optimization_process=env_cfg.sim2real.cmaes.save_optimization_process,
    )

    env.reset()
    opt.update_simulator(articulation, sim_joint_ids, initial_dof_pos)

    counter = 0
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # score: compare sim to real (only after start_time)
            if counter >= score_start_idx:
                if torque_mode:
                    articulaton_data = env.unwrapped.scene.articulations["robot"].data
                    sim_vel = articulaton_data.joint_vel[:, sim_joint_ids]
                    sim_pos = articulaton_data.joint_pos[:, sim_joint_ids]
                    opt.tell_vel(sim_vel, measured_dof_vel[counter, :].unsqueeze(0).repeat(env.unwrapped.num_envs, 1), sim_dof_pos=sim_pos)
                else:
                    opt.tell(env.unwrapped.scene.articulations["robot"].data.joint_pos[:, sim_joint_ids], measured_dof_pos[counter, :].unsqueeze(0).repeat(env.unwrapped.num_envs, 1))
            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            actions[:, sim_joint_ids] = target_actions[counter, :].unsqueeze(0).repeat(env.unwrapped.num_envs, 1)
            # apply actions
            env.step(actions)
            counter += 1
            if counter % 400 == 0:
                print(f"[INFO]: Step {counter * sim_dt:.1f} / {time_data[-1]:.1f} seconds ({counter / time_steps * 100:.1f} %)")
            if counter >= time_steps:
                print("[INFO]: Reached the end of the trajectory, exiting.")
                counter = 0
                opt.evolve()
                if opt.finished():
                    break
                env.reset()
                opt.update_simulator(env.unwrapped.scene["robot"], sim_joint_ids, initial_dof_pos)
    # close optimizer
    opt.close()
    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
