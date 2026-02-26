# Manual parameter testing for torque-based sysid
# Usage:
#   python scripts/pace/manual_test.py --armature 0.006 0.005 --viscous 0.01 0.01 \
#       --friction 0.5 0.5 --bias 0.05 -0.19 --delay 4 \
#       --data bracketbot/bb_torque_chirp_data_1.0Nm.pt

"""Launch Isaac Sim Simulator first."""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Manual sysid parameter testing.")
parser.add_argument("--armature", type=float, nargs=2, default=[0.006, 0.005], help="Armature [left, right]")
parser.add_argument("--viscous", type=float, nargs=2, default=[0.01, 0.01], help="Viscous friction [left, right]")
parser.add_argument("--friction", type=float, nargs=2, default=[0.5, 0.5], help="Static/Coulomb friction [left, right]")
parser.add_argument("--bias", type=float, nargs=2, default=[0.0, 0.0], help="Encoder bias [left, right]")
parser.add_argument("--delay", type=int, default=4, help="Action delay in sim steps")
parser.add_argument("--data", type=str, default="bracketbot/bb_torque_chirp_data_1.0Nm.pt", help="Data file relative to data/")
parser.add_argument("--out", type=str, default=None, help="Output image path (default: logs/pace/bb/manual_test.png)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.num_envs = 1

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch
import numpy as np
import matplotlib.pyplot as plt

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
import pace_sim2real.tasks  # noqa: F401
from pace_sim2real.utils import project_root


def main():
    env_cfg = parse_env_cfg("Isaac-Pace-BB-Torque-v0", device=args_cli.device, num_envs=1)
    env_cfg.sim2real.data_dir = args_cli.data
    env = gym.make("Isaac-Pace-BB-Torque-v0", cfg=env_cfg)

    articulation = env.unwrapped.scene["robot"]
    joint_order = env_cfg.sim2real.joint_order
    sim_joint_ids = torch.tensor([articulation.joint_names.index(name) for name in joint_order], device=env.unwrapped.device)

    data_file = project_root() / "data" / args_cli.data
    data = torch.load(data_file)
    time_data = data["time"].to(env.unwrapped.device)
    measured_dof_pos = data["dof_pos"].to(env.unwrapped.device)
    target_torques = data["des_torque"].to(env.unwrapped.device)
    measured_dof_vel = data.get("dof_vel")
    if measured_dof_vel is not None:
        measured_dof_vel = measured_dof_vel.to(env.unwrapped.device)

    initial_dof_pos = measured_dof_pos[0, :].unsqueeze(0)
    time_steps = time_data.shape[0]

    # Apply parameters
    env.reset()
    device = env.unwrapped.device
    env_ids = torch.arange(1, device=device)

    armature = torch.tensor([args_cli.armature], device=device)
    viscous = torch.tensor([args_cli.viscous], device=device)
    friction = torch.tensor([args_cli.friction], device=device)
    bias = torch.tensor([args_cli.bias], device=device)
    delay = torch.tensor([[args_cli.delay]], dtype=torch.int, device=device)

    articulation.write_joint_armature_to_sim(armature, joint_ids=sim_joint_ids, env_ids=env_ids)
    articulation.data.default_joint_armature[:, sim_joint_ids] = armature
    articulation.write_joint_viscous_friction_coefficient_to_sim(viscous, joint_ids=sim_joint_ids, env_ids=env_ids)
    articulation.data.default_joint_viscous_friction_coeff[:, sim_joint_ids] = viscous
    articulation.write_joint_friction_coefficient_to_sim(friction, joint_ids=sim_joint_ids, env_ids=env_ids)
    articulation.data.default_joint_friction_coeff[:, sim_joint_ids] = friction
    articulation.write_joint_position_to_sim(initial_dof_pos + bias, joint_ids=sim_joint_ids)
    articulation.write_joint_velocity_to_sim(torch.zeros_like(initial_dof_pos), joint_ids=sim_joint_ids)

    for drive_type in articulation.actuators.keys():
        drive_indices = articulation.actuators[drive_type].joint_indices
        if isinstance(drive_indices, slice):
            all_idx = torch.arange(sim_joint_ids.shape[0], device=device)
            drive_indices = all_idx[drive_indices]
        comparison_matrix = (sim_joint_ids.unsqueeze(1) == drive_indices.unsqueeze(0))
        drive_joint_idx = torch.argmax(comparison_matrix.int(), dim=0)
        articulation.actuators[drive_type].update_encoder_bias(bias[:, drive_joint_idx])
        articulation.actuators[drive_type].update_time_lags(delay)
        articulation.actuators[drive_type].reset(env_ids)

    # Run sim
    sim_pos_buf = torch.zeros(time_steps, len(joint_order), device=device)
    sim_vel_buf = torch.zeros(time_steps, len(joint_order), device=device)

    print(f"Running sim with: armature={args_cli.armature}, viscous={args_cli.viscous}, "
          f"friction={args_cli.friction}, bias={args_cli.bias}, delay={args_cli.delay}")

    for counter in range(time_steps):
        with torch.inference_mode():
            art_data = articulation.data
            sim_pos_buf[counter] = art_data.joint_pos[0, sim_joint_ids]
            sim_vel_buf[counter] = art_data.joint_vel[0, sim_joint_ids]

            actions = torch.zeros(env.action_space.shape, device=device)
            actions[:, sim_joint_ids] = target_torques[counter].unsqueeze(0)
            env.step(actions)

        if counter % 400 == 0:
            print(f"  Step {counter}/{time_steps} ({counter/time_steps*100:.0f}%)")

    env.close()

    # Plot
    time_np = time_data.cpu().numpy()
    real_pos_np = measured_dof_pos.cpu().numpy()
    sim_pos_np = np.zeros_like(sim_pos_buf.cpu().numpy())
    for i in range(len(joint_order)):
        sim_pos_np[:, i] = np.unwrap(sim_pos_buf[:, i].cpu().numpy())
    sim_vel_np = sim_vel_buf.cpu().numpy()

    if measured_dof_vel is not None:
        real_vel_np = measured_dof_vel.cpu().numpy()
    else:
        dt_np = np.diff(time_np)
        real_vel_np = np.diff(real_pos_np, axis=0) / dt_np[:, None]
        real_vel_np = np.vstack([np.zeros((1, len(joint_order))), real_vel_np])

    fig, axes = plt.subplots(2, 3, figsize=(18, 8))
    title = (f"armature={args_cli.armature}  viscous={args_cli.viscous}  "
             f"friction={args_cli.friction}  bias={args_cli.bias}  delay={args_cli.delay}")
    fig.suptitle(title, fontsize=10)

    for i in range(len(joint_order)):
        # Position
        axes[i, 0].plot(time_np, sim_pos_np[:, i], c="tab:red", label="Sim", linewidth=0.8)
        axes[i, 0].plot(time_np, real_pos_np[:, i], c="tab:blue", label="Real", linewidth=0.8)
        axes[i, 0].set_title(f"{joint_order[i]} — Position")
        axes[i, 0].set_ylabel("Position [rad]")
        axes[i, 0].legend()
        axes[i, 0].grid(True, alpha=0.3)

        # Velocity full
        axes[i, 1].plot(time_np, sim_vel_np[:, i], c="tab:red", label="Sim", linewidth=0.6, alpha=0.8)
        axes[i, 1].plot(time_np, real_vel_np[:, i], c="tab:blue", label="Real", linewidth=0.6, alpha=0.8)
        axes[i, 1].set_title(f"{joint_order[i]} — Velocity")
        axes[i, 1].set_ylabel("Velocity [rad/s]")
        axes[i, 1].legend()
        axes[i, 1].grid(True, alpha=0.3)

        # Velocity zoomed 10-22s
        mask = time_np > 10
        axes[i, 2].plot(time_np[mask], sim_vel_np[mask, i], c="tab:red", label="Sim", linewidth=0.8)
        axes[i, 2].plot(time_np[mask], real_vel_np[mask, i], c="tab:blue", label="Real", linewidth=0.8)
        axes[i, 2].set_title(f"{joint_order[i]} — Velocity (10-22s)")
        axes[i, 2].set_ylabel("Velocity [rad/s]")
        axes[i, 2].legend()
        axes[i, 2].grid(True, alpha=0.3)

    for ax in axes[1]:
        ax.set_xlabel("Time [s]")

    plt.tight_layout()
    out_path = args_cli.out or str(project_root() / "logs" / "pace" / "bb" / "manual_test.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
    simulation_app.close()
