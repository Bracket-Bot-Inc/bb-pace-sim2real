# BB Hoverboard Motor Configuration
# 36V, 6.5" hub motor, 2.65kg, 350W

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.utils import configclass

from pace_sim2real.utils import PaceDCMotorCfg
from pace_sim2real import PaceSim2realEnvCfg, PaceTorqueSim2realEnvCfg, PaceSim2realSceneCfg, PaceCfg

BB_USD_PATH = "/home/jliu/bb_RL/bb_isaaclab_ws/source/bb_isaaclab_ws/bb_isaaclab_ws/assets/bb.usd"

HOVERBOARD_PACE_ACTUATOR_CFG = PaceDCMotorCfg(
    joint_names_expr=[".*"],
    saturation_effort=3.0,    # match USD maxForce [Nm]
    effort_limit=3.0,         # [Nm]
    velocity_limit=73.0,      # ~700 RPM no-load [rad/s]
    stiffness={".*": 0.535},  # Nm/rad (vel_gain * pos_gain / 2π)
    damping={".*": 0.111},    # Nm·s/rad (vel_gain / 2π)
    encoder_bias=[0.0] * 2,   # 2 wheel joints
    max_delay=10,             # max delay in simulation steps
)


N_JOINTS = 2


@configclass
class BBPaceCfg(PaceCfg):
    """Pace configuration for BB (BracketBot) robot."""
    robot_name: str = "bb"
    data_dir: str = "bracketbot/bb_chirp_data.pt"
    # 2 armature + 2 damping + 2 friction + 2 bias + 1 delay = 9 parameters
    bounds_params: torch.Tensor = torch.zeros((N_JOINTS * 4 + 1, 2))
    joint_order: list[str] = ["drive_left", "drive_right"]

    def __post_init__(self):
        n = N_JOINTS
        self.bounds_params[:n, 0] = 1e-5
        self.bounds_params[:n, 1] = 1.0       # armature [kg·m²]
        self.bounds_params[n:2*n, 1] = 0.01   # viscous damping [Nm·s/rad]
        self.bounds_params[2*n:3*n, 1] = 0.4  # Coulomb friction
        self.bounds_params[3*n:4*n, 0] = -0.05
        self.bounds_params[3*n:4*n, 1] = 0.05  # encoder bias [rad]
        self.bounds_params[4*n, 1] = 10.0     # delay [sim steps]


@configclass
class BBPaceSceneCfg(PaceSim2realSceneCfg):
    """Scene configuration for BB robot in Pace Sim2Real environment."""
    robot: ArticulationCfg = ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path=BB_USD_PATH,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
            ),
        ),
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.3),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        actuators={"wheels": HOVERBOARD_PACE_ACTUATOR_CFG},
    )


@configclass
class BBPaceEnvCfg(PaceSim2realEnvCfg):

    scene: BBPaceSceneCfg = BBPaceSceneCfg()
    sim2real: PaceCfg = BBPaceCfg()

    def __post_init__(self):
        super().__post_init__()
        self.sim.dt = 0.005  # 200Hz simulation
        self.decimation = 1  # 200Hz control


# Torque-based sysid: direct torque actions, no PD controller
HOVERBOARD_TORQUE_ACTUATOR_CFG = PaceDCMotorCfg(
    joint_names_expr=[".*"],
    saturation_effort=3.0,
    effort_limit=3.0,
    velocity_limit=73.0,
    stiffness={".*": 0.0},    # no PD position control
    damping={".*": 0.0},      # no PD velocity control
    encoder_bias=[0.0] * 2,
    max_delay=10,
)


@configclass
class BBPaceTorqueCfg(PaceCfg):
    """Pace configuration for BB torque-based sysid."""
    robot_name: str = "bb"
    data_dir: str = "bracketbot/bb_torque_chirp_data.pt"
    bounds_params: torch.Tensor = torch.zeros((N_JOINTS * 4 + 1, 2))
    joint_order: list[str] = ["drive_left", "drive_right"]

    def __post_init__(self):
        n = N_JOINTS
        self.bounds_params[:n, 0] = 1e-5
        self.bounds_params[:n, 1] = 1.0       # armature [kg·m²]
        self.bounds_params[n:2*n, 1] = 0.01   # viscous damping [Nm·s/rad]
        self.bounds_params[2*n:3*n, 1] = 0.4  # Coulomb friction
        self.bounds_params[3*n:4*n, 0] = -0.05
        self.bounds_params[3*n:4*n, 1] = 0.05  # encoder bias [rad]
        self.bounds_params[4*n, 1] = 10.0     # delay [sim steps]


@configclass
class BBPaceTorqueSceneCfg(PaceSim2realSceneCfg):
    """Scene configuration for BB torque-based sysid."""
    robot: ArticulationCfg = ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path=BB_USD_PATH,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
            ),
        ),
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.3),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        actuators={"wheels": HOVERBOARD_TORQUE_ACTUATOR_CFG},
    )


@configclass
class BBPaceTorqueEnvCfg(PaceTorqueSim2realEnvCfg):

    scene: BBPaceTorqueSceneCfg = BBPaceTorqueSceneCfg()
    sim2real: PaceCfg = BBPaceTorqueCfg()

    def __post_init__(self):
        super().__post_init__()
        self.sim.dt = 0.005  # 200Hz simulation
        self.decimation = 1  # 200Hz control
