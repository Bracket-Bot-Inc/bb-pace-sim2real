# BB Hoverboard Motor Configuration
# 36V, 6.5" hub motor, 2.65kg, 350W

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.utils import configclass

from pace_sim2real.utils import PaceDCMotorCfg
from pace_sim2real import PaceSim2realEnvCfg, PaceSim2realSceneCfg, PaceCfg

BB_USD_PATH = "/home/jliu/bb_RL/bb_isaaclab_ws/source/bb_isaaclab_ws/bb_isaaclab_ws/assets/bb.usd"

HOVERBOARD_PACE_ACTUATOR_CFG = PaceDCMotorCfg(
    joint_names_expr=[".*"],
    saturation_effort=6.8,    # peak/stall torque [Nm]
    effort_limit=6.8,         # [Nm]
    velocity_limit=73.0,      # ~700 RPM no-load [rad/s]
    stiffness={".*": 0.7},   # PD P-gain [Nm/rad] - needs tuning
    damping={".*": 4.8},      # PD D-gain [Nm·s/rad] - needs tuning
    encoder_bias=[0.0] * 2,   # 2 wheel joints
    max_delay=10,             # max delay in simulation steps
)


N_JOINTS = 2


@configclass
class BBPaceCfg(PaceCfg):
    """Pace configuration for BB (BracketBot) robot."""
    robot_name: str = "bb"
    data_dir: str = "bb/chirp_data.pt"  # located in data/bb/chirp_data.pt
    # 2 armature + 2 damping + 2 friction + 2 bias + 1 delay = 9 parameters
    bounds_params: torch.Tensor = torch.zeros((N_JOINTS * 4 + 1, 2))
    joint_order: list[str] = ["drive_left", "drive_right"]

    def __post_init__(self):
        n = N_JOINTS
        self.bounds_params[:n, 0] = 1e-5
        self.bounds_params[:n, 1] = 1.0       # armature [kg·m²]
        self.bounds_params[n:2*n, 1] = 7.0    # viscous damping [Nm·s/rad]
        self.bounds_params[2*n:3*n, 1] = 0.5  # Coulomb friction
        self.bounds_params[3*n:4*n, 0] = -0.1
        self.bounds_params[3*n:4*n, 1] = 0.1  # encoder bias [rad]
        self.bounds_params[4*n, 1] = 10.0     # delay [sim steps]


@configclass
class BBPaceSceneCfg(PaceSim2realSceneCfg):
    """Scene configuration for BB robot in Pace Sim2Real environment."""
    robot: ArticulationCfg = ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(usd_path=BB_USD_PATH),
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.05),
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
        self.sim.dt = 0.0025  # 400Hz simulation
        self.decimation = 1   # 400Hz control
