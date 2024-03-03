import pdb
import os
import numpy as np
from copy import deepcopy
from tqdm import tqdm
from typing import Dict, Tuple, List
import argparse
import multiprocessing
import gymnasium as gym
import matplotlib.pyplot as plt

from pydrake.gym import DrakeGymEnv

# Even if all of these aren't explicitly used, they may be needed for python to
# recognize certain derived classes
from pydrake.systems.all import (
    Diagram,
    EventStatus,
    Context,
    Simulator,
    InputPort,
    OutputPort,
    DiagramBuilder,
    InputPortIndex,
    OutputPortIndex,
    ConstantVectorSource,
    ZeroOrderHold
)

from pydairlib.perceptive_locomotion.systems.alip_lqr_rl import (
    AlipFootstepLQROptions,
    AlipFootstepLQR,
    calc_collision_cost_grid
)

from pydairlib.perceptive_locomotion.systems. \
cassie_footstep_controller_gym_environment import (
    CassieFootstepControllerEnvironmentOptions,
    CassieFootstepControllerEnvironment,
    InitialConditionsServer
)
# Can use DrawAndSaveDiagramGraph for debugging if necessary
from pydairlib.systems.system_utils import DrawAndSaveDiagramGraph

perception_learning_base_folder = "bindings/pydairlib/perceptive_locomotion/perception_learning"
sim_params = CassieFootstepControllerEnvironmentOptions()

def build_diagram(sim_params: CassieFootstepControllerEnvironmentOptions) \
        -> Tuple[CassieFootstepControllerEnvironment, AlipFootstepLQR, Diagram]:
    builder = DiagramBuilder()
    sim_env = CassieFootstepControllerEnvironment(sim_params)
    sim_env.set_name("CassieFootstepControllerEnvironment")
    controller = sim_env.AddToBuilderWithFootstepController(builder, AlipFootstepLQR) # AlipFootstep
    controller.set_name("AlipFootstepLQR")
    ####
    observation = sim_env.AddToBuilderObservations(builder)
    reward = sim_env.AddToBuilderRewards(builder)
    builder.ExportInput(controller.get_input_port_by_name("action_ue"), "actions")
    ####
    footstep_zoh = ZeroOrderHold(1.0 / 30.0, 3)
    builder.AddSystem(footstep_zoh)
    builder.Connect(
        controller.get_output_port_by_name('footstep_command'),
        footstep_zoh.get_input_port()
    )
    builder.Connect(
        footstep_zoh.get_output_port(),
        sim_env.get_input_port_by_name('footstep_command')
    )

    diagram = builder.Build()

    DrawAndSaveDiagramGraph(diagram, '../ALIP_RL')
    return sim_env, controller, diagram


def reset_handler(simulator, diagram_context, seed): # context = self.simulator.get_mutable_context() ?
    
    np.random.seed(seed)

    # Get controller from context or simulator
    diagram = simulator.get_system()
    context = diagram.CreateDefaultContext()
    controller = diagram.GetSubsystemByName("AlipFootstepLQR")
    sim_env = diagram.GetSubsystemByName("CassieFootstepControllerEnvironment")
    controller_context = controller.GetMyMutableContextFromRoot(context)
    sim_context = sim_env.GetMyMutableContextFromRoot(context)

    # Generate datapoint
    # datapoint values set before we get it here:
    # ['stance', 'desired_velocity', 'phase', 'initial_swing_foot_pos', 'q', 'v']
    ic_generator = InitialConditionsServer(
        fname=os.path.join(
            perception_learning_base_folder,
            'tmp/initial_conditions_2.npz'
        )
    )
    v_des_theta = np.pi / 6
    v_des_norm = 1.0
    datapoint = ic_generator.random()
    v_theta = np.random.uniform(-v_des_theta, v_des_theta)
    v_norm = np.random.uniform(0.2, v_des_norm)
    datapoint['desired_velocity'] = np.array([v_norm * np.cos(v_theta), v_norm * np.sin(v_theta)]).flatten()

    # timing aliases
    t_ss = controller.params.single_stance_duration
    t_ds = controller.params.double_stance_duration
    t_s2s = t_ss + t_ds

    datapoint['stance'] = 0 if datapoint['stance'] == 'left' else 1

    #  First, align the timing with what's given by the initial condition
    t_init = datapoint['stance'] * t_s2s + datapoint['phase'] + t_ds
    context.SetTime(t_init)

    # set the context state with the initial conditions from the datapoint
    sim_env.initialize_state(context, diagram, datapoint['q'], datapoint['v'])
    sim_env.controller.SetSwingFootPositionAtLiftoff(
        context,
        datapoint['initial_swing_foot_pos']
    )
    controller.get_input_port_by_name("desired_velocity").FixValue(
        context=controller_context,
        value=datapoint['desired_velocity']
    )
    #simulator.reset_context(context)
    #simulator.AdvanceTo()


def simulate_init(sim_params):
    sim_env, controller, diagram = build_diagram(sim_params)
    simulator = Simulator(diagram)
    
    #contexts = initialize_sim(sim_env, controller, diagram, datapoint)

    #simulator.reset_context(contexts['root'])
    simulator.Initialize()

    def monitor(context):
        time_limit = 10
        plant = sim_env.cassie_sim.get_plant()
        plant_context = plant.GetMyContextFromRoot(context)
        
        # if center of mas is 20cm 
        left_toe_pos = plant.CalcPointsPositions(
            plant_context, plant.GetBodyByName("toe_left").body_frame(),
            np.array([0.02115, 0.056, 0.]), plant.world_frame()
        )
        right_toe_pos = plant.CalcPointsPositions(
            plant_context, plant.GetBodyByName("toe_right").body_frame(),
            np.array([0.02115, 0.056, 0.]), plant.world_frame()
        )
        com = plant.CalcCenterOfMassPositionInWorld(plant_context)
        z1 = com[2] - left_toe_pos[2]
        z2 = com[2] - right_toe_pos[2]
        
        if context.get_time() > time_limit:
            return EventStatus.ReachedTermination(diagram, "time limit")

        if z1 < 0.2:
            return EventStatus.ReachedTermination(diagram, "left toe exceeded")

        if z2 < 0.2:
            return EventStatus.ReachedTermination(diagram, "right toe exceeded")

        return EventStatus.Succeeded()

    simulator.set_monitor(monitor)
    return controller, simulator


def DrakeCassieEnv(sim_params: CassieFootstepControllerEnvironmentOptions):
    
    #sim_params = CassieFootstepControllerEnvironmentOptions()
    #sim_params.terrain = os.path.join(
    #    perception_learning_base_folder, 'params/flat.yaml'
    #)
    #sim_params.visualize = True

    #sim_env, controller, diagram = build_diagram(sim_params)
    #simulator = Simulator(diagram)
    #context = diagram.CreateDefaultContext()
    #simulator.set_monitor(monitor(sim_env, context, diagram))
    controller, simulator = simulate_init(sim_params)


    # Define Action space.
    la = np.array([-1., -1., -1.])
    ha = np.array([1., 1., 1.])
    action_space = gym.spaces.Box(low=np.asarray(la, dtype="float32"),
                                  high=np.asarray(ha, dtype="float32"),
                                  dtype=np.float32)

    # Define observation space. <- Full state + observation matrix
    lo = np.array([-np.inf, -np.inf, -np.inf, -np.inf])
    ho = np.array([np.inf, np.inf, np.inf, np.inf])
    observation_space = gym.spaces.Box(low=np.asarray(lo, dtype="float64"),
                                  high=np.asarray(ho, dtype="float64"),
                                  dtype=np.float64)

    #t_ss = controller.params.single_stance_duration
    #t_ds = controller.params.double_stance_duration
    #t_s2s = t_ss + t_ds
    #t_eps = 1e-2
    #time_step = t_s2s - t_eps - datapoint['phase']
    time_step = 10.0
    
    env = DrakeGymEnv(
        simulator=simulator,
        time_step=time_step,
        action_space=action_space,
        observation_space=observation_space,
        reward="reward",
        action_port_id="actions",
        observation_port_id="observations",
        reset_handler = reset_handler,
        #render_mode = 'human'
        )

    return env