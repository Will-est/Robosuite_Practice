import robosuite as suite
from robosuite.controllers import load_composite_controller_config
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import DummyVecEnv
from robosuite.wrappers import GymWrapper
from gym import Env
from gymnasium import Env, spaces
from stable_baselines3 import SAC

class CustomGymnasiumEnv(Env):
    def __init__(self, robosuite_env):
        super().__init__()
        self.robosuite_env = robosuite_env
        self.action_space = spaces.Box(
            low=robosuite_env.action_spec[0],
            high=robosuite_env.action_spec[1],
            dtype=np.float32
        )
        obs_spec = self.custom_observation(robosuite_env.observation_spec())
        self.observation_space = spaces.Box(
           low=-np.inf,
            high=np.inf,
            shape=obs_spec.shape,
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        obs = self.robosuite_env.reset()
        # Custom observation
        obs = self.custom_observation(obs)
        return obs, {}

    def step(self, action):
        obs, _, done, info = self.robosuite_env.step(action)

        # Custom observation
        np_obs = self.custom_observation(obs)

        success = info.get("success", False)

        # Compute custom reward
        reward = custom_reward(obs, success)

        return np_obs, reward, done, False, info
    
    def custom_observation(self, obs):
            # Extract end-effector position and orientation (quaternion)
            ee_pos = obs["robot0_eef_pos"]
            ee_quat = obs["robot0_eef_quat"]

            # Extract block position and orientation (quaternion)
            box_pos = obs["cube_pos"]
            box_quat = obs["cube_quat"]

            # Extract gripper joint positions
            gripper_qpose = obs["robot0_gripper_qpos"]

            # Extract joint velocities
            joint_velocities = obs["robot0_joint_vel"]

            # Return as a single array
            return np.concatenate([ee_pos, ee_quat, box_pos, box_quat, gripper_qpose, joint_velocities])
            # return np.concatenate([ee_pos, ee_quat, box_pos, box_quat, gripper_qpose])

    
def create_environment(render=False):
    

    # Load the OSC controller configuration for the Panda arm
    controller_config = load_composite_controller_config(controller="test.json")

    

    # Create the environment
    env = suite.make(
        env_name="Lift",  # Block lifting task
        robots="Panda",  # Panda arm
        controller_configs=controller_config,
        has_renderer=render,  # Rendering enabled/disabled
        has_offscreen_renderer=False,
        use_camera_obs=False,
        reward_shaping=True,  # Enable reward shaping
        horizon=200,  # Set time horizon to 500
    )
    gym_env = CustomGymnasiumEnv(env)

    return gym_env

def custom_reward(obs, success):

    # defining signals
    arm_position = obs["robot0_eef_pos"]  # End-effector position
    block_position = obs["cube_pos"]  # Block position   
    joint_velocities = obs["robot0_joint_vel"]

    # Distance between arm and block
    arm_to_block_dist = np.linalg.norm(arm_position - block_position)
    arm_to_block_dist_penalty = arm_to_block_dist * 5
    # Reward for touching the block (small distance between arm and block)
    touch_reward = 10 if arm_to_block_dist < 0.1 else 0.0  # Reward for being close to the block

    # Distance between block and goal
    # Desired end-effector quaternion
    desired_ee_quat = np.array([0.99914491, 0.00659652, 0.03877609, 0.01274182])

    # Compute quaternion deviation penalty
    ee_quat = obs["robot0_eef_quat"]
    quat_deviation_penalty = np.linalg.norm(ee_quat - desired_ee_quat) # Scale penalty

    block_height = block_position[2] - 0.821 if (block_position[2] - 0.821 > 0.001) else 0.0
    block_height_reward = np.min([block_height, 0.93]) * 1000  # Add limit to height

    # Penalize rapid joint movements
    joint_movement_penalty = np.sum(np.abs(joint_velocities)) * 0.1  # Scale penalty

    # Success term (1 if successful, 0 otherwise)
    success_term = 10 if success else 0
    # Print each term in the reward for debugging
    # print(f"Arm to Block Distance reward: {arm_to_block_dist_penalty}")
    # print(f"Touch Reward: {touch_reward}")
    # print(f"Block Height: {block_height}")
    # print(f"Joint Movement Penalty: {joint_movement_penalty}")
    # print(f"Quaternion Deviation Penalty: {quat_deviation_penalty}")
    # print(f"Success Term: {success_term}")
    # # Combine terms into a single reward
    reward = block_height_reward + success_term + touch_reward - arm_to_block_dist_penalty - joint_movement_penalty - quat_deviation_penalty
    return reward


def train_with_ppo(env):

    # Wrap the robosuite environment to make it Gym-compatible
    # gym_env = GymWrapper(env)
    obs = env.reset()
    # Wrap the Gym-compatible environment in a vectorized wrapper
    vec_env = DummyVecEnv([lambda: env])

    # Initialize the PPO model
    model = SAC("MlpPolicy", vec_env, verbose=1)

    # Train the model
    model.learn(total_timesteps=50000)  # Example: 100k timesteps

    # Save the trained model
    model.save("ppo_robosuite_model")

    return model

def evaluate_environment(env):
    # Load the trained model
    model = SAC.load("ppo_robosuite_model.zip")

    obs, _ = env.reset()
    done = False
    step = 0
    while not done:
        # Use the trained model to predict actions
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)

        # Optionally, print reward, step, or other info for evaluation
        print(f"Step: {step}, Reward: {reward}, Info: {info}")
        # # Print quaternion information
        # ee_quat = obs[3:7]  # Extract end-effector quaternion from obs
        # box_quat = obs[10:14]  # Extract block quaternion from obs
        # print(f"End-effector Quaternion: {ee_quat}")
        # print(f"Block Quaternion: {box_quat}")
        # Update step count
        step += 1

if __name__ == "__main__":
    # Training environment (no rendering)
    # train_env = create_environment(render=False)
    # train_with_ppo(train_env)

    # # Evaluation environment (with rendering)
    eval_env = create_environment(render=True)
    evaluate_environment(eval_env)