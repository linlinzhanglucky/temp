import gymnasium as gym
import evogym.envs
from evogym import sample_robot
import numpy as np
import imageio



if __name__ == '__main__':

    body, connections = sample_robot((5,5))
    body = np.random.randint(1, 5, size=(5, 5))
    body[2:5, 2] = 0
    
    
    env = gym.make('Walker-v0', body=body, render_mode='rgb_array')
    env.reset()
    frames = []
    while True:
        
        action = env.action_space.sample()
        ob, reward, terminated, truncated, info = env.step(action)
        frame = env.render()
        frames.append(frame)

        if terminated or truncated:
            env.reset()
        if terminated or truncated:
            break

    env.close()
    print(len(frames))
    imageio.mimsave('walker.gif', frames, fps=30)