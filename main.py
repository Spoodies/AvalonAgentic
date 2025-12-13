from AECSetup import CustomGameEnv

env = CustomGameEnv()
env.reset()

for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()
    
    if termination or truncation:
        action = None
    else:
        action = 1 # Your model predicts here
        
    env.step(action)