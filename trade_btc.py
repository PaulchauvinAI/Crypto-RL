from agents.elegantrl_models import DRLAgent as DRLAgent_erl
from agents.rllib_models import DRLAgent as DRLAgent_rllib
from agents.stablebaselines3_models import DRLAgent as DRLAgent_sb3
from meta.data_processor import DataProcessor
import os
#install talib https://gist.github.com/brunocapelao/ed1b4f566fccf630e1fb749e5992e964
# DRL models from ElegantRL: https://github.com/AI4Finance-Foundation/ElegantRL
import torch
from elegantrl.agents import AgentDDPG
from elegantrl.agents import AgentPPO
from elegantrl.agents import AgentSAC
from elegantrl.agents import AgentTD3
from elegantrl.train.config import Arguments
from elegantrl.train.run import init_agent
from elegantrl.train.run import train_and_evaluate
import numpy as np
import math
import fire
from meta.env_crypto_trading.env_multiple_crypto import CryptoEnv
import gym
from wandb.integration.sb3 import WandbCallback
#set up tensorboard:  tensorboard dev upload --logdir ./tensorboard_log
import wandb
run = wandb.init(
    project="FinRL",
    #config=config,
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    save_code=True,  # optional
)



"""
In principle, all information can be used in the "state". In practice, you can do some correlation analysis and put uncorrelated features into a state. Alternatively, you can input all raw features, and then employ an LSTM layer to encode them into a latent state.

"""

# from elegantrl.agents import AgentA2C

MODELS = {"ddpg": AgentDDPG, "td3": AgentTD3, "sac": AgentSAC, "ppo": AgentPPO}
OFF_POLICY_MODELS = ["ddpg", "td3", "sac"]
ON_POLICY_MODELS = ["ppo"]
"""MODEL_KWARGS = {x: config.__dict__[f"{x.upper()}_PARAMS"] for x in MODELS.keys()}
NOISE = {
    "normal": NormalActionNoise,
    "ornstein_uhlenbeck": OrnsteinUhlenbeckActionNoise,
}"""



def train(start_date, end_date, ticker_list, data_source, time_interval, 
          technical_indicator_list, drl_lib, env, model_name, if_vix=True,
          **kwargs):
    
    #process data using unified data processor
    DP = DataProcessor(data_source, start_date, end_date, time_interval, **kwargs)
    price_array, tech_array, turbulence_array = DP.run(ticker_list,
                                                        technical_indicator_list, 
                                                        if_vix, cache=True)

    data_config = {'price_array': price_array,
                   'tech_array': tech_array,
                   'turbulence_array': turbulence_array}

    #build environment using processed data
    env_instance = env(config=data_config)

    #read parameters and load agents
    current_working_dir = kwargs.get('current_working_dir','./')

    if drl_lib == 'elegantrl':
        break_step = kwargs.get('break_step', 1e6)
        erl_params = kwargs.get('erl_params')
        agent = DRLAgent_erl(env = env,
                             price_array = price_array,
                             tech_array=tech_array,
                             turbulence_array=turbulence_array)
        #breakpoint()
        
        model = agent.get_model(model_name, model_kwargs = erl_params)
        
        trained_model = agent.train_model(model=model, 
                                          cwd=current_working_dir,
                                          total_timesteps=break_step)
        
      
    elif drl_lib == 'rllib':
        total_episodes = kwargs.get('total_episodes', 100)
        rllib_params = kwargs.get('rllib_params')

        agent_rllib = DRLAgent_rllib(env = env,
                       price_array=price_array,
                       tech_array=tech_array,
                       turbulence_array=turbulence_array)

        model,model_config = agent_rllib.get_model(model_name)

        model_config['lr'] = rllib_params['lr']
        model_config['train_batch_size'] = rllib_params['train_batch_size']
        model_config['gamma'] = rllib_params['gamma']

        trained_model = agent_rllib.train_model(model=model, 
                                          model_name=model_name,
                                          model_config=model_config,
                                          total_episodes=total_episodes)
        trained_model.save(current_working_dir)
        
            
    elif drl_lib == 'stable_baselines3':
        total_timesteps = kwargs.get('total_timesteps',1e7)
        agent_params = kwargs.get('agent_params')
        agent = DRLAgent_sb3(env = env_instance)

        model = agent.get_model(model_name, model_kwargs = agent_params)
        trained_model = agent.train_model(model=model, 
                                tb_log_name=model_name,
                                callback=WandbCallback(model_save_path=f"models/{current_working_dir}",verbose=2, model_save_freq=100000),
                                total_timesteps=total_timesteps)
        print('Training finished!')
        trained_model.save(current_working_dir)
        print('Trained model saved in ' + str(current_working_dir))
    else:
        raise ValueError('DRL library input is NOT supported. Please check.') 


TICKER_LIST = ['BTCUSDT','ETHUSDT','BNBUSDT','XRPUSDT']
env = CryptoEnv


INDICATORS = ['macd', 'rsi', 'cci', 'dx'] #self-defined technical indicator list is NOT supported yet

ERL_PARAMS = {"learning_rate": 2**-15,"batch_size": 2**11,
                "gamma": 0.99, "seed":312,"net_dimension": 2**9, 
                "target_step": 5000, "eval_gap": 30, "eval_times": 1}



##### Part suceptible to change

TRAIN_START_DATE = '2022-01-04' #BTC
TRAIN_END_DATE = '2022-03-21'  #


#TRAIN_START_DATE = '2022-01-18'
#TRAIN_END_DATE = '2022-03-26' 






# models for sb3: {"a2c": A2C, "ddpg": DDPG, "td3": TD3, "sac": SAC, "ppo": PPO}
# supported time interval: '1m', '5m', '15m', '30m', '60m', '120m', '1d', '1w', '1M'

def main(model_name="ppo",time_interval="1m"):
    current_working_dir='./test_{}_{}'.format(model_name, time_interval)

    train(start_date=TRAIN_START_DATE, 
        end_date=TRAIN_END_DATE,
        ticker_list=TICKER_LIST, 
        data_source='binance',
        time_interval=time_interval, 
        technical_indicator_list=INDICATORS,
        drl_lib='stable_baselines3',#'elegantrl', #, 
        env=env, 
        model_name=model_name, 
        current_working_dir=current_working_dir,
        erl_params=ERL_PARAMS,
        break_step=5e4,
        if_vix=False
        )


if __name__=="__main__":
    fire.Fire(main)