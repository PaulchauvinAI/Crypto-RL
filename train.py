import os
import numpy as np
import fire
import wandb
from wandb.integration.sb3 import WandbCallback
from agents.elegantrl_models import DRLAgent as DRLAgent_erl
from agents.rllib_models import DRLAgent as DRLAgent_rllib
from agents.stablebaselines3_models import DRLAgent as DRLAgent_sb3
from agents.stablebaselines3_models import TensorboardCallback

# from agents.stablebaselines3_models import DRLEnsembleAgent as DRLEnsembleAgent_sb3
from meta.data_processor import DataProcessor
from meta.env_crypto_trading.env_multiple_crypto import CryptoEnv

# install talib https://gist.github.com/brunocapelao/ed1b4f566fccf630e1fb749e5992e964
# DRL models from ElegantRL: https://github.com/AI4Finance-Foundation/ElegantRL
# set up tensorboard:  tensorboard dev upload --logdir ./tensorboard_log


def train(
    start_date,
    end_date,
    ticker_list,
    data_source,
    time_interval,
    technical_indicator_list,
    drl_lib,
    model_save_freq,
    env,
    model_name,
    reward_type,
    use_wandb,
    if_vix=True,
    **kwargs,
):

    # process data using unified data processor
    DP = DataProcessor(data_source, start_date, end_date, time_interval, **kwargs)

    price_array, tech_array, turbulence_array = DP.run(
        ticker_list, technical_indicator_list, if_vix, cache=True
    )

    data_config = {
        "price_array": price_array,
        "tech_array": tech_array,
        "turbulence_array": turbulence_array,
        "reward_type": reward_type,
    }

    print(
        "\n number of crypto that we are trading: {} and number of timeframes to evaluate the model: {} \n".format(
            price_array.shape[1], price_array.shape[0]
        )
    )
    price_ratio = price_array[-1] / price_array[0]
    print(
        "The best buy and hold strategy has a return of : ",
        max(price_ratio),
        "on this period \n",
    )
    print(
        "The mean buy and hold strategy with a same proportion of all the crypto is  : ",
        np.mean(price_ratio),
        "on this period \n",
    )

    # build environment using processed data
    env_instance = env(config=data_config)

    # read parameters and load agents
    current_working_dir = kwargs.get("current_working_dir", "./")

    if drl_lib == "elegantrl":
        break_step = kwargs.get("break_step", 1e6)
        agent = DRLAgent_erl(env=env_instance)
        ERL_PARAMS = {
            "learning_rate": 2**-15,
            "batch_size": 2**11,
            "gamma": 0.99,
            "seed": 312,
            "net_dimension": 2**9,
            "target_step": 5000,
            "eval_gap": 30,
            "eval_times": 1,
        }
        model = agent.get_model(model_name, model_kwargs=ERL_PARAMS)

        agent.train_model(
            model=model, cwd=current_working_dir, total_timesteps=break_step
        )

    elif drl_lib == "rllib":
        # Only works with gpu instance? Tensorflow is needed
        total_episodes = kwargs.get("total_episodes", 100)
        # rllib_params = kwargs.get('rllib_params')
        agent_rllib = DRLAgent_rllib(
            env=env,
            price_array=price_array,
            tech_array=tech_array,
            turbulence_array=turbulence_array,
            reward_type=reward_type,
        )
        RLlib_PARAMS = {"lr": 5e-5, "train_batch_size": 500, "gamma": 0.99}

        model, model_config = agent_rllib.get_model(model_name)

        model_config["lr"] = RLlib_PARAMS["lr"]
        model_config["train_batch_size"] = RLlib_PARAMS["train_batch_size"]
        model_config["gamma"] = RLlib_PARAMS["gamma"]

        trained_model = agent_rllib.train_model(
            model=model,
            model_name=model_name,
            model_config=model_config,
            total_episodes=total_episodes,
        )
        trained_model.save(current_working_dir)

    elif drl_lib == "stable_baselines3":
        if use_wandb:
            callback = WandbCallback(
                model_save_path=f"models/{current_working_dir}",
                verbose=2,
                model_save_freq=model_save_freq,
            )
        else:
            callback = TensorboardCallback()

        total_timesteps = kwargs.get("total_timesteps", 1e7)
        # agent_params = kwargs.get('agent_params')
        agent = DRLAgent_sb3(env=env_instance)
        # agent = DRLEnsembleAgent_sb3(env = env_instance)
        model = agent.get_model(model_name)
        trained_model = agent.train_model(
            model=model,
            tb_log_name=model_name,
            callback=callback,
            total_timesteps=total_timesteps,
        )

        print("Training finished!")
        trained_model.save(current_working_dir)
        print("Trained model saved in " + str(current_working_dir))
    else:
        raise ValueError("DRL library input is NOT supported. Please check.")


##### Part suceptible to change

# TRAIN_START_DATE = '2022-01-04'
# TRAIN_END_DATE = '2022-01-10'

TRAIN_START_DATE = "2018-09-04"
TRAIN_END_DATE = "2021-04-10"

TICKER_LIST = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT"]
INDICATORS = [
    "macd",
    "rsi",
    "cci",
    "dx",
]  # TODO fix the cache problem when we want to try other indicators.


def main(
    model_name="ppo",
    time_interval="30m",
    reward_type="normal",
    drl_lib="stable_baselines3",
    model_save_freq=1e5,
    use_wandb=True,
):
    """
    model name must be in {"a2c": A2C, "ddpg": DDPG, "td3": TD3, "sac": SAC, "ppo": PPO}
    time interval must be in ['1m', '5m', '15m', '30m', '60m', '120m', '1d', '1w', '1M']
    drllib is eather stable_baselines3, elegantrl, or rllib
    """
    # 1e5 => 10min to save models.

    if use_wandb:
        wandb.init(
            project="FinRL",
            # config=config,
            sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
            save_code=True,  # optional
        )

    current_working_dir = f"./test_{model_name}_{time_interval}"

    train(
        start_date=TRAIN_START_DATE,
        end_date=TRAIN_END_DATE,
        ticker_list=TICKER_LIST,
        data_source="binance",
        time_interval=time_interval,
        technical_indicator_list=INDICATORS,
        drl_lib=drl_lib,
        model_save_freq=model_save_freq,
        env=CryptoEnv,
        model_name=model_name,
        current_working_dir=current_working_dir,
        break_step=5e4,  # only for elegantrl
        reward_type=reward_type,
        use_wandb=use_wandb,
        if_vix=False,
    )

    ## if users want to use rllib, or stable-baselines3, users can remove the following comments

    # # demo for rllib
    # import ray
    # ray.shutdown()  # always shutdown previous session if any
    # train(
    #     start_date=TRAIN_START_DATE,
    #     end_date=TRAIN_END_DATE,
    #     ticker_list=DOW_30_TICKER,
    #     data_source="yahoofinance",
    #     time_interval="1D",
    #     technical_indicator_list=INDICATORS,
    #     drl_lib="rllib",
    #     env=env,
    #     model_name="ppo",
    #     cwd="./test_ppo",
    #     rllib_params=RLlib_PARAMS,
    #     total_episodes=30,


if __name__ == "__main__":
    fire.Fire(main)
