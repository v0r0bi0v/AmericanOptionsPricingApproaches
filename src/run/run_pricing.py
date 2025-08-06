import numpy as np
from tqdm.auto import tqdm
import os
import pickle
from joblib import Parallel, delayed

from src.samplers.samplers import NAME_TO_SAMPLER
from src.pricers.pricers import NAME_TO_PRICER
from src.run.samplers_params import PARAMS as SAMPLERS_PARAMS
from src.run.pricers_params import PARAMS as PRICERS_PARAMS

SAMPLER_NAME = "GeometricBrownianMotionPutSampler"
assert SAMPLER_NAME in ["GeometricBrownianMotionPutSampler", "WienerRainbowPutOptionSampler"]

PRICER_NAME = "LSPIPricer"
assert PRICER_NAME in ["BinomialTreePricer", "LSPIPricer", "AmericanMonteCarloPricer"]


CNT_REPEATS = 8
CNT_TRAJECTORIES = np.linspace(1_000, 100_000, 5, dtype=int)

SAVES_DIR = "0"

save_dir = os.path.join(os.path.dirname(__file__), "..", "..", "saves", SAVES_DIR)
os.makedirs(save_dir, exist_ok=True)


def run_pricing_1time(
    cnt_trajectories: int = 1000,
    test: bool = False,
    quiet: bool = False
):
    sampler_type = NAME_TO_SAMPLER[SAMPLER_NAME]
    pricer_type = NAME_TO_PRICER[PRICER_NAME]

    sampler_params = SAMPLERS_PARAMS[SAMPLER_NAME].copy()
    sampler_params["cnt_trajectories"] = cnt_trajectories

    sampler = sampler_type(**sampler_params)
    pricer = pricer_type(**PRICERS_PARAMS[PRICER_NAME], sampler=sampler)

    train_price_history = pricer.price(test=False, quiet=quiet)
    test_price_history = None
    if test:
        test_price_history = pricer.price(test=True, quiet=quiet)

    return train_price_history[0], test_price_history[0]


def process_repeats(args):
    cnt_trajectories, test, quiet, repeat_idx = args
    train_price, test_price = run_pricing_1time(cnt_trajectories, test, quiet)
    return repeat_idx, train_price, test_price


def run_pricing_multiple_times(test=True, quiet=False, num_cores=None):
    if num_cores is None:
        num_cores = -

    train_prices = np.empty((len(CNT_TRAJECTORIES), CNT_REPEATS))
    test_prices = np.empty((len(CNT_TRAJECTORIES), CNT_REPEATS))
    
    for traj_idx, cnt_trajectories in tqdm(list(enumerate(CNT_TRAJECTORIES)), desc="Pricing with different number of trajectories"):
        print(f"Pricing {cnt_trajectories} / {CNT_TRAJECTORIES[-1]} ({traj_idx + 1} / {len(CNT_TRAJECTORIES)})")

        args_list = [
            (cnt_trajectories, test, quiet, repeat_idx) 
            for repeat_idx in range(CNT_REPEATS)
        ]
        
        results = Parallel(n_jobs=num_cores)(delayed(process_repeats)(args) for args in args_list) \
            if len(args_list) > 1 \
                else [process_repeats(args_list[0])]
        for repeat_idx, train_price, test_price in results:
            train_prices[traj_idx, repeat_idx] = train_price
            test_prices[traj_idx, repeat_idx] = test_price
        
    
    save_path = os.path.join(save_dir, f"{SAMPLER_NAME}_{PRICER_NAME}_{CNT_REPEATS}x{len(CNT_TRAJECTORIES)}.pkl")
    with open(save_path, "wb") as f:
        pickle.dump((train_prices, test_prices, CNT_TRAJECTORIES), f)

    return train_prices, test_prices


if __name__ == "__main__":
    run_pricing_multiple_times(test=True, quiet=False)
