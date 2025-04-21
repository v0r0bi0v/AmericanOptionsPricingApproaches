from src.samplers.samplers import NAME_TO_SAMPLER
from src.pricers.pricers import NAME_TO_PRICER
from src.run.samplers_params import PARAMS as SAMPLERS_PARAMS
from src.run.pricers_params import PARAMS as PRICERS_PARAMS

sampler_name = "GeometricBrownianMotionPutSampler"
pricer_name = "AmericanMonteCarloPricer"


def run_pricing():
    sampler_type = NAME_TO_SAMPLER[sampler_name]
    pricer_type = NAME_TO_PRICER[pricer_name]

    sampler = sampler_type(**SAMPLERS_PARAMS[sampler_name])
    pricer = pricer_type(**PRICERS_PARAMS[pricer_name], sampler=sampler)

    