from samplers.geometric_brownian_motion_put_sampler import GeometricBrownianMotionPutSampler
from samplers.wiener_rainbow_put_option_sampler import WienerRainbowPutOptionSampler


NAME_TO_SAMPLER = {
    "WienerRainbowPutOptionSampler": WienerRainbowPutOptionSampler,
    "GeometricBrownianMotionPutSampler": GeometricBrownianMotionPutSampler
}