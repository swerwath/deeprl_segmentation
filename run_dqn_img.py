import argparse
import os.path as osp
import random
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
from env import Environment
import dqn
from dqn_utils import *
from u_net import build_unet
from data_generator import generator_fn



def img_segment_learn(env,
                session,
                num_timesteps):
    # This is just a rough estimate
    num_iterations = float(num_timesteps) / 4.0

    lr_multiplier = 1.0
    lr_schedule = PiecewiseSchedule([
                                         (0,                   1e-4 * lr_multiplier),
                                         (num_iterations / 10, 1e-4 * lr_multiplier),
                                         (num_iterations / 2,  5e-5 * lr_multiplier),
                                    ],
                                    outside_value=5e-5 * lr_multiplier)
    optimizer = dqn.OptimizerSpec(
        constructor=tf.train.AdamOptimizer,
        kwargs=dict(epsilon=1e-4),
        lr_schedule=lr_schedule
    )

    def stopping_criterion(env, t):
        # notice that here t is the number of steps of the wrapped env,
        # which is different from the number of steps in the underlying env
        return get_wrapper_by_name(env, "Monitor").get_total_steps() >= num_timesteps

    exploration_schedule = PiecewiseSchedule(
        [
            (0, 1.0),
            (1e6, 0.1),
            (num_iterations / 2, 0.01),
        ], outside_value=0.01
    )

    return dqn.learn(
        env=env,
        q_func=build_unet,
        optimizer_spec=optimizer,
        session=session,
        exploration=exploration_schedule,
        replay_buffer_size=50000,
        batch_size=32,
        gamma=0.99,
        learning_starts=50000,
        learning_freq=4,
        target_update_freq=10000,
        grad_norm_clipping=10,
        double_q=False
    )

def get_available_gpus():
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    return [x.physical_device_desc for x in local_device_protos if x.device_type == 'GPU']

def get_session():
    tf.reset_default_graph()
    tf_config = tf.ConfigProto(
        inter_op_parallelism_threads=1,
        intra_op_parallelism_threads=1)
    session = tf.Session(config=tf_config)
    print("AVAILABLE GPUS: ", get_available_gpus())
    return session


def main():
    # Run training
    env = Environment(generator_fn(), img_shape=(256,256,3))
    #test_env = Environment(test_generator_fn)
    session = get_session()
    alg = img_segment_learn(env, session, num_timesteps=2e8)
    #test_results, test_rewards = alg.test(test_env, num_test_samples=1000)

    

if __name__ == "__main__":
    main()
