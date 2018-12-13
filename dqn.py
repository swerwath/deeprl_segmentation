import uuid
import time
import pickle
import sys
import itertools
import numpy as np
import random
import tensorflow                as tf
import tensorflow.contrib.layers as layers
from collections import namedtuple
from dqn_utils import *
import json
import os
import time
import logz

from convex_hull import ConvexHullPolicy

OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs", "lr_schedule"])
img_size = 256
window_size = 32

class QLearner(object):

    def __init__(
        self,
        env,
        q_func,
        optimizer_spec,
        session,
        exploration=LinearSchedule(1000000, 0.1),
        total_time_steps=2000000,
        replay_buffer_size=1000,
        batch_size=32,
        gamma=0.99,
        learning_starts=500,
        learning_freq=4,
        target_update_freq=10000,
        grad_norm_clipping=10,
        pixel_limit=50,
        rew_file=None,
        double_q=False,
        progress_dir=None):
        """Run Deep Q-learning algorithm.

        You can specify your own convnet using q_func.

        All schedules are w.r.t. total number of steps taken in the environment.

        Parameters
        ----------
        env: Image_Env
            Environment to train on.
        q_func: function
            Model to use for computing the q functions. It should accept the
            following named arguments:
                img_in: tf.Tensor
                    tensorflow tensor representing the input image
                scope: str
                    scope in which all the model related variables
                    should be created
                reuse: bool
                    whether previously created variables should be reused.
            Returns two tensors:
                q_class: (3)
                q_map: (256, 256)
        optimizer_spec: OptimizerSpec
            Specifying the constructor and kwargs, as well as learning rate schedule
            for the optimizer
        session: tf.Session
            tensorflow session to use.
        exploration: rl_algs.deepq.utils.schedules.Schedule
            schedule for probability of chosing random action.
        total_time_steps: int
            Total time steps we plan to run the RL algorithm for; replaces the stopping criterion
        replay_buffer_size: int
            How many memories to store in the replay buffer.
        batch_size: int
            How many transitions to sample each time experience is replayed.
        gamma: float
            Discount Factor
        learning_starts: int
            After how many environment steps to start replaying experiences
        learning_freq: int
            How many steps of environment to take between every experience replay
        target_update_freq: int
            How many experience replay rounds (not steps!) to perform between
            each update to the target Q network
        grad_norm_clipping: float or None
            If not None gradients' norms are clipped to this value.
        pixel_limit: int
            Number of pixels we limit the drawing step to
        double_q: bool
            If True, then use double Q-learning to compute target values. Otherwise, use vanilla DQN.
            https://papers.nips.cc/paper/3964-double-q-learning.pdf
        progress_dir: str
            Place to store logged image+masks for reference (helps if you have to terminate early)
        """
        self.target_update_freq = target_update_freq
        self.optimizer_spec = optimizer_spec
        self.batch_size = batch_size
        self.learning_freq = learning_freq
        self.learning_starts = learning_starts
        self.total_time_steps=total_time_steps
        self.env = env
        self.session = session
        self.exploration = exploration
        self.rew_file = str(uuid.uuid4()) + '.pkl' if rew_file is None else rew_file
        self.pixel_limit = pixel_limit
        self.progress_dir = progress_dir

        self.hull_policy = ConvexHullPolicy(img_size)

        ###############
        # BUILD MODEL #
        ###############

        
        input_shape = (window_size, window_size, 6)
        action_dim = 2 + window_size * window_size
        
        # set up placeholders
        # placeholder for current observation (or state)
        self.obs_t_ph              = tf.placeholder(tf.uint8, [None] + list(input_shape))
        # placeholder for current action
        self.act_t_ph              = tf.placeholder(tf.int32,   [None])
        # placeholder for current reward
        self.rew_t_ph              = tf.placeholder(tf.float32, [None])
        # placeholder for next observation (or state)
        self.obs_tp1_ph            = tf.placeholder(tf.uint8, [None] + list(input_shape))
        # placeholder for end of episode mask
        # this value is 1 if the next state corresponds to the end of an episode,
        # in which case there is no Q-value at the next state; at the end of an
        # episode, only the current state reward contributes to the target, not the
        # next state Q-value (i.e. target is just rew_t_ph, not rew_t_ph + gamma * q_tp1)
        self.done_mask_ph          = tf.placeholder(tf.float32, [None])

        # casting to float on GPU ensures lower data transfer times.
        self.obs_t_float   = tf.cast(self.obs_t_ph,   tf.float32) / 255.0
        self.obs_tp1_float = tf.cast(self.obs_tp1_ph, tf.float32) / 255.0

        # Here, you should fill in your own code to compute the Bellman error. This requires
        # evaluating the current and next Q-values and constructing the corresponding error.
        # TensorFlow will differentiate this error for you, you just need to pass it to the
        # optimizer. See assignment text for details.
        # Your code should produce one scalar-valued tensor: total_error
        # This will be passed to the optimizer in the provided code below.
        # Your code should also produce two collections of variables:
        # q_func_vars
        # target_q_func_vars
        # These should hold all of the variables of the Q-function network and target network,
        # respectively. A convenient way to get these is to make use of TF's "scope" feature.
        # For example, you can create your Q-function network with the scope "q_func" like this:
        # <something> = q_func(obs_t_float, num_actions, scope="q_func", reuse=False)
        # And then you can obtain the variables like this:
        # q_func_vars = tf.get_colletction(tf.GraphKeys.GLOBAL_VARIABLES, scope='q_func')
        # Older versions of TensorFlow may require using "VARIABLES" instead of "GLOBAL_VARIABLES"
        # Tip: use huber_loss (from dqn_utils) instead of squared error when defining self.total_error
        ######

        # YOUR CODE HERE
        curr_q_eval = q_func(self.obs_t_float, scope="q_func", reuse=False)
        self.q_action = q_func(self.obs_t_float, 'q_func', reuse=True)
        target_q_action = q_func(self.obs_tp1_float, 'target_func', reuse=False)
        if double_q:
            target_actions = tf.argmax(curr_q_eval, output_type=tf.int32)
            action_idx = tf.stack([tf.range(0, tf.shape(curr_q_eval)[0]), target_actions], axis=1)
            gamma_max_future_q_targets = tf.scalar_mul(gamma, tf.gather_nd(target_q_action, action_idx))
        else:
            gamma_max_future_q_targets = tf.scalar_mul(gamma, tf.reduce_max(target_q_action))
        
        q_targets = tf.stop_gradient(tf.add(self.rew_t_ph, gamma_max_future_q_targets - tf.multiply(self.done_mask_ph, gamma_max_future_q_targets)))
        idx = tf.range(0, tf.shape(self.act_t_ph)[0])
        cat_idx = tf.stack([idx, self.act_t_ph], axis=1)
        current_q_values = tf.gather_nd(curr_q_eval, cat_idx)
        self.total_error = tf.reduce_sum(huber_loss(current_q_values - q_targets))
        q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='q_func')
        target_q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_func')
        ######

        # construct optimization op (with gradient clipping)
        self.learning_rate = tf.placeholder(tf.float32, (), name="learning_rate")
        optimizer = self.optimizer_spec.constructor(learning_rate=self.learning_rate, **self.optimizer_spec.kwargs)
        self.train_fn = minimize_and_clip(optimizer, self.total_error,
                    var_list=q_func_vars, clip_val=grad_norm_clipping)

        # update_target_fn will be called periodically to copy Q network to target Q network
        update_target_fn = []
        for var, var_target in zip(sorted(q_func_vars,        key=lambda v: v.name),
                                sorted(target_q_func_vars, key=lambda v: v.name)):
            update_target_fn.append(var_target.assign(var))
        self.update_target_fn = tf.group(*update_target_fn)

        # construct the replay buffer
        self.replay_buffer = ReplayBuffer(replay_buffer_size)
        self.replay_buffer_idx = None

        ###############
        # RUN ENV     #
        ###############
        self.model_initialized = False
        self.num_param_updates = 0
        self.mean_episode_reward      = -float('nan')
        self.best_mean_episode_reward = -float('inf')
        self.last_obs = self.env.reset()
        self.log_every_n_steps = 1000

        self.start_time = None
        self.t = 0

        ###############
        # SETUP LOGGING #
        ###############
        if not(os.path.exists('data')):
            os.makedirs('data')
        logdir = 'img-'
        logdir = logdir + time.strftime("%d-%m-%Y_%H-%M-%S")
        logdir = os.path.join('data', logdir)
        logz.configure_output_dir(logdir)


    def stopping_criterion_met(self):
        return self.t >= self.total_time_steps

    def choose_random_action(self, last_obs, epsilon):
        # Randomly selects a legal action
        epsilon_flip = np.random.uniform()
        if epsilon_flip < 1/3:
            # Pen down
            pen_loc_map = last_obs[:, :, 5]
            x_dim, y_dim = pen_loc_map.shape
            pen_x, pen_y = next((idx for idx, val in np.ndenumerate(pen_loc_map) if val==1), (-1, -1))
            if pen_x == -1:
                # 1 not in last state map => pen up was the last action
                x_rnd, y_rnd = np.random.randint(0, x_dim-1), np.random.randint(0, y_dim-1)
            else:
                x_min, y_min = max(0, pen_x - self.pixel_limit//2), max(0, pen_y - self.pixel_limit//2)
                x_max, y_max = min(pen_x + self.pixel_limit//2, x_dim - 1),  min(pen_y + self.pixel_limit//2, y_dim - 1)
                distance_based_exploration_th = 0.01
                if epsilon > distance_based_exploration_th:
                    # Explore geometrically, giving points further from pen current x and y locations preference,
                    # weighted by the current exploration factor
                    x_range, y_range = np.arange(x_min, x_max), np.arange(y_min, y_max)
                    smoothing_factor = 1
                    x_weights, y_weights = np.abs(pen_x - x_range)*epsilon + smoothing_factor, np.abs(pen_y - y_range)*epsilon + smoothing_factor
                    x_p, y_p = x_weights/sum(x_weights), y_weights/sum(y_weights)
                    x_rnd, y_rnd = np.random.choice(x_range, p=x_p), np.random.choice(y_range, p=y_p)
                else:
                    x_rnd, y_rnd = np.random.randint(x_min, x_max), np.random.randint(y_min, y_max)
            return 2 + x_rnd * window_size + y_rnd
        elif epsilon_flip < 2/3:
            # Pen up
            return 0
        else:
            # Draw finish
            return 1

    def step_env(self):
        ### 2. Step the env and store the transition
        # At this point, "self.last_obs" contains the latest observation that was
        # recorded from the simulator. Here, your code needs to store this
        # observation and its outcome (reward, next observation, etc.) into
        # the replay buffer while stepping the simulator forward one step.
        # At the end of this block of code, the simulator should have been
        # advanced one step, and the replay buffer should contain one more
        # transition.
        # Specifically, self.last_obs must point to the new latest observation.
        # Useful functions you'll need to call:
        # obs, reward, done, info = env.step(action)
        # this steps the environment forward one step
        # obs = env.reset()
        # this resets the environment if you reached an episode boundary.
        # Don't forget to call env.reset() to get a new observation if done
        # is true!!
        # Don't forget to include epsilon greedy exploration!
        # And remember that the first time you enter this loop, the model
        # may not yet have been initialized (but of course, the first step
        # might as well be random, since you haven't trained your net...)

        #####

        # YOUR CODE HERE
        buf_idx = self.replay_buffer.store_observation(self.last_obs)
        epsilon = self.exploration.value(self.t)
        if not self.model_initialized:
            # Completely random
            action = self.choose_random_action(self.last_obs, epsilon)
            #action = self.hull_policy.get_action(self.last_obs, self.env.curr_mask)
        else:
            epsilon_flip = np.random.binomial(1, epsilon)
            if epsilon_flip == 1:
                action = self.choose_random_action(self.last_obs, epsilon)
            else:
                q_values = self.session.run(tf.squeeze(self.q_action), {self.obs_t_ph: np.expand_dims(self.last_obs, axis=0)})
                action = np.argmax(q_values)
        obs, reward, done = self.env.step(action)
        self.replay_buffer.store_effect(buf_idx, action, reward, done)
        if done:
            self.last_obs = self.env.reset()
        else:
            self.last_obs = obs

    def update_model(self):
        ### 3. Perform experience replay and train the network.
        # note that this is only done if the replay buffer contains enough samples
        # for us to learn something useful -- until then, the model will not be
        # initialized and random actions should be taken
        if (self.t > self.learning_starts and \
            self.t % self.learning_freq == 0 and \
            self.replay_buffer.can_sample(self.batch_size)):

        # Here, you should perform training. Training consists of four steps:
        # 3.a: use the replay buffer to sample a batch of transitions (see the
        # replay buffer code for function definition, each batch that you sample
        # should consist of current observations, current actions, rewards,
        # next observations, and done indicator).
        # 3.b: initialize the model if it has not been initialized yet; to do
        # that, call
        #    initialize_interdependent_variables(self.session, tf.global_variables(), {
        #        self.obs_t_ph: obs_t_batch,
        #        self.obs_tp1_ph: obs_tp1_batch,
        #    })
        # where obs_t_batch and obs_tp1_batch are the batches of observations at
        # the current and next time step. The boolean variable model_initialized
        # indicates whether or not the model has been initialized.
        # Remember that you have to update the target network too (see 3.d)!
        # 3.c: train the model. To do this, you'll need to use the self.train_fn and
        # self.total_error ops that were created earlier: self.total_error is what you
        # created to compute the total Bellman error in a batch, and self.train_fn
        # will actually perform a gradient step and update the network parameters
        # to reduce total_error. When calling self.session.run on these you'll need to
        # populate the following placeholders:
        # self.obs_t_ph
        # self.act_t_ph
        # self.rew_t_ph
        # self.obs_tp1_ph
        # self.done_mask_ph
        # (this is needed for computing self.total_error)
        # self.learning_rate -- you can get this from self.optimizer_spec.lr_schedule.value(t)
        # (this is needed by the optimizer to choose the learning rate)
        # 3.d: periodically update the target network by calling
        # self.session.run(self.update_target_fn)
        # you should update every target_update_freq steps, and you may find the
        # variable self.num_param_updates useful for this (it was initialized to 0)
        #####

        # YOUR CODE HERE
            obs_t_batch, act_t_batch, rew_t_batch, obs_tp1_batch, done_mask = self.replay_buffer.sample(self.batch_size)
            if not self.model_initialized:
                initialize_interdependent_variables(self.session, tf.global_variables(), 
                {
                self.obs_t_ph: obs_t_batch,
                self.obs_tp1_ph: obs_tp1_batch,
                })
                self.model_initialized = True
            self.session.run([self.total_error, self.train_fn], {
                self.obs_t_ph: obs_t_batch,
                self.act_t_ph: act_t_batch,
                self.rew_t_ph: rew_t_batch,
                self.obs_tp1_ph: obs_tp1_batch,
                self.done_mask_ph: done_mask,
                self.learning_rate: self.optimizer_spec.lr_schedule.value(self.t)
                })
            self.num_param_updates += 1
            if self.num_param_updates % self.target_update_freq == 0:
                self.session.run(self.update_target_fn)
        self.t += 1

    def log_progress(self):
        if self.t % self.log_every_n_steps == 0 and self.model_initialized:
            log_episodes = 5
            episodes = [self.predict(self.env) for i in range(log_episodes)]
            episode_results = [prediction[0] for prediction in episodes]
            episode_rewards = [prediction[1] for prediction in episodes]
            episode_lengths = [prediction[2] for prediction in episodes]
            self.mean_episode_reward = sum(episode_rewards)/len(episode_rewards)
            self.best_mean_episode_reward = max(self.best_mean_episode_reward, self.mean_episode_reward)
            print("Timestep %d" % (self.t,))
            print("mean reward (5 episodes) %f" % self.mean_episode_reward)
            print("best mean reward %f" % self.best_mean_episode_reward)
            print("exploration %f" % self.exploration.value(self.t))
            print("learning_rate %f" % self.optimizer_spec.lr_schedule.value(self.t))
            print("Episode lengths ", episode_lengths)
            if self.start_time is not None:
                print("running time %f" % ((time.time() - self.start_time) / 60.))
            self.start_time = time.time()
            logz.log_tabular("Timestep", self.t)
            logz.log_tabular("Mean Reward (5 episodes)", self.mean_episode_reward)
            logz.log_tabular("Best Mean Reward", self.best_mean_episode_reward)
            logz.dump_tabular()
            sys.stdout.flush()
            with open(self.rew_file, 'wb') as f:
                pickle.dump(episode_rewards, f, pickle.HIGHEST_PROTOCOL)
            if self.progress_dir is None:
                return
            for count, result in enumerate(episode_results):
                result_file_name = "result_" + str(count) + "_t_" + str(self.t) + ".npy"
                np.save('%s/%s'%(self.progress_dir, result_file_name), result)

    
    def predict(self, test_env):
        # Runs the prediction algorithm on one image, and returns [img_c_1, img_c_2, img_c_3, img_mask], reward
        # since we no longer need the last two, and to keep the reward for logging purposes
        done = False
        self.last_obs = test_env.reset()
        count = 0
        reward_sum = 0
        while(not done and count < 100):
            q_values = self.session.run([tf.squeeze(self.q_action)], {self.obs_t_ph: np.expand_dims(self.last_obs, axis=0)})
            action = np.argmax(q_values)
            obs, reward, done = test_env.step(action)
            self.last_obs = obs
            reward_sum += reward
            count += 1
        if not done:
            # Run a pen finish
            action = 1
            obs, reward, done = test_env.step(action)
            self.last_obs = obs
            reward_sum += reward
        #return self.last_obs[:,:,:4], reward_sum
        return test_env.get_full_state(), reward_sum, count
    
    def test(self, test_env, num_test_samples):
        results, rewards = [], []
        for sample in range(num_test_samples):
            curr_result, curr_reward, _ = self.predict(test_env)
            results.append(curr_result)
            rewards.append(curr_reward)
        return results, rewards


def learn(*args, **kwargs):
    alg = QLearner(*args, **kwargs)
    while not alg.stopping_criterion_met():
        alg.step_env()
        # at this point, the environment should have been advanced one step (and
        # reset if done was true), and self.last_obs should point to the new latest 
        # observation
        alg.update_model()
        alg.log_progress()
    return alg



    
