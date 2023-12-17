import numpy as np
import random
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
import os
import math
from collections import deque

# D3QN
np.random.seed(1)
tf.set_random_seed(1)


class D3QN:
    def __init__(
        self,
        args,
        n_actions,
        n_features,
        reward_decay=0.9,
        e_greedy=1,
        replace_target_iter=200,
        memory_size=3000,
        e_greedy_increment=10,
        output_graph=False,
        double_q=True,
        sess=None,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = args.d3qn_lr
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = args.d3qn_batch
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0

        self.double_q = double_q  # decide to use double q or not
        self.dueling = True

        self.learn_step_counter = 0
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        self._build_net()
        t_params = tf.get_collection("target_net_params")
        e_params = tf.get_collection("eval_net_params")
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        if sess is None:
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())
        else:
            self.sess = sess
        if output_graph:
            tf.summary.FileWriter("logs/", self.sess.graph)
        self.cost_his = []

    def _build_net(self):
        tf.reset_default_graph()

        def build_layers(s, c_names, n_l1, w_initializer, b_initializer):
            # first layer
            with tf.variable_scope("l1"):
                w1 = tf.get_variable(
                    "w1",
                    [self.n_features, n_l1],
                    initializer=w_initializer,
                    collections=c_names,
                )
                b1 = tf.get_variable(
                    "b1", [1, n_l1], initializer=b_initializer, collections=c_names
                )
                l1 = tf.nn.relu(tf.matmul(tf.concat([s], 1), w1) + b1)

            # second layer
            with tf.variable_scope("l12"):
                w12 = tf.get_variable(
                    "w12", [n_l1, n_l1], initializer=w_initializer, collections=c_names
                )
                b12 = tf.get_variable(
                    "b12", [1, n_l1], initializer=b_initializer, collections=c_names
                )
                l12 = tf.nn.relu(tf.matmul(l1, w12) + b12)

            # the second layer is different
            if self.dueling:
                # Dueling DQN
                # a single output n_l1 -> 1
                with tf.variable_scope("Value"):
                    w2 = tf.get_variable(
                        "w2", [n_l1, 1], initializer=w_initializer, collections=c_names
                    )
                    b2 = tf.get_variable(
                        "b2", [1, 1], initializer=b_initializer, collections=c_names
                    )
                    self.V = tf.matmul(l12, w2) + b2
                # n_l1 -> n_actions
                with tf.variable_scope("Advantage"):
                    w2 = tf.get_variable(
                        "w2",
                        [n_l1, self.n_actions],
                        initializer=w_initializer,
                        collections=c_names,
                    )
                    b2 = tf.get_variable(
                        "b2",
                        [1, self.n_actions],
                        initializer=b_initializer,
                        collections=c_names,
                    )
                    self.A = tf.matmul(l12, w2) + b2

                with tf.variable_scope("Q"):
                    out = self.V + (
                        self.A - tf.reduce_mean(self.A, axis=1, keep_dims=True)
                    )  # Q = V(s) +A(s,a)

            else:
                with tf.variable_scope("Q"):
                    w2 = tf.get_variable(
                        "w2",
                        [n_l1, self.n_actions],
                        initializer=w_initializer,
                        collections=c_names,
                    )
                    b2 = tf.get_variable(
                        "b2",
                        [1, self.n_actions],
                        initializer=b_initializer,
                        collections=c_names,
                    )
                    out = tf.matmul(l1, w2) + b2
            return out

        # ------------------ build evaluate_net ------------------

        self.s = tf.placeholder(tf.float32, [None, self.n_features], name="s")  # input
        self.q_target = tf.placeholder(
            tf.float32, [None, self.n_actions], name="Q_target"
        )  # for calculating loss

        with tf.variable_scope("eval_net"):
            c_names, n_l1, w_initializer, b_initializer = (
                ["eval_net_params", tf.GraphKeys.GLOBAL_VARIABLES],
                20,
                tf.random_normal_initializer(0.0, 0.3),
                tf.constant_initializer(0.1),
            )  # config of layers

            self.q_eval = build_layers(
                self.s, c_names, n_l1, w_initializer, b_initializer
            )

        with tf.variable_scope("loss"):
            self.loss = tf.reduce_mean(
                tf.squared_difference(self.q_target, self.q_eval)
            )
        with tf.variable_scope("train"):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        # ------------------ build target_net ------------------
        self.s_ = tf.placeholder(
            tf.float32, [None, self.n_features], name="s_"
        )  # input
        with tf.variable_scope("target_net"):
            c_names = ["target_net_params", tf.GraphKeys.GLOBAL_VARIABLES]

            self.q_next = build_layers(
                self.s_, c_names, n_l1, w_initializer, b_initializer
            )

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, "memory_counter"):
            self.memory_counter = 0
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, observation):
        observation = observation[np.newaxis, :]
        actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
        action = np.argmax(actions_value)

        if not hasattr(self, "q"):  # record action value it gets
            self.q = []
            self.running_q = 0
        self.running_q = self.running_q * 0.99 + 0.01 * np.max(actions_value)
        self.q.append(self.running_q)

        if np.random.uniform() > self.epsilon:  # choosing action
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            # print('\ntarget_params_replaced\n')

        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        q_next, q_eval4next = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={
                self.s_: batch_memory[:, -self.n_features :],  # next observation
                self.s: batch_memory[:, -self.n_features :],
            },
        )  # next observation
        q_eval = self.sess.run(
            self.q_eval, {self.s: batch_memory[:, : self.n_features]}
        )

        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]

        if self.double_q:
            max_act4next = np.argmax(
                q_eval4next, axis=1
            )  # the action that brings the highest value is evaluated by q_eval
            selected_q_next = q_next[
                batch_index, max_act4next
            ]  # Double DQN, select q_next depending on above actions
        else:
            selected_q_next = np.max(q_next, axis=1)  # the natural DQN

        q_target[batch_index, eval_act_index] = reward + self.gamma * selected_q_next

        _, self.cost = self.sess.run(
            [self._train_op, self.loss],
            feed_dict={
                self.s: batch_memory[:, : self.n_features],
                self.q_target: q_target,
            },
        )
        self.cost_his.append(self.cost)

        self.epsilon = (
            self.epsilon + self.epsilon_increment
            if self.epsilon < self.epsilon_max
            else self.epsilon_max
        )
        # print(self.epsilon)
        self.learn_step_counter += 1


## DDPG
random_seed = 1

np.random.seed(random_seed)  # set random seed for numpy
tf.random.set_random_seed(random_seed)  # set random seed for tensorflow-cpu
os.environ["TF_DETERMINISTIC_OPS"] = "1"  # set random seed for tensorflow-gpu

# Hyper Parameters:

REPLAY_BUFFER_SIZE = 1000000
REPLAY_START_SIZE = 10000
GAMMA = 0.99


class DDPG:
    """docstring for DDPG"""

    def __init__(self, args, env):
        tf.reset_default_graph()
        self.name = "DDPG"  # name for uploading results
        self.environment = env
        # Randomly initialize actor network and critic network
        # with both their target networks
        self.state_dim = 1
        self.action_dim = 1
        self.batch_size = args.ddpg_batch
        self.sess = tf.InteractiveSession()

        self.actor_network = ActorNetwork(
            args, self.sess, self.state_dim, self.action_dim
        )
        self.critic_network = CriticNetwork(
            args, self.sess, self.state_dim, self.action_dim
        )

        # initialize replay buffer
        self.replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)

        # Initialize a random process the Ornstein-Uhlenbeck process for action exploration
        self.exploration_noise = OUNoise(self.action_dim)

    def train(self):
        # print "train step",self.time_step
        # Sample a random minibatch of N transitions from replay buffer
        minibatch = self.replay_buffer.get_batch(self.batch_size)
        state_batch = np.asarray([data[0] for data in minibatch])
        action_batch = np.asarray([data[1] for data in minibatch])
        reward_batch = np.asarray([data[2] for data in minibatch])
        next_state_batch = np.asarray([data[3] for data in minibatch])
        done_batch = np.asarray([data[4] for data in minibatch])

        # for action_dim = 1
        action_batch = np.resize(action_batch, [self.batch_size, self.action_dim])

        # Calculate y_batch

        next_action_batch = self.actor_network.target_actions(next_state_batch)
        q_value_batch = self.critic_network.target_q(
            next_state_batch, next_action_batch
        )
        y_batch = []
        for i in range(len(minibatch)):
            if done_batch[i]:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch[i] + GAMMA * q_value_batch[i])
        y_batch = np.resize(y_batch, [self.batch_size, 1])
        # Update critic by minimizing the loss L
        self.critic_network.train(y_batch, state_batch, action_batch)

        # Update the actor policy using the sampled gradient:
        action_batch_for_gradients = self.actor_network.actions(state_batch)
        q_gradient_batch = self.critic_network.gradients(
            state_batch, action_batch_for_gradients
        )

        self.actor_network.train(q_gradient_batch, state_batch)

        # Update the target networks
        self.actor_network.update_target()
        self.critic_network.update_target()

    def noise_action(self, state):
        # Select action a_t according to the current policy and exploration noise
        action = self.actor_network.action(state)
        return action, action + self.exploration_noise.noise()

    def action(self, state):
        action = self.actor_network.action(state)
        return action

    def perceive(self, state, action, reward, next_state, done):
        # Store transition (s_t,a_t,r_t,s_{t+1}) in replay buffer
        self.replay_buffer.add(state, action, reward, next_state, done)

        # Re-iniitialize the random process when an episode ends
        if done:
            self.exploration_noise.reset()


# Actor Network
# Hyper Parameters
LAYER1_SIZE = 400
LAYER2_SIZE = 300
TAU = 0.001


class ActorNetwork:
    """docstring for ActorNetwork"""

    def __init__(self, args, sess, state_dim, action_dim):
        self.actor_lr = args.actor_lr
        self.sess = sess
        self.state_dim = state_dim
        self.action_dim = action_dim
        # create actor network
        (
            self.state_input,
            self.action_output,
            self.net,
            self.is_training,
        ) = self.create_network(state_dim, action_dim)

        # create target actor network
        (
            self.target_state_input,
            self.target_action_output,
            self.target_update,
            self.target_is_training,
        ) = self.create_target_network(state_dim, action_dim, self.net)

        # define training rules
        self.create_training_method()

        self.sess.run(tf.initialize_all_variables())

        self.update_target()
        # self.load_network()

    def create_training_method(self):
        self.q_gradient_input = tf.placeholder("float", [None, self.action_dim])
        self.parameters_gradients = tf.gradients(
            self.action_output, self.net, -self.q_gradient_input
        )
        self.optimizer = tf.train.AdamOptimizer(self.actor_lr).apply_gradients(
            zip(self.parameters_gradients, self.net)
        )

    def create_network(self, state_dim, action_dim):
        layer1_size = LAYER1_SIZE
        layer2_size = LAYER2_SIZE

        state_input = tf.placeholder("float", [None, state_dim])
        is_training = tf.placeholder(tf.bool)

        W1 = self.variable([state_dim, layer1_size], state_dim)
        b1 = self.variable([layer1_size], state_dim)
        W2 = self.variable([layer1_size, layer2_size], layer1_size)
        b2 = self.variable([layer2_size], layer1_size)
        W3 = tf.Variable(tf.random_uniform([layer2_size, action_dim], -3e-3, 3e-3))
        b3 = tf.Variable(tf.random_uniform([action_dim], -3e-3, 3e-3))

        layer0_bn = self.batch_norm_layer(
            state_input,
            training_phase=is_training,
            scope_bn="batch_norm_0",
            activation=tf.identity,
        )
        layer1 = tf.matmul(layer0_bn, W1) + b1
        layer1_bn = self.batch_norm_layer(
            layer1,
            training_phase=is_training,
            scope_bn="batch_norm_1",
            activation=tf.nn.relu,
        )
        layer2 = tf.matmul(layer1_bn, W2) + b2
        layer2_bn = self.batch_norm_layer(
            layer2,
            training_phase=is_training,
            scope_bn="batch_norm_2",
            activation=tf.nn.relu,
        )

        action_output = tf.tanh(tf.matmul(layer2_bn, W3) + b3)

        return state_input, action_output, [W1, b1, W2, b2, W3, b3], is_training

    def create_target_network(self, state_dim, action_dim, net):
        state_input = tf.placeholder("float", [None, state_dim])
        is_training = tf.placeholder(tf.bool)
        ema = tf.train.ExponentialMovingAverage(decay=1 - TAU)
        target_update = ema.apply(net)
        target_net = [ema.average(x) for x in net]

        layer0_bn = self.batch_norm_layer(
            state_input,
            training_phase=is_training,
            scope_bn="target_batch_norm_0",
            activation=tf.identity,
        )

        layer1 = tf.matmul(layer0_bn, target_net[0]) + target_net[1]
        layer1_bn = self.batch_norm_layer(
            layer1,
            training_phase=is_training,
            scope_bn="target_batch_norm_1",
            activation=tf.nn.relu,
        )
        layer2 = tf.matmul(layer1_bn, target_net[2]) + target_net[3]
        layer2_bn = self.batch_norm_layer(
            layer2,
            training_phase=is_training,
            scope_bn="target_batch_norm_2",
            activation=tf.nn.relu,
        )

        action_output = tf.tanh(tf.matmul(layer2_bn, target_net[4]) + target_net[5])

        return state_input, action_output, target_update, is_training

    def update_target(self):
        self.sess.run(self.target_update)

    def train(self, q_gradient_batch, state_batch):
        self.sess.run(
            self.optimizer,
            feed_dict={
                self.q_gradient_input: q_gradient_batch,
                self.state_input: state_batch,
                self.is_training: True,
            },
        )

    def actions(self, state_batch):
        return self.sess.run(
            self.action_output,
            feed_dict={self.state_input: state_batch, self.is_training: True},
        )

    def action(self, state):
        return self.sess.run(
            self.action_output,
            feed_dict={self.state_input: [state], self.is_training: False},
        )[0]

    def target_actions(self, state_batch):
        return self.sess.run(
            self.target_action_output,
            feed_dict={
                self.target_state_input: state_batch,
                self.target_is_training: True,
            },
        )

    # f fan-in size
    def variable(self, shape, f):
        return tf.Variable(
            tf.random_uniform(shape, -1 / math.sqrt(f), 1 / math.sqrt(f))
        )

    def batch_norm_layer(self, x, training_phase, scope_bn, activation=None):
        return tf.cond(
            training_phase,
            lambda: tf.layers.batch_normalization(
                x,
                center=True,
                scale=True,
                training=True,
                renorm_momentum=0.9,
                name=scope_bn,
                reuse=None,
                epsilon=1e-5,
            ),
            lambda: tf.layers.batch_normalization(
                x,
                center=True,
                scale=True,
                training=False,
                renorm_momentum=0.9,
                name=scope_bn,
                reuse=True,
                epsilon=1e-5,
            ),
        )


# Critic Network
LAYER1_SIZE = 400
LAYER2_SIZE = 300

TAU = 0.001
L2 = 0.01


class CriticNetwork:
    """docstring for CriticNetwork"""

    def __init__(self, args, sess, state_dim, action_dim):
        self.time_step = 0
        self.critic_lr = args.critic_lr
        self.sess = sess
        # create q network
        (
            self.state_input,
            self.action_input,
            self.q_value_output,
            self.net,
        ) = self.create_q_network(state_dim, action_dim)

        # create target q network (the same structure with q network)
        (
            self.target_state_input,
            self.target_action_input,
            self.target_q_value_output,
            self.target_update,
        ) = self.create_target_q_network(state_dim, action_dim, self.net)

        self.create_training_method()

        # initialization
        self.sess.run(tf.initialize_all_variables())

        self.update_target()

    def create_training_method(self):
        # Define training optimizer
        self.y_input = tf.placeholder("float", [None, 1])
        weight_decay = tf.add_n([L2 * tf.nn.l2_loss(var) for var in self.net])
        self.cost = (
            tf.reduce_mean(tf.square(self.y_input - self.q_value_output)) + weight_decay
        )
        self.optimizer = tf.train.AdamOptimizer(self.critic_lr).minimize(self.cost)
        self.action_gradients = tf.gradients(self.q_value_output, self.action_input)

    def create_q_network(self, state_dim, action_dim):
        # the layer size could be changed
        layer1_size = LAYER1_SIZE
        layer2_size = LAYER2_SIZE

        state_input = tf.placeholder("float", [None, state_dim])
        action_input = tf.placeholder("float", [None, action_dim])

        W1 = self.variable([state_dim, layer1_size], state_dim)
        b1 = self.variable([layer1_size], state_dim)
        W2 = self.variable([layer1_size, layer2_size], layer1_size + action_dim)
        W2_action = self.variable([action_dim, layer2_size], layer1_size + action_dim)
        b2 = self.variable([layer2_size], layer1_size + action_dim)
        W3 = tf.Variable(tf.random_uniform([layer2_size, 1], -3e-3, 3e-3))
        b3 = tf.Variable(tf.random_uniform([1], -3e-3, 3e-3))

        layer1 = tf.nn.relu(tf.matmul(state_input, W1) + b1)
        layer2 = tf.nn.relu(
            tf.matmul(layer1, W2) + tf.matmul(action_input, W2_action) + b2
        )
        q_value_output = tf.identity(tf.matmul(layer2, W3) + b3)

        return (
            state_input,
            action_input,
            q_value_output,
            [W1, b1, W2, W2_action, b2, W3, b3],
        )

    def create_target_q_network(self, state_dim, action_dim, net):
        state_input = tf.placeholder("float", [None, state_dim])
        action_input = tf.placeholder("float", [None, action_dim])

        ema = tf.train.ExponentialMovingAverage(decay=1 - TAU)
        target_update = ema.apply(net)
        target_net = [ema.average(x) for x in net]

        layer1 = tf.nn.relu(tf.matmul(state_input, target_net[0]) + target_net[1])
        layer2 = tf.nn.relu(
            tf.matmul(layer1, target_net[2])
            + tf.matmul(action_input, target_net[3])
            + target_net[4]
        )
        q_value_output = tf.identity(tf.matmul(layer2, target_net[5]) + target_net[6])

        return state_input, action_input, q_value_output, target_update

    def update_target(self):
        self.sess.run(self.target_update)

    def train(self, y_batch, state_batch, action_batch):
        self.time_step += 1
        self.sess.run(
            self.optimizer,
            feed_dict={
                self.y_input: y_batch,
                self.state_input: state_batch,
                self.action_input: action_batch,
            },
        )

    def gradients(self, state_batch, action_batch):
        return self.sess.run(
            self.action_gradients,
            feed_dict={self.state_input: state_batch, self.action_input: action_batch},
        )[0]

    def target_q(self, state_batch, action_batch):
        return self.sess.run(
            self.target_q_value_output,
            feed_dict={
                self.target_state_input: state_batch,
                self.target_action_input: action_batch,
            },
        )

    def q_value(self, state_batch, action_batch):
        return self.sess.run(
            self.q_value_output,
            feed_dict={self.state_input: state_batch, self.action_input: action_batch},
        )

    # f fan-in size
    def variable(self, shape, f):
        return tf.Variable(
            tf.random_uniform(shape, -1 / math.sqrt(f), 1 / math.sqrt(f))
        )


# RepalyBuffer


class ReplayBuffer(object):
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.num_experiences = 0
        self.buffer = deque()

    def get_batch(self, batch_size):
        # Randomly sample batch_size examples
        return random.sample(self.buffer, batch_size)

    def size(self):
        return self.buffer_size

    def add(self, state, action, reward, new_state, done):
        experience = (state, action, reward, new_state, done)
        if self.num_experiences < self.buffer_size:
            self.buffer.append(experience)
            self.num_experiences += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def count(self):
        # if buffer is full, return buffer size
        # otherwise, return experience counter
        return self.num_experiences

    def erase(self):
        self.buffer = deque()
        self.num_experiences = 0


# OUNoise
class OUNoise:
    """docstring for OUNoise"""

    def __init__(self, action_dimension, mu=0, theta=0.15, sigma=0.2):
        self.action_dimension = action_dimension
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state


if __name__ == "__main__":
    ou = OUNoise(3)
    states = []
    for i in range(1000):
        states.append(ou.noise())
    import matplotlib.pyplot as plt

    plt.plot(states)
    plt.show()
