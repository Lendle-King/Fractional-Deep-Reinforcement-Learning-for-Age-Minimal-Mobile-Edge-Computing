"""
The double DQN based on this paper: https://arxiv.org/abs/1509.06461
View more on my tutorial page: https://morvanzhou.github.io/tutorials/
Using:
Tensorflow: 1.0
gym: 0.8.0
"""
import os

import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

np.random.seed(1)
tf.set_random_seed(1)


class DoubleDQN:
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
