import os, time
import tensorflow as tf
import numpy as np

tf.compat.v1.disable_eager_execution()


class OUActionNoise(object):
    def __init__(self, mu, sigma=0.15, theta=.2, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev)*self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)

        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)


class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = 1 - done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal


class Actor(object):
    def __init__(self, lr, n_actions, name, input_dims, sess, fc1_dims,
                 fc2_dims, action_bound, batch_size=64, chkpt_dir='tmp/ddpg'):
        self.lr = lr
        self.n_actions = n_actions
        self.name = name
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.sess = sess
        self.action_bound = action_bound
        self.build_network()
        self.params = tf.compat.v1.trainable_variables(scope=self.name)
        self.saver = tf.compat.v1.train.Saver()
        self.checkpoint_file = os.path.join(chkpt_dir, name + '_ddpg.ckpt')

        self.unnormalized_actor_gradients = tf.gradients(
            self.mu, self.params, -self.action_gradient)

        self.actor_gradients = list(map(lambda x: tf.divide(x, self.batch_size),
                                        self.unnormalized_actor_gradients))

        self.optimize = tf.compat.v1.train.AdamOptimizer(self.lr).\
            apply_gradients(zip(self.actor_gradients, self.params))

    def build_network(self):
        with tf.compat.v1.variable_scope(self.name):
            self.input = tf.compat.v1.placeholder(tf.float32,
                                                  shape=[None, *self.input_dims],
                                                  name='inputs')

            self.action_gradient = tf.compat.v1.placeholder(tf.float32,
                                                            shape=[None, self.n_actions],
                                                            name='gradients')
            f1 = 1. / np.sqrt(self.fc1_dims)
            dense1 = tf.compat.v1.layers.dense(self.input, units=self.fc1_dims,
                                               kernel_initializer=tf.compat.v1.initializers.random_uniform(-f1, f1),
                                               bias_initializer=tf.compat.v1.initializers.random_uniform(-f1, f1))
            batch1 = tf.compat.v1.layers.batch_normalization(dense1)
            layer1_activation = tf.nn.relu(batch1)

            f2 = 1. / np.sqrt(self.fc2_dims)
            dense2 = tf.compat.v1.layers.dense(layer1_activation, units=self.fc2_dims,
                                               kernel_initializer=tf.compat.v1.initializers.random_uniform(-f2, f2),
                                               bias_initializer=tf.compat.v1.initializers.random_uniform(-f2, f2))
            batch2 = tf.compat.v1.layers.batch_normalization(dense2)
            layer2_activation = tf.nn.relu(batch2)

            f3 = 0.003
            mu = tf.compat.v1.layers.dense(layer2_activation, units=self.n_actions,
                                           activation='tanh',
                                           kernel_initializer=tf.compat.v1.initializers.random_uniform(-f3, f3),
                                           bias_initializer=tf.compat.v1.initializers.random_uniform(-f3, f3))

            self.mu = tf.multiply(mu, self.action_bound)

    def predict(self, inputs):
        return self.sess.run(self.mu, feed_dict={self.input:inputs})

    def train(self, inputs, gradients):
        self.sess.run(self.optimize,
                      feed_dict={self.input:inputs,
                                 self.action_gradient: gradients})

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.saver.restore(self.sess, self.checkpoint_file)

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        self.saver.save(self.sess, self.checkpoint_file)


class Critic(object):
    def __init__(self, lr, n_actions, name, input_dims, sess, fc1_dims,
                 fc2_dims, batch_size=64, chkpt_dir='tmp/ddpg'):
        self.lr = lr
        self.n_actions = n_actions
        self.name = name
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.sess = sess
        self.build_network()
        self.params = tf.compat.v1.trainable_variables(scope=self.name)
        self.saver = tf.compat.v1.train.Saver()
        self.checkpoint_file = os.path.join(chkpt_dir, name + '_ddpg.ckpt')

        self.optimize = tf.compat.v1.train.AdamOptimizer(self.lr).minimize(self.loss)

        self.action_gradients = tf.gradients(self.q, self.actions)

    def build_network(self):
        with tf.compat.v1.variable_scope(self.name):
            self.input = tf.compat.v1.placeholder(tf.float32,
                                                  shape = [None, *self.input_dims],
                                                  name ='inputs')
            self.actions = tf.compat.v1.placeholder(tf.float32,
                                                    shape=[None, self.n_actions],
                                                    name='actions')
            self.q_target = tf.compat.v1.placeholder(tf.float32,
                                                     shape=[None,1],
                                                     name='targets')
            f1 = 1. / np.sqrt(self.fc1_dims)
            dense1 = tf.compat.v1.layers.dense(self.input, units=self.fc1_dims,
                                               kernel_initializer=tf.compat.v1.initializers.random_uniform(-f1, f1),
                                               bias_initializer=tf.compat.v1.initializers.random_uniform(-f1, f1))
            batch1 = tf.compat.v1.layers.batch_normalization(dense1)
            layer1_activation = tf.nn.relu(batch1)

            f2 = 1. / np.sqrt(self.fc2_dims)
            dense2 = tf.compat.v1.layers.dense(layer1_activation, units=self.fc2_dims,
                                               kernel_initializer=tf.compat.v1.initializers.random_uniform(-f2, f2),
                                               bias_initializer=tf.compat.v1.initializers.random_uniform(-f2, f2))
            batch2 = tf.compat.v1.layers.batch_normalization(dense2)

            action_in = tf.compat.v1.layers.dense(self.actions, units=self.fc2_dims,
                                                  activation='relu')

            state_actions = tf.add(batch2, action_in)
            state_actions = tf.nn.relu(state_actions)

            f3 = 0.003
            self.q = tf.compat.v1.layers.dense(state_actions, units=1,
                                               kernel_initializer=tf.compat.v1.initializers.random_uniform(-f3, f3),
                                               bias_initializer=tf.compat.v1.initializers.random_uniform(-f3, f3),
                                               kernel_regularizer=tf.keras.regularizers.l2(0.01))
            self.loss = tf.compat.v1.losses.mean_squared_error(self.q_target, self.q)

    def predict(self, inputs, actions):
        return self.sess.run(self.q,
                             feed_dict={self.input: inputs,
                                        self.actions: actions})

    def train(self, inputs, actions, q_target):
        return self.sess.run(self.optimize,
                             feed_dict={self.input: inputs,
                                        self.actions: actions,
                                        self.q_target: q_target})

    def get_action_gradients(self, inputs, actions):
        return self.sess.run(self.action_gradients,
                             feed_dict={self.input: inputs,
                                        self.actions: actions})

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.saver.restore(self.sess, self.checkpoint_file)

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        self.saver.save(self.sess, self.checkpoint_file)


class Agent(object):
    def __init__(self, alpha, beta, input_dims, tau, env, gamma=0.99,
                 n_actions=2, max_size = 1000000, layer1_size=400, layer2_size=300,
                 batch_size=64, chkpt_dir='tmp/ddpg'):

        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.sess = tf.compat.v1.Session()

        self.Actor = Actor(alpha, n_actions, 'Actor', input_dims, self.sess,
                           layer1_size, layer2_size, env.action_space.high,
                           chkpt_dir=chkpt_dir)

        self.Critic = Critic(beta, n_actions, 'Critic', input_dims, self.sess,
                             layer1_size, layer2_size, chkpt_dir=chkpt_dir)

        self.target_Actor = Actor(alpha, n_actions, 'target_Actor', input_dims, self.sess,
                                  layer1_size, layer2_size, env.action_space.high,
                                  chkpt_dir=chkpt_dir)

        self.target_Critic = Critic(beta, n_actions, 'target_Critic', input_dims, self.sess,
                                    layer1_size, layer2_size, chkpt_dir=chkpt_dir)

        self.noise = OUActionNoise(mu=np.zeros(n_actions))

        self.update_critic = \
         [self.target_Critic.params[i].assign(
             tf.multiply(self.Critic.params[i], self.tau) \
             + tf.multiply(self.target_Critic.params[i], 1. - self.tau))
             for i in range(len(self.target_Critic.params))]

        self.update_actor = \
            [self.target_Actor.params[i].assign(
                tf.multiply(self.Actor.params[i], self.tau) \
                + tf.multiply(self.target_Actor.params[i], 1. - self.tau))
                for i in range(len(self.target_Actor.params))]

        self.sess.run(tf.compat.v1.global_variables_initializer())
        self.update_network_parameters(first=True)

    def update_network_parameters(self, first=True):
        if first:
            old_tau = self.tau
            self.tau = 1.0
            self.target_Critic.sess.run(self.update_critic)
            self.target_Actor.sess.run(self.update_actor)
            self.tau = old_tau
        else:
            self.target_Critic.sess.run(self.update_critic)
            self.target_Actor.sess.run(self.update_actor)

    def remember(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def choose_action(self, state):
        state = state[np.newaxis, :]
        mu = self.Actor.predict(state)

        mu_prime = mu + self.noise()

        return mu_prime[0]

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        state, action, reward, new_state, done =\
                            self.memory.sample_buffer(self.batch_size)
        critic_value_ = self.target_Critic.predict(new_state,
                                                   self.target_Actor.predict(new_state))

        target = []
        for j in range(self.batch_size):
            target.append(reward[j] + self.gamma*critic_value_[j]*done[j])
        target = np.reshape(target, (self.batch_size, 1))

        _ = self.Critic.train(state, action, target)

        a_outs = self.Actor.predict(state)
        grads = self.Critic.get_action_gradients(state, a_outs)
        self.Actor.train(state, grads[0])

        self.update_network_parameters()

    def save_models(self):
        self.Actor.save_checkpoint()
        self.target_Actor.save_checkpoint()
        self.Critic.save_checkpoint()
        self.target_Critic.save_checkpoint()

    def load_models(self):
        self.Actor.load_checkpoint()
        self.target_Actor.load_checkpoint()
        self.Critic.load_checkpoint()
        self.target_Critic.load_checkpoint()

