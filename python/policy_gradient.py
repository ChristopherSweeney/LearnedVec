import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from rl import Environment

def mlp(x, sizes, activation=tf.tanh, output_activation=None):
    # Build a feedforward neural network.
    for size in sizes[:-1]:
        x = tf.layers.dense(x, units=size, activation=activation)
    return tf.layers.dense(x, units=sizes[-1], activation=output_activation)

def train(env, hidden_sizes=[32], lr=1e-2, 
          epochs=50, batch_size=5000,test_graph=None):

    # make environment, check spaces, get obs / act dims
    # Note, this only works for continuous observation space, discrete action space

    obs_dim = env.observation_space.shape[0]
    n_acts = env.action_space_size
    action_index = np.linspace(1.01,2,n_acts)

    # make core of policy network
    obs_ph = tf.placeholder(shape=(None, obs_dim), dtype=tf.float32)
    logits = mlp(obs_ph, sizes=hidden_sizes+[n_acts])

    # make action selection op (outputs int actions, sampled from policy)
    actions = tf.squeeze(tf.multinomial(logits=logits,num_samples=1), axis=1)


    # make loss function whose gradient, for the right data, is policy gradient
    weights_ph = tf.placeholder(shape=(None,), dtype=tf.float32)
    act_ph = tf.placeholder(shape=(None,), dtype=tf.int32)
    action_masks = tf.one_hot(act_ph, n_acts)
    log_probs = tf.reduce_sum(action_masks * tf.nn.log_softmax(logits), axis=1)
    loss = -tf.reduce_mean(weights_ph * log_probs)

    # make train op
    train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    # for training policy
    def train_one_epoch():
        # make some empty lists for logging.
        batch_obs = []          # for observations
        batch_acts = []         # for actions
        batch_weights = []      # for R(tau) weighting in policy gradient
        batch_rets = []         # for measuring episode returns
        batch_lens = []         # for measuring episode lengths

        # reset episode-specific variables
        env.reset()
        obs = env.observation()       # first obs comes from starting distribution
        done = False            # signal from environment that episode is over
        ep_rews = []            # list for rewards accrued throughout ep

        # collect experience by acting in the environment with current policy
        while True:
            # act in the environment
            if env.sizing_regime=='U':
                batch_obs.append(np.copy(obs))

                act = action_index[sess.run(actions, {obs_ph: obs.reshape(1,-1)})[0]-1]
                obs, rew, done = env.step(act)
                # save action, reward
                batch_acts.append(act)
                ep_rews.append(rew)
            else:
                act = .5
                obs, rew, done = env.step(act)

            # print "action taken: ", act
           
            if done:
                # if episode is over, record info about episode
                ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                batch_rets.append(ep_ret)
                batch_lens.append(ep_len)

                # the weight for each logprob(a|s) is R(tau)
                batch_weights += [ep_ret] * ep_len

                # reset episode-specific variables
                obs, done, ep_rews = env.reset(), False, []
                obs = env.observation()
                # end experience loop if we have enough of it
                if len(batch_obs) > batch_size:
                    break

        # take a single policy gradient update step
        batch_loss, _ = sess.run([loss, train_op],
                                 feed_dict={
                                    obs_ph: np.array(batch_obs),
                                    act_ph: np.array(batch_acts),
                                    weights_ph: np.array(batch_weights)
                                 })
        return batch_loss, batch_rets, batch_lens
    
    def test_run():
        # reset episode-specific variables
        env.reset()
        env.graph = test_graph

        obs = env.observation()       # first obs comes from starting distribution
        done = False            # signal from environment that episode is over

        # collect experience by acting in the environment with current policy
        while not done:
                # act in the environment
                if env.sizing_regime=='U':
                    act = action_index[sess.run(actions, {obs_ph: obs.reshape(1,-1)})[0]-1]
                    # print "took action: ", act, "at self.n = ",env.n
                    obs, rew, done = env.step(act)
                else:
                    act = .5
                    obs, rew, done = env.step(act)

    # training loop
    for i in range(epochs):
        batch_loss, batch_rets, batch_lens = train_one_epoch()
        print('epoch: %3d \t loss: %.3f \t return: %.3f \t ep_len: %.3f'%
                (i, batch_loss, np.mean(batch_rets), np.mean(batch_lens)))
    
    print "performing test run"
    test_run()

if __name__ == '__main__':
    graph = nx.complete_graph(100)

    for i in np.linspace(0,1,10):
        print "testing lamb = ", i
        env = Environment(lamb = i)
        train(env=env,epochs=100,batch_size=20,test_graph = graph)
        plt.figure()
        plt.plot(env.history_n)
        plt.plot(env.history_capacity)
        plt.ylim(0,8000)
        plt.show()

        # en = Environment(graph=env.get_graph())
        # for i in range(50):
        #     step = en.observation()

        #     if not step is None:
        #         if en.capacity==en.n:
        #             en.action(2)
        #         else:
        #             en.action(.5)
        #     else:
        #         pass
        #         # print "finished graph"

        # plt.figure()
        # plt.plot(en.history_n)
        # plt.plot(en.history_capacity)
        # plt.show()


    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--env_name', '--env', type=str, default='CartPole-v0')
    # parser.add_argument('--render', action='store_true')
    # parser.add_argument('--lr', type=float, default=1e-2)
    # args = parser.parse_args()
    # print('\nUsing simplest formulation of policy gradient.\n')
    # train(env_name=args.env_name, lr=args.lr)




