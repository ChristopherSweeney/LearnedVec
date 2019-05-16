import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from tensorflow.python.ops import rnn, rnn_cell
from rl import Environment
import util
from learned_standard_vector import LearnedDynamicArray

def lstm(x):
    n_classes = 10
    rnn_size = 4
    layer = {'weights':tf.Variable(tf.random_normal([rnn_size,n_classes])),
             'biases':tf.Variable(tf.random_normal([n_classes]))}

    x = tf.expand_dims(x, 2)
    lstm_cell = rnn_cell.BasicLSTMCell(rnn_size,state_is_tuple=True)
    outputs, states = rnn.static_rnn(lstm_cell,  tf.unstack(tf.transpose(x, perm=[1, 0, 2])), dtype=tf.float32)
    output = tf.matmul(outputs[-1],layer['weights']) + layer['biases']
    return output

def mlp(x, sizes, activation=tf.tanh, output_activation=None):
    # Build a feedforward neural network.
    for size in sizes[:-1]:
        x = tf.layers.dense(x, units=size, activation=activation)
    return tf.layers.dense(x, units=sizes[-1], activation=output_activation)

def train(env, hidden_sizes=[32,64,32], lr=1e-3, 
          epochs=50, batch_size=5000,test_graph=None, gamma=2):

    # make environment, check spaces, get obs / act dims
    # Note, this only works for continuous observation space, discrete action space
    tf.reset_default_graph()

    obs_dim = env.observation_space.shape[0]
    n_acts = env.action_space_size
    relative = 2
    upsize_action_index = np.linspace(1.1,2.,n_acts)
    downsize_action_index = np.linspace(1/gamma,1/gamma*relative,n_acts)
    print "upsize action space is [{l},{u}]".format(l= gamma, u = gamma*relative)
    print "downsize action space is [{l},{u}]".format(l= 1./gamma, u = 1./gamma*relative)

    # make core of policy network
    obs_ph = tf.placeholder(shape=(None, obs_dim), dtype=tf.float32)
    rep = mlp(obs_ph, sizes=hidden_sizes)
    logits = tf.layers.dense(rep, units = n_acts, activation=None)
    # downsize_logits = tf.layers.dense(rep, units = n_acts, activation=None)
    logits = lstm(obs_ph)

    # make action selection op (outputs int actions, sampled from policy)
    actions = tf.squeeze(tf.multinomial(logits=logits,num_samples=1), axis=1)
    # downsize_actions = tf.squeeze(tf.multinomial(logits=downsize_logits,num_samples=1), axis=1)


    # make loss function whose gradient, for the right data, is policy gradient
    weights_ph = tf.placeholder(shape=(None,), dtype=tf.float32)
    act_ph = tf.placeholder(shape=(None,), dtype=tf.int32)
    action_masks = tf.one_hot(act_ph, n_acts)

    log_probs = tf.reduce_sum(action_masks * tf.nn.log_softmax(logits), axis=1)
    # downsize_log_probs = tf.reduce_sum(action_masks * tf.nn.log_softmax(logits), axis=1)

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

        env.reset()
        obs = env.observation()       # first obs comes from starting distribution
        done = False            # signal from environment that episode is over
        ep_rews = []            # list for rewards accrued throughout ep

        # collect experience by acting in the environment with current policy
        while True:
            # act in the environment
            try:
                if env.sizing_regime=='U':
                    batch_obs.append(np.copy(obs))

                    act = upsize_action_index[sess.run(actions, {obs_ph: obs.reshape(1,-1)})[0]-1]
                    obs, rew, done = env.step(act)
                    # print "resize up"
                    # save action, reward
                    batch_acts.append(act)
                    ep_rews.append(rew)
                else:
                    act = .5
                    obs, rew, done = env.step(act)
                    # print "resize down"
            except:
                done = True
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
                    act = upsize_action_index[sess.run(actions, {obs_ph: obs.reshape(1,-1)})[0]-1]
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
    # graph = nx.complete_graph(100)
    graph =nx.complete_graph(110)
    wasted_mem=[]
    operations = []
    lambdas = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    for i in lambdas:
        print "testing lamb = ", i
        env = Environment(lamb = i,buckets =20)
        train(env=env,epochs=50,batch_size=60,test_graph = graph)
        # plt.figure()
        # plt.plot(env.history_n)
        # plt.plot(env.history_capacity)
        # # plt.ylim(0,8000)
        # plt.show()
        wasted_mem.append(np.sum(np.array(env.history_capacity)-np.array(env.history_n)))
        operations.append(env.operations)

    fig, ax = plt.subplots()
    ax.scatter(wasted_mem,operations)
    for i, txt in enumerate(lambdas):
        ax.annotate(txt, (wasted_mem[i], operations[i]))

    ax.set_xlabel("wasted memory")
    ax.set_ylabel("operations")
    ax.set_xlim(0, 1e8)
    ax.set_ylim(0, 1000000)

    en = Environment()
    en.reset()
    en.graph = graph
    for i in range(60):
        step = en.observation()
        if not step is None:
            if en.capacity==en.n:
                en.action(2)
            else:
                en.action(.5)
        else:
            print "finished graph"
    # ax.scatter(np.sum(np.array(en.history_capacity)-np.array(en.history_n)),en.operations)
    
    gammas = [1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.]
    wasted_mem1 = []
    operations1 = []
    for gamma in gammas:
        regular_stack = LearnedDynamicArray(default_resize_up = gamma,default_resize_down=1./gamma,default_downsize_point=1./(gamma**2)) #with no model, this is just a regular dynamic array
    
        answer = util.dfs_iterative(graph, 1,regular_stack)
   
        reg_wasted_memory = np.array(regular_stack.history_capacity)- np.array(regular_stack.history_n)
        operations1.append(regular_stack.operations)
        wasted_mem1.append(np.sum(reg_wasted_memory))

    ax.scatter(wasted_mem1,operations1,color="y")
    for i, txt in enumerate(gammas):
        ax.annotate(txt, (wasted_mem1[i], operations1[i]))


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

    plt.figure()
    plt.plot(en.history_n)
    plt.plot(en.history_capacity)
    plt.show()


    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--env_name', '--env', type=str, default='CartPole-v0')
    # parser.add_argument('--render', action='store_true')
    # parser.add_argument('--lr', type=float, default=1e-2)
    # args = parser.parse_args()
    # print('\nUsing simplest formulation of policy gradient.\n')
    # train(env_name=args.env_name, lr=args.lr)




