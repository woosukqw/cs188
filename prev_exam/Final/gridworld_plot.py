
import util
import random
import numpy as np
import matplotlib.pyplot as plt
import randomcolor

from gridworld import parseOptions, GridworldEnvironment, runEpisode


def simplified_gridworld(opts):
    """
    Below is from gridworld.py, removing unused code
    """
    ###########################
    # GET THE GRIDWORLD
    ###########################

    import gridworld
    mdpFunction = getattr(gridworld, "get"+opts.grid)
    mdp = mdpFunction()
    mdp.setLivingReward(opts.livingReward)
    mdp.setNoise(opts.noise)
    env = gridworld.GridworldEnvironment(mdp)

    ###########################
    # GET THE AGENT
    ###########################

    import valueIterationAgents, qlearningAgents
    gridWorldEnv = GridworldEnvironment(mdp)
    actionFn = lambda state: mdp.getPossibleActions(state)
    qLearnOpts = {'gamma': opts.discount,
                    'alpha': opts.learningRate,
                    'epsilon': opts.epsilon,
                    'actionFn': actionFn}
    a = qlearningAgents.QLearningAgent(**qLearnOpts)

    ###########################
    # RUN EPISODES
    ###########################

    displayCallback = lambda x: None
    messageCallback = lambda x: None
    pauseCallback = lambda : None
    decisionCallback = a.getAction

    returns = []
    qvs = []
    for episode in range(1, opts.episodes+1):
        returns.append(runEpisode(a, env, opts.discount, decisionCallback, displayCallback, messageCallback, pauseCallback, episode))

        qValues = util.Counter()
        states = mdp.getStates()
        for state in states:
            for action in mdp.getPossibleActions(state):
                qValues[(state, action)] = a.getQValue(state, action)

        qvs.append(list(qValues.values()))

    return qvs, returns



if __name__ == "__main__":
    opts = parseOptions()
    rand_color = randomcolor.RandomColor()

    # 'epsilon' or 'learningRate'
    what = 'learningRate'

    opts.agent = 'q'
    opts.episodes = 1000
    opts.noise = 0

    step = 0.2
    ministep = [0.01, 0.05, 0.1, 0.15, 0.2]
    step_val = 0

    if what == 'learningRate':
        # add case '0.001' for learningRate graphing
        ministep.insert(0, 0.001)

    target_colors = {
        'Mean': 'blue',
        'Variance': 'orange',
    }
    results = {
        'Mean': [],
        'Variance': [],
    }
    steps = []

    print('target = {}'.format(what))
    while step_val <= 1:
        setattr(opts, what, step_val)
        qv, returns = simplified_gridworld(opts)
        steps.append(step_val)

        results['Mean'].append([np.mean(q) for q in qv])
        results['Variance'].append([np.var(q) for q in qv])

        if len(ministep) > 0:
            step_val = ministep.pop(0)
        else:
            step_val += step


    for key, result in results.items():
        colors = rand_color.generate(hue=target_colors[key], count=len(result))
        for idx, item in enumerate(result):
            plt.plot(range(opts.episodes), item, label='Q-Value {} (a={:.3f})'.format(key, steps[idx]), color=colors.pop())

    plt.legend(loc='upper right')
    plt.title('Q-Value Convergence by {}'.format(what))
    plt.xlabel('Episodes')
    plt.show()

