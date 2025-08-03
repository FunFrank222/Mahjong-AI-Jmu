from feature import FeatureAgent
import numpy as np
import json

obs = [[] for i in range(4)]
actions = [[] for i in range(4)]
matchid = -1

l = []

def saveData():
    # global actions,obs
    # assert [len(x) for x in obs] == [len(x) for x in actions], 'obs actions not matching!'
    # index = [[] for i in range(4)]
    # actions_new = [[] for i in range(4)]
    # obs_new = [[] for i in range(4)]
    # for i in range(4):
    #     index[i] = [i for i, x in enumerate(actions[i]) if x == 0]
    #
    # for i in range(4):
    #     for j in range(len(actions[i])):
    #         if j in index[i]:
    #             continue
    #         else:
    #             actions_new[i].append(actions[i][j])
    #             obs_new[i].append(obs[i][j])
    # actions = actions_new
    # obs = obs_new

    global actions,obs
    actions_new = [[]for i in range(4)]
    obs_new = [[]for i in range(4)]
    for i in range(4):
        for j,o in enumerate(obs[i]):
            if o['action_mask'].sum() != 1:
                obs_new[i].append(o)
                actions_new[i].append(actions[i][j])
    obs = obs_new
    actions = actions_new



    l.append(sum([len(x) for x in obs]))
    np.savez('data/%d.npz'%matchid
        , obs = np.stack([x['observation'] for i in range(4) for x in obs[i]]).astype(np.int8)
        , mask = np.stack([x['action_mask'] for i in range(4) for x in obs[i]]).astype(np.int8)
        , act = np.array([x for i in range(4) for x in actions[i]])
    )
    for x in obs: x.clear()
    for x in actions: x.clear()

with open('data/data.txt', encoding='UTF-8') as f:
    line = f.readline()
    while line:
        t = line.split()
        if len(t) == 0:
            line = f.readline()
            continue
        if t[0] == 'Match':
            agents = [FeatureAgent(i) for i in range(4)]
            matchid += 1
            print('Processing match %d %s...' % (matchid, t[1]))
        elif t[0] == 'Wind':
            for agent in agents:
                agent.request2obs(line)
        elif t[0] == 'Player':
            p = int(t[1])
            if t[2] == 'Deal':
                agents[p].request2obs(' '.join(t[2:]))
            elif t[2] == 'Draw':
                for i in range(4):
                    if i == p:
                        obs[p].append(agents[p].request2obs(' '.join(t[2:])))
                        actions[p].append(0)
                    else:
                        agents[i].request2obs(' '.join(t[:3]))
            elif t[2] == 'Play':
                actions[p].pop()
                actions[p].append(agents[p].response2action(' '.join(t[2:])))
                for i in range(4):
                    if i == p:
                        agents[p].request2obs(line)
                    else:
                        obs[i].append(agents[i].request2obs(line))
                        actions[i].append(0)
                curTile = t[3]
            elif t[2] == 'Chi':
                actions[p].pop()
                actions[p].append(agents[p].response2action('Chi %s %s' % (curTile, t[3])))
                for i in range(4):
                    if i == p:
                        obs[p].append(agents[p].request2obs('Player %d Chi %s' % (p, t[3])))
                        actions[p].append(0)
                    else:
                        agents[i].request2obs('Player %d Chi %s' % (p, t[3]))
            elif t[2] == 'Peng':
                actions[p].pop()
                actions[p].append(agents[p].response2action('Peng %s' % t[3]))
                for i in range(4):
                    if i == p:
                        obs[p].append(agents[p].request2obs('Player %d Peng %s' % (p, t[3])))
                        actions[p].append(0)
                    else:
                        agents[i].request2obs('Player %d Peng %s' % (p, t[3]))
            elif t[2] == 'Gang':
                actions[p].pop()
                actions[p].append(agents[p].response2action('Gang %s' % t[3]))
                for i in range(4):
                    agents[i].request2obs('Player %d Gang %s' % (p, t[3]))
            elif t[2] == 'AnGang':
                actions[p].pop()
                actions[p].append(agents[p].response2action('AnGang %s' % t[3]))
                for i in range(4):
                    if i == p:
                        agents[p].request2obs('Player %d AnGang %s' % (p, t[3]))
                    else:
                        agents[i].request2obs('Player %d AnGang' % p)
            elif t[2] == 'BuGang':
                actions[p].pop()
                actions[p].append(agents[p].response2action('BuGang %s' % t[3]))
                for i in range(4):
                    if i == p:
                        agents[p].request2obs('Player %d BuGang %s' % (p, t[3]))
                    else:
                        obs[i].append(agents[i].request2obs('Player %d BuGang %s' % (p, t[3])))
                        actions[i].append(0)
            elif t[2] == 'Hu':
                actions[p].pop()
                actions[p].append(agents[p].response2action('Hu'))
            # Deal with Ignore clause
            if t[2] in ['Peng', 'Gang', 'Hu']:
                for k in range(5, 15, 5):
                    if len(t) > k:
                        p = int(t[k + 1])
                        if t[k + 2] == 'Chi':
                            actions[p].pop()
                            actions[p].append(agents[p].response2action('Chi %s %s' % (curTile, t[k + 3])))
                        elif t[k + 2] == 'Peng':
                            actions[p].pop()
                            actions[p].append(agents[p].response2action('Peng %s' % t[k + 3]))
                        elif t[k + 2] == 'Gang':
                            actions[p].pop()
                            actions[p].append(agents[p].response2action('Gang %s' % t[k + 3]))
                        elif t[k + 2] == 'Hu':
                            actions[p].pop()
                            actions[p].append(agents[p].response2action('Hu'))
                    else: break
        elif t[0] == 'Score':
            saveData()
        line = f.readline()
with open('data/count.json', 'w') as f:
    json.dump(l, f)

