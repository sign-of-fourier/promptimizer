import random
import requests
import json

def create_batches(gpr, rollout_embeddings, n_batches, batch_size):
    batch_mu = []
    batch_sigma = []

    batches = []
    batch_idx = []
    n_to_choose_from = len(rollout_embeddings)
    for z in range(n_batches):
        batch = []
        for x in range(batch_size):
            rx = random.randint(0, n_to_choose_from-1)
            while rx in batch:
                rx = random.randint(0, n_to_choose_from-1)
            batch.append(rx)
            
        batch_idx.append(batch)
        m, s = gpr.predict([rollout_embeddings[i] for i in batch], return_cov=True)
        batch_mu.append(','.join([str(x) for x in m]))
        sigma = []
        for x in s:
            sigma.append(','.join([str(y) for y in x]))
        batch_sigma.append(';'.join(sigma))
    return batch_idx, batch_mu, batch_sigma

def get_best_batch(batch_mu, batch_sigma, n):

    url = 'https://boaz.onrender.com/qei?y_best=.02&n=' + str(n)
    data = {'k': ';'.join(batch_mu),
            'sigma': '|'.join(batch_sigma)}
    response = requests.post(url, json.dumps(data))
    try:
        boaz = eval(response.content.decode('utf-8'))
    except Exception as e:
        print(e)
        return(e)
    fboaz = [float(x) for x in boaz['scores'].split(',')]
    best = -1
    for i, mx in enumerate(fboaz):
        if mx > best:
            best = float(mx)
            best_idx = i
    return best_idx
