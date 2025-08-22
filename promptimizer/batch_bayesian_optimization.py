import random
import requests
import json
from sklearn.metrics import roc_auc_score

def probability(word):
    if word == 'very likely':
        return .9
    elif word == 'likely':
        return .7
    elif word == 'unlikely':
        return .3
    elif word == 'very unlikely':
        return .1
    else:
        return .5


def score_prompts(filename_id, par):

    azure_client = openai.AzureOpenAI(
            api_key=os.environ['AZURE_OPENAI_KEY'],
            api_version="2024-10-21",
            azure_endpoint = os.environ["AZURE_ENDPOINT"]
            )

    predictions = []
    prompt_ids = []
    record_ids = []
    completion_tokens = []
    total_tokens = []
    prompt_tokens = []

    for i, filename in enumerate(par['filename_ids'].split(';')):
        print(filename)

        for raw in azure_client.files.content(filename).text.strip().split("\n"):
            if True:
                jsponse = json.loads(raw)
                custom_ids_components = jsponse['custom_id'].split('_')
                match = re.search(r'(\{.*\})', jsponse['response']['body']['choices'][0]['message']['content'], re.DOTALL)

                if match:
                    try:
                        content = json.loads(match.group(0))
                        if request.form['label'] in content.keys():
                            prediction = content[request.form['label']]
                            #if ~np.isnan(prediction):
                            prompt_ids.append(custom_ids_components[1])
                            record_ids.append(custom_ids_components[2])
                            predictions.append(prediction)


                            usage = jsponse['response']['body']['usage']
                            completion_tokens.append(usage['completion_tokens'])
                            prompt_tokens.append(usage['prompt_tokens'])
                            total_tokens.append(usage['total_tokens'])


                    except Exception as e:
                        print(custom_ids_components)
                        print("failed response will be ignored.")
                        print(e)


    predictions_df = pd.DataFrame({'prompt_id': prompt_ids,
                                   'record_id': record_ids,
                                   'prediction': predictions,
                                   'usage': total_tokens})
    predictions_df.to_csv('s3://' + bucket + '/' + par['setup_id'] + '/predictions/', index=False)
    print('get training data', par['key_path'], par['setup_id'])
    training_df = pd.read_csv('s3://' + bucket + '/' + par['key_path'] + '/training_data/' + par['setup_id'])
    truth = training_df['output']




def auc(predictions_df, truth):

    performance_report = "<table><tr><td>ID</td><td>Score</td><td>Token Usage</td></tr>"
    prompt_auc = {}
    tokens = {}
    predict_proba = []
    target = []
    print(predictions_df['prompt_id'].unique())
    for prompt_id in predictions_df['prompt_id'].unique():
        df = predictions_df[predictions_df['prompt_id'] == prompt_id]
        prompt_auc[prompt_id] = roc_auc_score([1 if truth[int(x)] == True else 0 for x in df['record_id']],
                                              [probability(x.lower()) for x in df['prediction']])
        tokens[prompt_id] = df['usage'].sum()
    print(prompt_auc)
    print(':::::::::::')
    for k in prompt_auc.keys():
        performance_report += threerows.format(k, prompt_auc[k], tokens[k])
    print(performance_report)
    return prompt_auc, performance_report +'</table>'




def accuracy(predictions_df, truth):
    performance_report = "<table><tr><td><b>Prompt ID</b></td><td><b>Score</b></td><td>Token Usage</td></tr>\n"
    total_collect_scores = {}
    tokens = {}
    #results = [json.loads(model)['modelOutput']['output']['message']['content'][0]['text']
    prompt_accuracy = {}
    test_size = {}
    for p in predictions_df['prompt_id'].unique():
        test_size[p] = predictions_df[predictions_df['prompt_id'] == p].shape[0]
        prompt_accuracy[p] = 0
        tokens[k] = predictions_df[predictions_df['prompt_id'] == p]['tokens'].sum()

    for prompt_id, record_id, prediction in zip(predictions_df['prompt_id'], predictions_df['record_id'], predictions_df['prediction']):
        if prediction.lower() == truth[int(record_id)]:
            prompt_accuracy[prompt_id] += 1
    #    else:
    #        print(prediction.lower(), ' : ', record_id, ' : ', truth[str(record_id)])

    for prompt_id in prompt_accuracy.keys():
        performance_report += threerows.format(prompt_id,
                                               round(prompt_accuracy[prompt_id]/test_size[prompt_id],4),
                                               tokens[prompt_id])


    return prompt_accuracy, performance_report + '</table>'


def bayes_pipeline(use_case, filename_id, par, stats):

    X = {}
    for x in par.keys():
        X[x] = par[x]

    if 'filename_ids' in par.keys():
        X['filename_ids'] = X['filename_ids'] + ';' + filename_id
    else:
        X['filename_ids'] = filename_id

    scores_by_prompt, performance_report = score_prompts(filename_id, X)


    usage = user_db.dynamo_usage().get_usage({'email_address': request.form['email_address']})
    stats += tworows.format('Balance', usage['current_tokens']) + '</table>'

    s3 = boto3.client('s3', aws_access_key_id=os.environ['AWS_ACCESS_KEY'],
                       aws_secret_access_key=os.environ['AWS_SECRET_KEY'], region_name='us-east-2')

    obj = s3.get_object(Bucket=bucket, Key=par['key_path']+'/output/'+par['setup_id'] + '/consolidated.csv')
    prompts = obj['Body'].read().decode('utf-8').split("|")

    obj = s3.get_object(Bucket=bucket, Key=par['key_path'] + '/embeddings/' + par['setup_id'] + '.mbd')
    embeddings_raw = [[float(x) for x in e.split(',')] for e in obj['Body'].read().decode('utf-8').split("\n")]
    scored_embeddings = [embeddings_raw[int(r)]  for r in scores_by_prompt.keys()]
    unscored_embeddings = []
    unscored_embeddings_id_map = {}

    ct = 0
    for x in range(len(embeddings_raw)):
        if str(x) not in scores_by_prompt.keys():
            unscored_embeddings.append(embeddings_raw[x])
            unscored_embeddings_id_map[ct] = x
            ct += 1

    Q = [scores_by_prompt[k] for k in scores_by_prompt.keys()]

    print(scores_by_prompt)
    best = -1000
    s = [-1]
    for s in scores_by_prompt.keys():
        if scores_by_prompt[s] > best:
            best = scores_by_prompt[s]
            best_prompt_id = s
        print('prompt id:', s, ', score: ', scores_by_prompt[s])

    print(prompts[int(best_prompt_id)])

    gpr = GaussianProcessRegressor(kernel = Matern() + WhiteKernel())
    scores_ecdf = ecdf(Q)

    transformed_scores = np.log(lognorm.ppf(scores_ecdf.cdf.evaluate(Q) * .999 + .0005, 1))
    gpr.fit(scored_embeddings, transformed_scores)
    mu, sigma = gpr.predict(unscored_embeddings, return_cov=True)

    batch_idx, batch_mu, batch_sigma = bbo.create_batches(gpr, unscored_embeddings, int(par['n_batches']), int(par['batch_size']))
    try:
        best_idx = bbo.get_best_batch(batch_mu, batch_sigma, par['batch_size'])
    except Exception as e:
        print(e)
        print('might have the wrong evaluation function', par['evaluator'])
        best_idx = random.sample(range(len(batch_idx)), 1)[0]
    performance_report += "</table>"
    best_prompt = " &nbsp; <i>Best Prompt So Far:</i> <hr>{}<hr>\nRaw Score: {}\n".format(prompts[int(best_prompt_id)], max(Q))
    print(batch_idx[best_idx])
    print([unscored_embeddings_id_map[x] for x in batch_idx[best_idx]])
    return optimize(use_case, [unscored_embeddings_id_map[x] for x in batch_idx[best_idx]], X,
                    (performance_report, best_prompt, stats))











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
