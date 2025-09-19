import random
import requests
import json
import openai
from sklearn.metrics import roc_auc_score
import os
import re
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, Matern, DotProduct
from scipy.stats import ecdf, lognorm
#from multiprocessing import Pool
from scipy.stats import norm
import pandas as pd
from promptimizer import ops, user_db, css, webpages 
import boto3
import numpy as np
import datetime
import pickle
import more_itertools

def write_log(message):
    print(message)
    with open("/tmp/bayes.log", 'a') as f:
        f.write('[' + str(datetime.datetime.today()) +'] ' + message + "\n")


def quantify_relevance(word):
    if word == 'critical relevance':
        return .9
    elif word == 'very relevant':
        return .7
    elif word == 'not very relevant':
        return .3
    elif word == 'completely irrelevant':
        return .1
    else:
        return .5

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


def score_prompts(use_case, filename_id, par):

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
    ct = 0
    print('score_prompt (filename_ids)', par['filename_ids'])
    for i, filename in enumerate(par['filename_ids'].split(';')):

        for raw in azure_client.files.content(filename).text.strip().split("\n"):
            ct += 1
            if True:
                jsponse = json.loads(raw)
                custom_ids_components = jsponse['custom_id'].split('_')
                match = re.search(r'(\{.*\})', jsponse['response']['body']['choices'][0]['message']['content'], re.DOTALL)

                if match:
                    try:
                        content = json.loads(match.group(0))
                        if par['label'] in content.keys():
                            prediction = content[par['label']]
                            #if ~np.isnan(prediction):
                            if use_case in ['rag', 'search']:
                                prompt_ids.append(custom_ids_components[3])
                            else:
                                prompt_ids.append(custom_ids_components[1])

                            record_ids.append(custom_ids_components[2])
                            predictions.append(prediction)
                           
                            usage = jsponse['response']['body']['usage']
                            completion_tokens.append(usage['completion_tokens'])
                            prompt_tokens.append(usage['prompt_tokens'])
                            total_tokens.append(usage['total_tokens'])


                    except Exception as e:
                        print(jsponse['response']['body']['choices'][0])
                        print(custom_ids_components)
                        print("failed response will be ignored.")
                        print(e)
                else:
                    print('no match', jsponse['response']['body']['choices'][0]['message']['content'])

    predictions_df = pd.DataFrame({'prompt_id': prompt_ids,
                                   'record_id': record_ids,
                                   'prediction': predictions,
                                   'usage': total_tokens})
    predictions_df.to_csv('s3://' + ops.bucket + '/' +par['key_path'] + '/predictions/' + par['setup_id'] + '.csv', index=False)
    if use_case not in ['rag', 'search']:
        training_df = pd.read_csv('s3://' + ops.bucket + '/' + par['key_path'] + '/training_data/' + par['setup_id'])
        truth = training_df['output']

    azure_client.close()
    if par['evaluator'].lower() == 'accuracy':
        return accuracy(predictions_df, truth)
    elif par['evaluator'].lower() == 'auc':
        return auc(predictions_df, truth)
    elif par['evaluator'].lower() == 'prompt':
        return llm_evaluation(use_case, predictions_df, '/'.join(['s3:/', ops.bucket, par['key_path'], 'output', 
                                                                  par['setup_id'], 'demonstrations.csv']), par['task_system'])
    else:
        write_log("score_prompts: ERROR No evaluator")
        return "ERROR", "No Evaluator"

def auc(predictions_df, truth):

    performance_report = "<table><tr><td>ID</td><td>Score</td><td>Token Usage</td></tr>"
    prompt_auc = {}
    tokens = {}
    predict_proba = []
    target = []
    for prompt_id in predictions_df['prompt_id'].unique():
        df = predictions_df[predictions_df['prompt_id'] == prompt_id]
        prompt_auc[prompt_id] = roc_auc_score([1 if truth[int(x)] == True else 0 for x in df['record_id']],
                                              [probability(x.lower()) for x in df['prediction']])
        tokens[prompt_id] = df['usage'].sum()
    for k in prompt_auc.keys():
        performance_report += webpages.threerows.format(k, prompt_auc[k], tokens[k])
    return prompt_auc, performance_report +'</table>'

def llm_evaluation(use_case, relevance_df, rag_path, question):

    relevance_score = {}
    relevance_scores = []

    for r, i in zip(relevance_df['prediction'], relevance_df['prompt_id']):
        relevance = quantify_relevance(r)
        relevance_score[i] = relevance
        relevance_scores.append(relevance)
    relevance_df['relevance'] = relevance_scores

    snippets = pd.read_csv(rag_path)

    if use_case == 'rag':
        evaluator_prompt = "I'm going to ask you a question. Before I give you the question, I am going to also give you some reference material. The reference material is based on a search of a corpus. It may or may not be relevant. It may help you answer the question."



        few_shot = [snippets.iloc[int(x)]['passage'] for x in relevance_df.sort_values('relevance', ascending=False).iloc[:4]['prompt_id']]
        references = "\n------\n".join(few_shot)
        messages = [{"role": "system", "content": "Be helpful.",
                     "role": "user", "content": "\n".join([evaluator_prompt, "\n", "### QUESTION ###", question, "\n", '### REFERENCES ###',
                                                           "\n", references ])}]
        azure_client = openai.AzureOpenAI(
                api_key=os.environ['AZURE_OPENAI_KEY'],
                api_version="2024-10-21",
                azure_endpoint = os.environ["AZURE_ENDPOINT"]
                )
        response = azure_client.chat.completions.create(
                model='gpt-4o',
                messages = messages
                )
        azure_client.close()
        report = '<b>Question</b>'+question+'<br><b>Answer to question using this sample</b><br>' +  response.choices[0].message.content
    elif use_case == 'search':
        report =  '<b>Query </b>'+question

    return relevance_score, report


def accuracy(predictions_df, truth):
    performance_report = "<table><tr><td><b>Prompt ID</b></td><td><b>Score</b></td><td>Token Usage</td></tr>\n"
    total_collect_scores = {}
    tokens = {}
    #results = [json.loads(model)['modelOutput']['output']['message']['content'][0]['text']
    prompt_accuracy = {}
    test_size = {}
    tokens = {}
    for p in predictions_df['prompt_id'].unique():
        test_size[p] = predictions_df[predictions_df['prompt_id'] == p].shape[0]
        prompt_accuracy[p] = 0
        tokens[p] = predictions_df[predictions_df['prompt_id'] == p]['usage'].sum()

    for prompt_id, record_id, prediction in zip(predictions_df['prompt_id'], predictions_df['record_id'], predictions_df['prediction']):
        if prediction.lower() == truth[int(record_id)]:
            prompt_accuracy[prompt_id] += 1
    #    else:
    #        print(prediction.lower(), ' : ', record_id, ' : ', truth[str(record_id)])

    for prompt_id in prompt_accuracy.keys():
        performance_report += webpages.threerows.format(prompt_id,
                                               round(prompt_accuracy[prompt_id]/test_size[prompt_id],4),
                                               tokens[prompt_id])

    return prompt_accuracy, performance_report + '</table>'

#def generate_preview(key_path, setup_id):


def optimize(use_case, prompt_ids, parameters, performance_report = ()):

    s3 = boto3.client('s3', aws_access_key_id=os.environ['AWS_ACCESS_KEY'],
                       aws_secret_access_key=os.environ['AWS_SECRET_KEY'], region_name='us-east-2')
    if use_case in ['rag', 'search']:
        df = pd.read_csv('/'.join(['s3:/', ops.bucket, parameters['key_path'], 'output',  parameters['setup_id'], 'demonstrations.csv']))
    else:
        df = pd.read_csv('s3://' + ops.bucket + '/' + parameters['key_path'] + '/training_data/' + parameters['setup_id'])
        obj = s3.get_object(Bucket=ops.bucket, Key=parameters['key_path'] + '/output/'+ parameters['setup_id'] + '/consolidated.csv')
        prompts = obj['Body'].read().decode('utf-8').split("|")


        if ('input' in df.columns) & ('output' in df.columns):
            preview_data = '<table border=0><tr><td></td><td><b>Data Preivew</b></td><td></td></tr>'
            for x in range(min(3, df.shape[0])):
                preview_data += "<tr>\n    <td>"+str(x+1)+"</td>\n   <td>" + df['input'].iloc[x] + "</td>\n"
                preview_data += "    <td>" + str(df['output'].iloc[x]) + "</td>\n</tr>\n"
            preview_data += "</table>"

        else:
            return "Your file must contain columns with the names 'input' and 'output'."

    n_training_examples = df.shape[0]

    print('writing {} new files.'.format(len(prompt_ids)))
    write_log('optimize (key_path): ' + parameters['key_path'])
    write_log('optimize (setup_id): ' + parameters['setup_id'])

    #if parameters['examples'] != '':
    #    examples = pd.read_csv('s3://' + ops.bucket + '/' + parameters['key_path'] + '/output/'+ parameters['setup_id'] + '/examples.csv')

    query = {'method': 'POST',
             'url': '/chat/completions',
             'body': {
                'model': 'gpt-4o-batch',
                'temperature': .03,
                }
             }
    evaluation_jsonl = []
    for h, idx in enumerate(prompt_ids):
        if use_case == 'rag':
            job = user_db.dynamo_jobs().get_job(parameters)
            query['custom_id'] = 'RAG_DOCUMENT_ID_{}'.format(idx)
            query['body']['messages'] = [{'role': 'system', 'content': job['meta_system']},
                                         {'role': 'user', 'content': "\n".join([job['meta_user'], "\n",
                                                                                "### QUESTION ###",
                                                                                parameters['task_system'],"\n",
                                                                                "### REFERENCES ###",
                                                                                df['passage'].iloc[idx]])}
                                         ]
        elif use_case == 'search':
            job = user_db.dynamo_jobs().get_job(parameters)
            samples = [{"type": "image_url","image_url": { "url": 'http://images.cocodataset.org/' + idx }}]
            query['custom_id'] = 'JOB_' + str(h) + '_RECORD_' + idx
            query['body']['messages'] = [{'role': 'system', 'content': job['meta_system']},
                                         {'role': 'user', 'content': [{"type": "text", "text": job['meta_user'] + "\n\n### QUERY ###\n" + parameters['task_system'] + "\n"}] + samples}
                                         ]
        else:
            prompt = prompts[idx]
            for i, text in enumerate(df['input']):
                query['custom_id'] = 'PROMPT_{}_{}'.format(idx, i)
                query['body']['messages'] = [{'role': 'system', 'content': parameters['task_system']},
                                             {'role': 'user', 'content': prompt + "\n" + parameters['separator']+"\n" + text}
                                             ]
        evaluation_jsonl.append(json.dumps(query))

    batch_response_id, azure_file_id = ops.azure_batch([evaluation_jsonl])
    write_log('optimize (azure_file_id): ' + str(azure_file_id))
    azure_file_ids = parameters['azure_file_id'] + ';' + azure_file_id[0]

    jdb = user_db.dynamo_jobs()
    history = jdb.get_job({'email_address': parameters['email_address'],
                           'setup_id': parameters['setup_id']})

    write_log('optimize (dynamo_jobs().get_jobs): ' + str(history))
    write_log('optimize (batch_response_id): ' + batch_response_id[0])
    history['iterations'].append(batch_response_id[0])
    print('optimize history')
    print(history)
    print(jdb.update(history))

    sidebar = "<table>" + webpages.tworows.format("Evaluator", parameters['evaluator'])+\
            webpages.tworows.format("Use Case", use_case)+\
            webpages.tworows.format("N Rows", n_training_examples)+ "</table>"

    hidden_variables = webpages.hidden.format('azure_job_id', batch_response_id[0])+\
            webpages.hidden.format('azure_file_id', azure_file_ids)+\
            webpages.hidden.format('jobArn', '')

    for k in ['setup_id', 'key_path', 'setup_id', 'evaluator', 'label',
              'n_batches', 'batch_size', 'separator', 'task_system',
              'filename_ids', 'email_address', 'examples']:
        if k in parameters.keys():
            hidden_variables += webpages.hidden.format(k, parameters[k])
        else:
            print(k, 'not defined in optimize')

    if len(performance_report) > 0:
        history, best_prompt, stats = performance_report
        preview_data = ''
    else:
        history = ''
        best_prompt = ''
        stats = ''

    return webpages.check_status_form.format(css.style, webpages.navbar, sidebar + stats, use_case,
                                             'iterate', preview_data+hidden_variables+history, best_prompt)


img = "<td><a href=\"http://images.cocodataset.org/{}\"><img src=\"http://images.cocodataset.org/{}\" height=100></img></a></td>\n"

def bayes_pipeline(use_case, filename_id, par, stats):

    X = {}
    for x in par.keys():
        X[x] = par[x]

    if 'filename_ids' in par.keys():
        X['filename_ids'] = X['filename_ids'] + ';' + filename_id
    else:
        X['filename_ids'] = filename_id

    scores_by_prompt, performance_report = score_prompts(use_case, filename_id, X)

    usage = user_db.dynamo_usage().get_usage({'email_address': X['email_address']})
    stats += webpages.tworows.format('Balance', usage['current_tokens'][-1]) + '</table>'

    s3 = boto3.client('s3', aws_access_key_id=os.environ['AWS_ACCESS_KEY'],
                       aws_secret_access_key=os.environ['AWS_SECRET_KEY'], region_name='us-east-2')


    unscored_embeddings = []
    unscored_embeddings_id_map = {}


    if use_case == 'rag':
        corpus = pd.read_csv('/'.join(['s3:/', ops.bucket, par['key_path'], 'output', par['setup_id'], 'demonstrations.csv']))
        embeddings = pd.read_csv('/'.join(['s3:/', ops.bucket, par['key_path'], 'embeddings', par['setup_id'] +'.mbd']))
        embeddings_raw = [[float(x) for x in e.split(',')] for e in embeddings['embedding']]
        scored_embeddings = [embeddings_raw[int(r)]  for r in scores_by_prompt.keys()]
        ct = 0
        for x in range(len(embeddings_raw)):
            if str(x) not in scores_by_prompt.keys():
                unscored_embeddings.append(embeddings_raw[x])
                unscored_embeddings_id_map[ct] = x
                ct += 1
    elif use_case == 'search':
        obj = s3.get_object(Bucket=ops.bucket, Key='kaggle/coco2012/consolidated_embeddings.json')
   
        E = pickle.loads(obj['Body'].read())
        
        ct = 0
        print(len(E), ' embeddings')
        for e in E.keys():
            if e not in scores_by_prompt.keys():
                unscored_embeddings.append(E[e])
                unscored_embeddings_id_map[ct] = e
                ct += 1
        scored_embeddings = [E[k] for k in scores_by_prompt.keys()]


    else:
        obj = s3.get_object(Bucket=ops.bucket, Key=par['key_path']+'/output/'+par['setup_id'] + '/consolidated.csv')
        prompts = obj['Body'].read().decode('utf-8').split("|")
        obj = s3.get_object(Bucket=ops.bucket, Key=par['key_path'] + '/embeddings/' + par['setup_id'] + '.mbd')
        embeddings_raw = [[float(x) for x in e.split(',')] for e in obj['Body'].read().decode('utf-8').split("\n")]
        scored_embeddings = [embeddings_raw[int(r)]  for r in scores_by_prompt.keys()]

        ct = 0
        for x in range(len(embeddings_raw)):
            if str(x) not in scores_by_prompt.keys():
                unscored_embeddings.append(embeddings_raw[x])
                unscored_embeddings_id_map[ct] = x
                ct += 1

    Q = [scores_by_prompt[k] for k in scores_by_prompt.keys()]
    print('scores_by_prompt', scores_by_prompt)

    best = -1000
    s = [-1]
    for s in scores_by_prompt.keys():
        if scores_by_prompt[s] > best:
            best = scores_by_prompt[s]
            best_prompt_id = s


    gpr = GaussianProcessRegressor(kernel = Matern() + WhiteKernel())
    scores_ecdf = ecdf(Q)

    transformed_scores = np.log(lognorm.ppf(scores_ecdf.cdf.evaluate(Q) * .999 + .0005, 1))
    gpr.fit(scored_embeddings, transformed_scores)
    mu, sigma = gpr.predict(unscored_embeddings, return_cov=True)

    batch_idx, batch_mu, batch_sigma = create_batches(gpr, unscored_embeddings, int(par['n_batches']), int(par['batch_size']))
    try:
        best_idx = get_best_batch(batch_mu, batch_sigma, par['batch_size'])
    except Exception as e:
        print(e)
        print('might have the wrong evaluation function', par['evaluator'])
        best_idx = random.sample(range(len(batch_idx)), 1)[0]
    performance_report += "</table>"

    if use_case == 'rag':
        print(corpus['passage'].iloc[int(best_prompt_id)])
        best_prompt = " &nbsp; <i>Best Passage So Far:</i> <hr>{}<hr>\nRaw Score: {}\n".format(corpus['passage'].iloc[int(best_prompt_id)], max(Q))
    elif use_case == 'search':
        predictions = pd.read_csv('/'.join(['s3:/', ops.bucket, par['key_path'], 'predictions', par['setup_id'] + '.csv']))
        predictions['score'] = [quantify_relevance(r) for r in predictions['prediction']]
        best_prompt = '<table>'
        for images in more_itertools.batched(predictions.sort_values('score', ascending=False).iloc[:4]['prompt_id'], 2):
            best_prompt += '<tr>' + img.format(images[0], images[0]) + img.format(images[1], images[1])+ '</tr>'
        best_prompt += '</table>'


    else:
        print(prompts[int(best_prompt_id)])
        best_prompt = " &nbsp; <i>Best Prompt So Far:</i> <hr>{}<hr>\nRaw Score: {}\n".format(prompts[int(best_prompt_id)], max(Q))
    if type(best_idx) != int:
        print('ERROR: best_idx not an integer')
        best_idx = random.sample(range(len(batch_idx)), 1)[0]
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
    try:
        url = 'https://boaz.onrender.com/qei?y_best=.02&n=' + str(n)
        data = {'k': ';'.join(batch_mu),
                'sigma': '|'.join(batch_sigma)}
        response = requests.post(url, json.dumps(data))
        boaz = eval(response.content.decode('utf-8'))
    except Exception as e:
        print('Bayesian Issues:', e)
        print(batch_sigma)
        return random.randint(0, len(batch_mu))
    fboaz = [float(x) for x in boaz['scores'].split(',')]
    best = -1
    for i, mx in enumerate(fboaz):
        if mx > best:
            best = float(mx)
            best_idx = i
    return best_idx
