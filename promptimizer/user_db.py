import boto3
import json
from datetime import datetime as dt
import random
import string
# Initialize a session using Amazon DynamoDB
import pandas as pd
from decimal import Decimal

bucket = 'sagemaker-us-east-2-344400919253'

initial_df = pd.DataFrame({'prompt_id': [0], 
                           'use_case': ['messy'],
                           'writer_system': ['My First Saved Prompt'],
                           'writer_user': ['that too'],
                           'task_system': ['task_system'],
                           'separator': ['---'],
                           'label': ['probability'],
                           'evaluator': ['AUC'],
                           'setup_id': ['jbn5'], # can be used to find training data and embeddings if available
                           'save_date': [dt.now()]
                           })


from boto3.dynamodb.conditions import Key


class dynamo_jobs:
    def __init__(self):
        dynamodb = boto3.resource('dynamodb', region_name='us-east-2')
        self.db = dynamodb.Table('Jobs2')
        self.keys = ['email_address', 'setup_id', 'key_path', 'meta_user', 'use_case', 'meta_system']
    def initialize(self, P):

        job_id = self.max_job_id(P)
        X = {'iterations': [], 'job_id': job_id + 1,
             'transaction_timestamp': str(dt.now())}
        for n in self.keys:
            if n in P.keys():
                X[n] = P[n] 
            else:
                print('missing ' + n)
                return 'missing ' + n
        return self.db.put_item(Item = X)

    def max_job_id(self, P):
        J = self.get_jobs({'email_address': P['email_address']})
        if J:
            if len(J['job_id']) < 1:
                print('job_id is empty !')
                return 0
            else:
                return max(J['job_id'])
        else:
            return 0

    def get_jobs(self, P):
        if 'setup_id' in P.keys():
            response = self.db.query(
                    KeyConditionExpression=Key('email_address').eq(P['email_address']) &
                    Key('setup_id').eq(P['setup_id'])
                    )
            return response.get('Items')
        else:
            response = self.db.query(
                    KeyConditionExpression=Key('email_address').eq(P['email_address']) &
                    Key('setup_id').between('0000000000000000', 'zzzzzzzzzzzzzzzz')
                    )

            items = response.get('Items')
            if items:
                X = {}
                keys = self.keys + ['transaction_timestamp', 'iterations', 'job_id']
                for k in keys:
                    X[k] = []
                for item in items:
                    for k in keys:
                        if k in item.keys():
                            X[k].append(item[k])
                return X
            else:
                return None

    def update(self, P):

        X = { 'transaction_timestamp':  str(dt.now())}

        for k in self.keys + ['iterations', 'job_id']:
            if k not in P.keys():
                return 'jobs::update missing ' + k
            X[k] = P[k]
        return self.db.put_item(Item = X)


class dynamo_usage:
    def __init__(self):
        dynamodb = boto3.resource('dynamodb', region_name='us-east-2')
        self.db = dynamodb.Table('Usage')

    def get_usage(self, P):

        if P['email_address'] == '':
            return {}
        else:
            response = self.db.query(
                    KeyConditionExpression=Key('email_address').eq(P['email_address']) & 
                    #Key('transaction_timestamp').between('01-01-1000 10', '01-01-3000 10')
                    Key('transaction_timestamp').between('2020-08-19 15:45:52.810449', '2050-01-19 15:45:52.810449')
                    )
            items = response['Items']
            int_keys = ['delta_tokens', 'previous_tokens', 'current_tokens']
            str_keys = ['transaction_timestamp', 'note']
            ledger ={}
            for k in int_keys + str_keys:
                ledger[k] = []
            for item in items:
                for k in int_keys:
                    ledger[k].append(int(item[k]))
                for k in str_keys:
                    ledger[k].append(item[k])

            return ledger
            #print(P['email_address'])
            #response = self.db.get_item(Key={'email_address': P['email_address']})
            #return response.get('Item')


    def initial(self, P):

        P['prompt_tokens'] = 0
        P['completion_tokens'] = 0
        P['previous_tokens'] = 0
        P['current_tokens'] = P['delta_tokens']
        P['delta_tokens'] = P['delta_tokens']
        P['transaction_timestamp'] = str(dt.now())
        print(P)
        return self.db.put_item(Item = P)

    def update(self, P):

        usage = pd.DataFrame(self.get_usage(P))
        #print(pd.DataFrame(usage))
        #return "OK"
        #return user
        current_tokens = usage['current_tokens'].iloc[-1]
        if usage.shape[0] > 0:
            for n in ['email_address', 'prompt_tokens', 'completion_tokens',
                      'delta_tokens']:
                if n not in P.keys():
                    return 'usage::initial missing ' + n

            entry = {'email_address': P['email_address'],
                     'prompt_tokens': P['prompt_tokens'],
                     'completion_tokens': P['completion_tokens'],
                     'previous_tokens': current_tokens.item(),
                     'delta_tokens': P['delta_tokens'],
                     'current_tokens': Decimal(current_tokens.item() + P['delta_tokens']), 
                     'note': P['note'],
                     'transaction_timestamp': str(dt.now())}
            return self.db.put_item(Item = entry)
        else:
            return "No user found"

    def get_user_ledger(self, P):

        user = self.get_user(P)
        return user



class dynamo_user:
    def __init__(self):
        dynamodb = boto3.resource('dynamodb', region_name='us-east-2')

        self.db = dynamodb.Table('Users')

    def new_user(self, P):

        exists = self.get_user(P)

        if exists:
            return 'Customoer Already Exists'
        else:

            item = {'transaction_time': str(dt.now())}
            for p in ['email_address', 'firstname', 'lastname', 'n_credits',
                      'password']:
                item[p] = P[p]

            response = self.db.put_item(Item=item)
            if response['ResponseMetadata']['HTTPStatusCode'] == 200:

                initial_df.to_csv('s3://' + bucket+'/users/{}/saved_prompts.csv'.format(P['email_address']), index=False)
                ledger = dynamo_usage()
                print(ledger.db)
                entry = {'email_address': P['email_address'],
                         'delta_tokens': 1000000,
                         'note': 'Initial Balance'}
                init = ledger.initial(entry)
                print(init)
                return 'Success'
            else:
                return 'HTTP Failure'

    def everything(self):

        items = []
        response = self.db.scan()
        items.extend(response.get('Items', []))

        while 'LastEvaluatedKey' in response:
            response = table.scan(ExclusiveStartKey=response['LastEvaluatedKey'])
            items.extend(response.get('Items', []))

        print("All items in the table:")
        for item in items:
            print(item)


    def get_user(self, P):
        
        if P['email_address'] == '':
            return None
        else:
            response = self.db.get_item(Key={'email_address': P['email_address']})
            return response.get('Item')


    def delete_user(self, P):
        return self.db.delete_item(Key={'email_address': P['email_address'] })

    def decrement_usage(self):
        c = self.get_customer(P['email_address'])
        print(c)
        n_credits = c['n_credits']
        print(n_credits, n_credits-P['delta'])
        self.update_customer(email_address, {'n_credits': n_credits-P['delta']})


    def increment_usage(self, P):

        u = self.get_user(P['email_address'])

        if n_credits + P['delta'] < 0:
            response = self.db.update_item(
                    Key={'customer_id': P['email_address']},  
                    UpdateExpression="SET n_credits = 0",
                    ReturnValues="UPDATED_NEW"
                    )

            return 'This will drop user to less than zero credits'
            print (response['Attributes'])
        else:
            response = self.db.update_item(
                    Key={'customer_id': P['email_address']},  # Specify the primary key
                    UpdateExpression="SET n_credits = n_credits + :inc",
                    ExpressionAttributeValues = {':inc': P['delta']},
                    ReturnValues="UPDATED_NEW"
                    )
            print("Update succeeded:", response['Attributes'])
            return 'Success'


 


    def update_customer(self, customer_id, updates):

        update_string = []
        Xpression = {}

        for u in updates.keys():
            update_string.append(u + ' = ' + self.x_map[u])
            Xpression[self.x_map[u]] = updates[u]

        print(','.join(update_string))
        response = self.db.update_item(
                Key={'customer_id': customer_id},  # Specify the primary key
                #UpdateExpression="SET subscription_type = :stype, n_credits = n_credits + :inc",
                UpdateExpression='SET + update_string',
                ExpressionAttributeValues=Xpression,
                #{
                #    ':stype': 'Potato',
                #    ':inc': 13
                #    },
                ReturnValues="UPDATED_NEW"
                )
        print("Update succeeded:", response['Attributes'])

#if __name__ == '__main__':
#    db = customer_db()
#    new_customer = db.new_customer('Liza',  'Cher', 'Plastic', 100, 'imma star')
    #print(new_customer)
    #print(db.get_customer(new_customer))
    #db.delete_customer('OYFWGZYO')
    #db.increment_usage('UEmlWPbX', -10)
    #db.everything()
#    db.update_customer('CUST12345', {'credits': 10})
#    print(db.get_customer('CUST12345'))





