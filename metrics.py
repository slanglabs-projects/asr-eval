from copy import deepcopy
from tqdm.notebook import tqdm
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# to check word-wise tagging accuracy instead of looking at chunks
def convert_gold_response_to_list(x, ignorelist, reprocesslist):
    for entity in x['labeled_entities']:
        if entity in ignorelist:
            continue
        if isinstance(x['labeled_entities'][entity], list):
            x['labeled_entities'][entity] = [i.lower() for i in x['labeled_entities'][entity]]
        elif isinstance(x['labeled_entities'][entity], dict):
            composite_entity = []
            for k, v in x['labeled_entities'][entity].items():
                if isinstance(v, str):
                    composite_entity.append(v.lower())
                else:
                    composite_entity.append(v)
            x['labeled_entities'][entity] = composite_entity
        else:
            x['labeled_entities'][entity] = str(x['labeled_entities'][entity]).lower().split()
    return x


def convert_pred_response_to_list(x):
    for entity in x['entities'][0]:
        if isinstance(x['entities'][0][entity], str):
            x['entities'][0][entity] = x['entities'][0][entity].lower().split()
        elif isinstance(x['entities'][0][entity], list):
            if len(x['entities'][0][entity]) >= 1:
                if isinstance(x['entities'][0][entity][0], dict):
                    continue
            x['entities'][0][entity] = [
                i.lower() for i in x['entities'][0][entity]
            ]

            temp_list = []
            for i in x['entities'][0][entity]:
                temp_list.extend(i.split())
            x['entities'][0][entity] = temp_list
        elif isinstance(x['entities'][0][entity], dict):
            composite_entity = []
            for k, v in x['entities'][0][entity].items():
                if isinstance(v, str):
                    composite_entity.append(v.lower())
                else:
                    composite_entity.append(v)
            x['entities'][0][entity] = composite_entity
        elif isinstance(x['entities'][0][entity], float):
            x['entities'][0][entity] = [str(int(x['entities'][0][entity]))]
    return x

def find_true_tag(word, expected_entities):
    for key, value in expected_entities.items():
        if word.strip() in value:
            return key
    return "missing"

def compute_entity_metrics(expected, predicted, debug):
        counts = defaultdict(lambda: defaultdict(lambda: 0))
        tracked_entities = [
            'payments_bill_type',
            'payments_bill_action',
            'payments_navigation_target',
            'payments_transaction_action',
            'payments_name',
            'payments_transaction_amount'
        ]
        errors = []
        for idx, (e, p) in enumerate(tqdm(zip(expected, predicted))):
            if e['labeled_intent'] != p['intent']:
                continue

            p_entities = p['entities'][0]
            e_entities = e['labeled_entities']

            all_entities = set(p_entities.keys()) | set(e_entities.keys())
            for entity in all_entities:
                if entity not in tracked_entities:
                    continue
                if entity in p_entities and entity not in e_entities:
                    counts[entity]['fp'] += len(p_entities[entity])
                    error_string = f"{entity} predicted but not in expected"
                    errors.append([e['index'], error_string])
                elif entity not in p_entities and entity in e_entities:
                    counts[entity]['fn'] += len(e_entities[entity])
                    error_string = f"{entity} not predicted but in expected"
                    errors.append([e['index'], error_string])
                else:
                    for val in p_entities[entity]:
                        if val.lower() in e_entities[entity]:
                            counts[entity]['tp'] += 1
                        else:
                            counts[entity]['fn'] += 1
                            true_tag = find_true_tag(val, e_entities)
                            counts[true_tag]['fp'] += 1
                            error_string = f"{val} in {entity} is predicted but not in expected"  # noqa
                            errors.append([e['index'], error_string])

        tp_sum = fp_sum = fn_sum = 0
        for k, v in counts.items():
            tp_sum += counts[k]['tp']
            fp_sum += counts[k]['fp']
            fn_sum += counts[k]['fn']

            try:
                p = counts[k]['tp'] / (counts[k]['tp'] + counts[k]['fp'])
            except ZeroDivisionError:
                p = 0
            try:
                r = counts[k]['tp'] / (counts[k]['tp'] + counts[k]['fn'])
            except ZeroDivisionError:
                r = 0
            try:
                f1 = 2 * p * r / (p + r)
            except ZeroDivisionError:
                f1 = 0

            counts[k]['precision'] = p
            counts[k]['recall'] = r
            counts[k]['f1-score'] = f1
            
        details = {}
        metrics = dict(counts)
        for k, v in metrics.items():
            details[k] = v['f1-score']
            
        overall_precision = tp_sum / (tp_sum + fp_sum)
        overall_recall = tp_sum / (tp_sum + fn_sum)
        overall_f1_score = 2 * overall_precision * overall_recall \
            / (overall_precision + overall_recall)
            
        return {
            'precision': overall_precision,
            'recall': overall_recall,
            'f1': overall_f1_score,
            'details': details,
        }
    
def evaluate_entities(gold_responses, pred_responses, debug):
        print('Evaluating entities (len: {})'.format(len(gold_responses)))

        preprocessed_pred = []
        preprocessed_gold = []

        tmp_pred = deepcopy(pred_responses)
        tmp_gold = deepcopy(gold_responses)

        for idx, response in enumerate(tqdm(
                tmp_gold,
                desc='Preprocessing gold'
        )):
            
            preprocessed_gold.append(
                convert_gold_response_to_list(
                    response,
                    ignorelist=[],
                    reprocesslist=[],
                )
            )

        for idx, response in enumerate(tqdm(
                tmp_pred,
                desc='Preprocessing predicted'
        )):

            preprocessed_pred.append(
                convert_pred_response_to_list(response)
            )

        print('Finished pre-processing {} gold and {} predicted'.format(
            len(preprocessed_gold),
            len(preprocessed_pred),
        ))

        metrics = compute_entity_metrics(
            preprocessed_gold,
            preprocessed_pred,
            debug
        )
        
        return metrics
    
def evaluate_intents(data, vv_responses):
    valid_intents = ['payments_bill', 
                     'small_talk_greeting', 
                     'payments_transactions', 
                     'no_intent', 
                     'payments_navigation' ]
    predicted = []
    expected = []
    label_encoder = LabelEncoder()
    for idx, (e, p) in enumerate(zip(data, vv_responses)):
        if e['labeled_intent'] == 'n/a':
            continue
        predicted.append(p['intent'] if p['intent'] else 'no_intent')
        expected.append(e['labeled_intent'])

    label_encoder.fit_transform(valid_intents)
    p_intents = label_encoder.transform(predicted)
    e_intents = label_encoder.transform(expected)
    target_names = label_encoder.inverse_transform(list(set(p_intents)))
    d = classification_report(
            e_intents,
            p_intents,
            digits=4,
            output_dict=True
        )
    return d