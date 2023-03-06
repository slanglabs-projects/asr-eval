import re
import random
import requests
import yaml
from tqdm import tqdm

from google.protobuf.json_format import MessageToDict

try:
    SlangResponsePB()
except NameError:
    from slang_types_pb2 import (REQUEST_TEXT, SlangResponsePB,
                                 SlangRequestPB, SlangContextItemPB)
from time import perf_counter


# helper functions
def create_pb_request(assistant_id, input, etfs=[], iname='', lang='en-IN'):
    request = SlangRequestPB()
    request.type = REQUEST_TEXT
    requestobj = request.text_request
    requestobj.header.app_id = assistant_id
    requestobj.header.request_id = str(random.randint(10, 1000))
    requestobj.header.session_id = str(random.randint(10, 1000))
    requestobj.header.language = lang
    requestobj.header.auth_token = ''
    requestobj.text = input

    if len(etfs) > 0:
        ctx_item = SlangContextItemPB()
        ctx_item.intent_string = iname
        ctx_item.entities_to_resolve.extend(etfs)
        requestobj.header.context.stack.extend([ctx_item])

    return request.SerializeToString()


def parse_list_value(e):
    ret = []
    for item in e.get('listValues', []):
        if item.get('isList', False):
            ret.append(parse_list_value(item))
        elif item.get('isComposite', False):
            ret.append(parse_composite_value(item))
        else:
            ret.append(item['strVal'])
    return ret


def parse_composite_value(e):
    ret = {}
    for k, v in e.get('compositeValues', {}).items():
        if v.get('isList', False):
            ret[k] = parse_list_value(v)
        elif v.get('isComposite', False):
            ret[k] = parse_composite_value(v)
        else:
            ret[k] = v['strVal']
    return ret


def response_pb2dict(pbstr):
    resp_pb = SlangResponsePB()
    resp_pb.ParseFromString(pbstr)
    response = MessageToDict(resp_pb)

    if 'intentResponse' not in response:
        return {}

    intent = response['intentResponse'].get('intentString', '')
    entities = response['intentResponse'].get('entities', [])
    emap = {}
    for e in entities:
        key = e['key']
        if e.get('isList', False):
            emap[key] = parse_list_value(e)
        elif e.get('isComposite', False):
            emap[key] = parse_composite_value(e)
        else:
            emap[key] = e.get('strVal', '')

    return {'intent': intent, 'entities': [emap]}


# preprocessing functions for splitting every entity to a list. this helps us
# to check word-wise tagging accuracy instead of looking at chunks
def convert_gold_response_to_list(config, x, ignorelist, reprocesslist):
    for entity in x['entities']:
        if entity in ignorelist:
            continue
        if entity in reprocesslist:
            req = create_pb_request(config.get_id(), x['entities'][entity])
            resp = requests.post(
                config.get_t2i_url(),
                data=req,
                auth=config.get_auth()
            )
            resp_d = response_pb2dict(resp.content)
            try:
                entities = resp_d.get('entities', [{}])
                x['entities'][entity] = entities[0].get(
                    entity,
                    x['entities'][entity]
                )
            except Exception:
                pass
        if isinstance(x['entities'][entity], list):
            x['entities'][entity] = [i.lower() for i in x['entities'][entity]]
        else:
            x['entities'][entity] = str(x['entities'][entity]).lower().split()
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


class Config(object):
    def __init__(self, tier):
        with open('asr_config.yaml') as f:
            common = yaml.safe_load(f)
        self.tier = tier
        self.host = common.get('hosts', {}).get(self.tier, "")
        self.config = {}
        self.file_name = common.get('data', '')
        self.reference_dump_file = common.get('reference_dump', '')
        self.results_dump_file = common.get('results_dump', '')
        self.config = common

    def __str__(self):
        return "tier: {}".format(
            self.tier
        )

    def get_dataset(self):
        return self.config.get('dataset', None)

    def get_contexts(self):
        return self.config.get('contexts', {})

    def get_prompt_context(self):
        return self.get_contexts().get('prompt_context', {})

    def get_nlu(self):
        return self.config.get('nlu', {})

    def get_all_intents(self):
        return self.get_nlu().get('intents', {}).get('all', [])

    def get_tracked_intents(self):
        return self.get_nlu().get('intents', {}).get('tracked', [])

    def get_tracked_entities(self):
        return self.get_nlu().get('entities', {}).get('tracked', [])

    def get_ignored_entities(self):
        return self.get_nlu().get('entities', {}).get('ignored', [])

    def get_normalized_entities(self):
        return self.get_nlu().get('entities', {}).get('normalized', [])

    def get_identifiers(self):
        m = self.config.get('tiers', {}).get(self.tier, {})
        return m['id'], m['key'], m['env'], m['version']

    def get_id(self):
        return self.get_identifiers()[0]

    def get_key(self):
        return self.get_identifiers()[1]

    def get_env(self):
        return self.get_identifiers()[2]

    def get_version(self):
        return self.get_identifiers()[3]

    def get_auth(self):
        return self.get_key(), 'nothing'

    def get_t2i_url(self):
        url = "{host}/v1.1/assistants/{id}/text2intent/?env={env}&assistant_version={version}&source=script"  # noqa
        return url.format(
            host=self.host,
            id=self.get_id(),
            env=self.get_env(),
            version=self.get_version(),
        )

    def get_t2e_url(self, intent, entity):
        url = "{host}/v1.1/assistants/{id}/intents/{intent}/entities/{entity}/text2entity/?env=stage&assistant_version={version}&source=script"  # noqa
        return url.format(
            host=self.host,
            id=self.get_id(),
            intent=intent,
            entity=entity,
            env=self.get_env(),
            version=self.get_version(),
        )

    def get_skip_ids(self):
        skip_ids = self.config.get('skip_ids', [])
        return [(item['start'], item['finish']) for item in skip_ids]

    def normalize_str(self, string):
        string = re.sub(r'[!"#$*,\/;<=>?[\]^_`{|}~]+', '', string)  # noqa  Remove special characters
        string = re.sub(r'[()]', ' ', string)
        return string.strip().lower()

    def send_and_time_request(self, utterances):
        pred_responses = []
        response_times = []

        # Request to load model
        req = create_pb_request(self.get_id(), "dummy")
        resp = requests.post(self.get_t2i_url(), data=req, auth=self.get_auth())   # noqa

        for idx, res in enumerate(tqdm(utterances)):
            start = perf_counter()
            req = create_pb_request(self.get_id(), self.normalize_str(res))
            resp = requests.post(self.get_t2i_url(), data=req, auth=self.get_auth())  # noqa
            response_time = perf_counter() - start
            response_times.append(response_time)
            response = response_pb2dict(resp.content)
            response['text'] = res
            response['entities'][0].pop('unrecognised_words', '')
            pred_responses.append(response)
        return pred_responses, response_times
