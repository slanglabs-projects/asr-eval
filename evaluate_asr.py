import os
import io
import re
import ast
import json
import jiwer
import argparse
import requests
import pandas as pd
from tqdm import tqdm
from copy import deepcopy
from tqdm.contrib import tzip
from slang_metrics import SlangMetrics
from metrics import evaluate_intents, evaluate_entities
import base64
from utils import Config
from google.cloud import speech
import urllib.request
from subprocess import PIPE, Popen
from slanglabs_nlu.entity_extraction.parsers import parse_numbers
from pydub import AudioSegment


hypothesis_cache = {}
dropped_idx = []
urls_checked = False

class Driver(object):
    def __init__(self, config, cohort): 
        global urls_checked, dropped_idx
        self.config = config
        self.cohort = cohort
        file_name = self.config.config['data']
        with open(file_name) as f:
            self.data = json.load(f)
        # self.df = pd.read_csv(file_name)[:10]
        # self.df = self.df.head(250)
        self.dump_file = self.config.config['reference_dump']
        self.results_dump_file = self.config.config['results_dump']
        self.metrics = SlangMetrics()
        self.directory = self.config.config['directory']
        self.bhashini_ip = self.config.config['bhashini_ip']
        self.use_slang_normalizer = self.config.config['use_slang_normalizer']
        self.use_nemo = self.config.config['use_nemo']
        self.use_number_parser = self.config.config['use_number_parser']
        self.use_url = self.config.config['use_url']
        asr_hints_file = self.config.config['asr_hints_file']
        self.asr_hints = self.read_asr_hints(asr_hints_file)
        self.base_url = self.config.config['base_url']
        self.dropped_idx = []
        self.df = pd.DataFrame([])
        if self.use_nemo:
            from nemo_text_processing.inverse_text_normalization.inverse_normalize import InverseNormalizer
            self.inverse_normalizer  = InverseNormalizer(lang='en') 
        if self.use_url is False:
            self.files = os.listdir(self.directory)
        else:
            pass
            if urls_checked == False:
                print("Checking and dropping wrong URLs")
                dropped_idx = self.check_url_exists()
                urls_checked = True
            #self.df = self.df.drop(dropped_idx)
            print(dropped_idx)
            print(f"Dropped {len(dropped_idx)} URLs. {len(self.df)} URLs left") # noqa
        self.asr_engines = self.config.config['engines']

    def check_url_exists(self):
        indices = []
        for idx, f in enumerate(tqdm(self.data)):
            f = self.data[idx]['file_name']
            url = self.base_url + f.replace(" ", "-") + '.wav'
            status_code = requests.get(url).status_code
            if status_code != 200:
                indices.append(idx)
        return indices

    def read_asr_hints(self, hints_file):
        with open(hints_file) as f:
            hints = json.load(f)
        return hints

    def normalize_str(self, string):
        string = re.sub(r'[!"#$*,\/;<=>?[\]^_`{|}~.]+', '', string)  # noqa  Remove special characters
        string = re.sub(r'[()]', ' ', string)
        return string.strip().lower()

    
    def filter_entities(self, responses, pred_response=False):
        _filter = [
            'payments_bill_type',
            'payments_bill_action',
            'payments_navigation_target',
            'payments_transaction_action',
            'payments_name',
            'payments_transaction_amount'
        ]

        _responses = responses
        for i, item in enumerate(_responses):
            ent = []
            if len(item["entities"]) > 0:
                obj = {}
                if pred_response:
                    for k, v in item["entities"][0].items():
                        if k in _filter:
                            obj[k] = v
                else:
                    for k, v in item["entities"].items():
                        if k in _filter:
                            obj[k] = v

                ent.append(obj)
            if len(ent) > 0:
                item["entities"] = ent
            else:
                item["entities"] = [{}]
            _responses[i] = item
        return _responses

    def transform_text(self, text):
        out = text
        if self.use_slang_normalizer:
            out = self.normalize_str(out)

        if self.use_nemo:
            out = self.inverse_normalizer.inverse_normalize(out, verbose=False)

        if self.use_number_parser:
            out = self.apply_number_parser(out)

        return out
            

    def apply_number_parser(self, text):
        updated_text = deepcopy(text)
        numbers = parse_numbers(text)
        for number in numbers:
            start_idx, end_idx = number[1]
            substring = text[start_idx:end_idx+1]
            updated_text = updated_text.replace(substring, str(number[0]))
        return updated_text

    def prepare_google_speech(self, use_asr_hints):
        self.client = speech.SpeechClient()
        
        if use_asr_hints:
            context = {
                'phrases': self.asr_hints
            }
        else:
            context = {}
        self.asr_config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code="en-IN",
            speech_contexts=[context]
        )

    def transcribe_google_speech(self, speech_url):
        """Transcribe the given audio url"""
        speech_file = "temp.wav"
        if self.use_url:
            urllib.request.urlretrieve(speech_url, speech_file)
        else:
            speech_file = speech_url

        sound = AudioSegment.from_wav(speech_file)
        sound = sound.set_channels(1)
        sound.export(speech_file, format="wav")


        with io.open(speech_file, "rb") as audio_file:
            content = audio_file.read()
        audio = speech.RecognitionAudio(content=content)
        response = self.client.recognize(config=self.asr_config, audio=audio)
        google_transcript = ''
        for result in response.results:
            google_transcript = result.alternatives[0].transcript
        return google_transcript

    def transcribe_bhashini(self, audio):
        command = "bash transcribe.sh {audio}".format(audio=audio)
        output = None
        with Popen(command, stdout=PIPE, stderr=None, shell=True) as process:
            output = process.communicate()[0].decode("utf-8")
        try:
            return ast.literal_eval(output)['output'][0]['source']
        except: # noqa
            return ""

    def transcribe_indic_conformer(self, speech_url):
        # API URL
        API_URL = f'http://{self.bhashini_ip}:4992/recognize/en'

        speech_file = "temp.wav"

        urllib.request.urlretrieve(speech_url, speech_file)

        # Load the wav file into the base64 format
        with open("temp.wav", "rb") as wav_file:
            encoded_string = base64.b64encode(wav_file.read())
        # Encode the file.
        encoded_string = str(encoded_string, 'ascii', 'ignore')

        # POST request data format
        data = {
            "config": {
                "language": {
                    "sourceLanguage": "en"
                },
                "transcriptionFormat": {
                    "value": "transcript"
                },
                "audioFormat": "wav",
                "samplingRate": "16000",
                "postProcessors": None
            },
            "audio": [
                {
                    "audioContent": encoded_string
                }
            ]
        }

        # Send the API request
        x = requests.post(API_URL, data=json.dumps(data))
        return json.loads(x.text)["output"][0]["source"]

    def transcript_audio(self, url, engine):
        if engine == 'google_speech':
            self.prepare_google_speech(False)
            out = self.transcribe_google_speech(url)
        elif engine == 'slang_google_speech':
            self.prepare_google_speech(True)
            out = self.transcribe_google_speech(url)
        elif engine == 'indic_conformer':
            out = self.transcribe_indic_conformer(url)
        '''
        elif engine == 'nemo_indic_conformer':
            out = self.transcribe_indic_conformer(url)
            out = inverse_normalizer.inverse_normalize(out, verbose=False)
        elif engine == 'slang_indic_conformer':
            out = self.transcribe_indic_conformer(url)
            out = inverse_normalizer.inverse_normalize(out, verbose=False)
            out = self.apply_number_parser(out)
        elif engine == 'nemo_google_speech':
            self.prepare_google_speech(True)
            out = self.transcribe_google_speech(url)
            out = inverse_normalizer.inverse_normalize(out, verbose=False)
        elif engine == 'indic_wav2vec':
            out = self.transcribe_bhashini(url)
        elif engine == 'slang_indic_wav2vec':
            out = self.transcribe_bhashini(url)
            out = self.normalize_str(out)
            out = self.apply_number_parser(out)
        '''
        return out
    
    def get_best_wer(self, asr_transcript, references):
        wers = []
        for ref in references:
            wers.append(jiwer.wer(ref, asr_transcript))
        return min(wers)

    def run(self):
        with open(self.dump_file) as f:
            reference_responses = json.load(f)
        reference_responses = [res for i, res in enumerate(reference_responses) if i not in dropped_idx]   # noqa
        self.data = [res for i, res in enumerate(self.data) if i not in dropped_idx]
        all_references = []
        for engine in self.asr_engines:
            print(f"Computing WER for {engine}")
            predicted = []
            wers = []
            score_list = []
            urls = []
            raw_trascriptions = []
            for i, f in enumerate(tqdm(self.data)):
                references = f['references']
                f = f['file_name']
                if self.use_url:
                    filepath = self.base_url + f.replace(" ", "-") + '.wav'
                else:
                    filepath = os.path.join(self.directory, f + '.wav')
                urls.append(filepath)
                try:
                    check_hypothesis_cache = hypothesis_cache.get(engine)
                    if check_hypothesis_cache is None:
                        hypothesis_cache.update({engine:[]})

                    if len(hypothesis_cache.get(engine)) == len(reference_responses):
                        hypothesis = hypothesis_cache[engine][i]
                    else:
                        hypothesis = self.transcript_audio(filepath, engine)
                        hypothesis_cache[engine].append(hypothesis)

                    raw_trascriptions.append(hypothesis)
                    hypothesis = self.transform_text(hypothesis)
                    references = [self.transform_text(reference) for reference in references]
                    # wer = jiwer.wer(reference, hypothesis)
                    wer = self.get_best_wer(hypothesis, references)
                    all_references.append(",".join(references))
                    predicted.append(hypothesis)
                    wers.append(wer)
                except Exception as e:
                    print('Exception caught', e)
                    predicted.append('')
                    wers.append(1.0)
            self.df['References'] = [",".join(i['references']) for i in self.data]
            if len(all_references) == len(self.data):
                self.df["transformed_references"] = all_references
            self.df[engine+'_transcription_raw'] = raw_trascriptions
            self.df[engine + '_transcription'] = predicted
            self.df[engine + '_wer'] = wers
            if self.config.config['score_nlp']:
                pred_responses, _ = self.config.send_and_time_request(self.df[engine + '_transcription'])   # noqa
                pred_responses = self.filter_entities(pred_responses, pred_response=True)
                self.df[engine+'_pred_response'] = pred_responses
                intent_metrics = evaluate_intents(reference_responses, pred_responses)
                entity_metrics = evaluate_entities(reference_responses, pred_responses, False)
                with open(f'intent_metrics_{engine}_{self.cohort}.json', 'w') as json_file:
                    json.dump(intent_metrics, json_file)

                with open(f'entity_metrics_{engine}_{self.cohort}.json', 'w') as json_file:
                    json.dump(entity_metrics, json_file)
                '''
                for i, (expected, predicted) in enumerate(tzip(reference_responses, pred_responses)):         # noqa
                    asr_score = self.metrics.compute_asr_score(expected, predicted)
                    score_list.append(asr_score)
                self.df[engine+'_slang_score'] = score_list
                '''

        # self.df['reference_response'] = reference_responses
        self.df['audio_links'] = urls
        self.df.to_csv(self.results_dump_file, index=False)


def main(tier):
    config = Config(tier=tier, config_file='asr_config.yaml')
    driver = Driver(config, cohort="1")
    driver.run()

    # config2 = Config(tier=tier, config_file='asr_config_nemo.yaml')
    # driver2 = Driver(config2, cohort="2")
    # driver2.run()

    config3 = Config(tier=tier, config_file='asr_config_np.yaml')
    driver3 = Driver(config3, cohort="3")
    driver3.run()

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--tier',
        choices=['local', 'stage', 'prod'],
        required=True,
        help='tier on which to run the evaluation',
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(
        args.tier
    )
