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
import base64
from utils import Config
from google.cloud import speech
import urllib.request
from subprocess import PIPE, Popen
from slanglabs_nlu.entity_extraction.parsers import parse_numbers


class Driver(object):
    def __init__(self, config):
        self.config = config
        file_name = self.config.config['data']
        self.df = pd.read_csv(file_name)
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
        if self.use_nemo:
            from nemo_text_processing.inverse_text_normalization.inverse_normalize import InverseNormalizer
            self.inverse_normalizer  = InverseNormalizer
        if self.use_url is False:
            self.files = os.listdir(self.directory)
        else:
            print("Checking and dropping wrong URLs")
            self.dropped_idx = self.check_url_exists()
            self.df = self.df.drop(self.dropped_idx)
            print(self.dropped_idx)
            print(f"Dropped {len(self.dropped_idx)} URLs. {len(self.df)} URLs left") # noqa
        self.asr_engines = self.config.config['engines']

    def check_url_exists(self):
        indices = []
        for idx, f in enumerate(tqdm(self.df['File_name'])):
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

    def filter_entities(self, responses):
        for item in responses:
            _filter = [
                'payments_bill_type',
                'payments_bill_action',
                'payments_navigation_target',
                'payments_transaction_action',
                'payments_transaction_name',
                'payments_transaction_amount'
            ]

            ent = []
            for i in item["entities"]:
                n = next(iter(i))
                if n in _filter:
                    ent.append(i)

            item["entities"] = ent

        return responses

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
            language_code="en-US",
            speech_contexts=[context]
        )

    def transcribe_google_speech(self, speech_url):
        """Transcribe the given audio url"""
        speech_file = "temp.wav"
        if self.use_url:
            urllib.request.urlretrieve(speech_url, speech_file)
        else:
            speech_file = speech_url

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

    def run(self):
        with open(self.dump_file) as f:
            reference_responses = json.load(f)
        reference_responses = [res for i, res in enumerate(reference_responses) if i not in self.dropped_idx]   # noqa
        reference_responses = self.filter_entities(reference_responses)
        for engine in self.asr_engines:
            print(f"Computing WER for {engine}")
            predicted = []
            wers = []
            score_list = []
            urls = []
            for i, f in enumerate(tqdm(self.df['File_name'])):
                reference = self.df['Reference_transcript '].iloc[i]
                if self.use_url:
                    filepath = self.base_url + f.replace(" ", "-") + '.wav'
                    urls.append(filepath)
                else:
                    filepath = os.path.join(self.directory, f + '.wav')
                try:
                    hypothesis = self.transcript_audio(filepath, engine)
                    hypothesis = self.transform_text(hypothesis)
                    reference = self.transform_text(reference)
                    wer = jiwer.wer(reference, hypothesis)
                    predicted.append(hypothesis)
                    wers.append(wer)
                except Exception as e:
                    print('Exception caught', e)
                    predicted.append('')
                    wers.append(1.0)
            self.df[engine + '_transcription'] = predicted
            self.df[engine + '_wer'] = wers
            pred_responses, _ = self.config.send_and_time_request(self.df[engine + '_transcription'])   # noqa
            for i, (expected, predicted) in enumerate(tzip(reference_responses, pred_responses)):         # noqa
                asr_score = self.metrics.compute_asr_score(expected, predicted)
                score_list.append(asr_score)
            self.df[engine+'_slang_score'] = score_list
        self.df['audio_links'] = urls
        self.df.to_csv(self.results_dump_file, index=False)


def main(tier):
    config = Config(tier=tier)
    driver = Driver(config)
    driver.run()


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
