grpcurl -import-path /Users/harikrishnanc/speech-recognition-open-api/proto -proto speech-recognition-open-api.proto -plaintext -d @ grpc-bhaskar.slanglabs.in:50051 ekstep.speech_recognition.SpeechRecognizer.recognize <<EOM
{
    "config": { 
      "language": {
      "sourceLanguage": "en"
    },
        "transcriptionFormat": {
            "value": "transcript"
        },
        "audioFormat": "wav",
        "punctuation": true,
        "enableInverseTextNormalization": true
    },
    "audio": [
        {
            "audioUri": "$1"
        }
    ]
}
EOM