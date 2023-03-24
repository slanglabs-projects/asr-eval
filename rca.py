import pandas as pd

df1 = pd.read_csv('ASR_quality_metrics_1.csv')
df2 = pd.read_csv('ASR_quality_metrics_2.csv')
df3 = pd.read_csv('ASR_quality_metrics_3.csv')

combined_df = pd.DataFrame()
combined_df['raw'] = df1['References']
combined_df['nemo'] = df2['transformed_references']
combined_df['number_parser'] = df3['transformed_references']
combined_df['raw_indic_conformer'] = df1['indic_conformer_transcription']
combined_df['raw_google_speech'] = df1['google_speech_transcription']
combined_df['raw_slang_google_speech'] = df1['slang_google_speech_transcription']
combined_df['nemo_indic_conformer'] = df2['indic_conformer_transcription']
combined_df['nemo_google_speech'] = df2['google_speech_transcription']
combined_df['nemo_slang_google_speech'] = df2['slang_google_speech_transcription']
combined_df['np_indic_conformer'] = df3['indic_conformer_transcription']
combined_df['np_google_speech'] = df3['google_speech_transcription']
combined_df['np_slang_google_speech'] = df3['slang_google_speech_transcription']

combined_df.to_csv('rca.csv', index=False)

