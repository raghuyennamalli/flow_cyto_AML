from FlowCytometryTools import FCMeasurement

sample = FCMeasurement(ID='X', datafile="/storage/mezya.sezen/mphasis/dataset/final_scaled/BLAST110_100_P1_Cleaned_transformed_scaled.fcs")

meta = sample.meta['_channels_']

instrument_cols = meta['$PnN'].tolist()

df = sample.data.copy()
df.columns = instrument_cols

print(df.columns)
