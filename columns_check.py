from FlowCytometryTools import FCMeasurement

f = "/storage/mezya.sezen/mphasis/dataset/final_scaled/BLAST110_100_P1_Cleaned_transformed_scaled.fcs"
sample = FCMeasurement(ID=f, datafile=f)
print(sample.data.columns)
