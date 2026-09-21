from FlowCytometryTools import FCMeasurement

f = "/storage/mezya.sezen/mphasis/dataset/final_scaled/BLAST110_100_P1_Cleaned_transformed_scaled.fcs"
s = FCMeasurement(ID="x", datafile=f)

for i in range(1, 40):
    print(i,
          "Detector:", s.meta.get(f"$P{i}N"),
          "Marker:", s.meta.get(f"$P{i}S"))
