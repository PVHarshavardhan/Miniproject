
from ultralytics import YOLO

import pandas as pd

best_model = YOLO(r'C:\New folder\runs\segment\train\weights\best.pt')
metrics = best_model.val(split='val')
metrics_df = pd.DataFrame.from_dict(metrics.results_dict, orient='index', columns=['Metric Value'])
print(metrics)
metrics_df.round(3)