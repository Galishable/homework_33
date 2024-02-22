import pandas as pd
import dill
import os
import json


def prediction() -> None:
    path = os.environ.get('PROJECT_PATH', '..')
    df = pd.DataFrame()
    with open(f'{path}/data/models/cars_pipe.pkl', 'rb') as file:
        model = dill.load(file)
    folder_path = f'{path}/data/test'
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        if os.path.isfile(file_path):
            with open(file_path) as json_file:
                data = json.load(json_file)
            y = pd.DataFrame.from_dict({'id': data['id'],
                                        'model': model.predict(pd.DataFrame([data]))
})
            df = pd.concat([df, y], ignore_index=True)
    df.to_csv(f'{path}/data/predictions/preds.csv', index=False, header=None)


if __name__ == '__main__':
    prediction()
