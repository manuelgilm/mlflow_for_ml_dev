from package.feature.data_processing import get_feature_dataframe
from package.ml_training.retrieval import get_train_test_score_set

import json
import requests
from pprint import pprint



if __name__ == "__main__":
    df = get_feature_dataframe()
    x_train, x_test, x_score, y_train, y_test, y_score = get_train_test_score_set(df)
    features = [f for f in x_train.columns if f not in ["id", "target", "MedHouseVal"]]

    feature_values = json.loads(x_score[features].iloc[1:2].to_json(orient="split"))
    # print(feature_values)
    payload = {"dataframe_split": feature_values}
    # pprint(
    #     payload,
    #     indent=4,
    #     depth=10,
    #     compact=True,
    # )

    BASE_URI = "http://127.0.0.1:5000/"
    headers = {"Content-Type": "application/json"}
    endpoint = BASE_URI + "invocations"
    r = requests.post(endpoint, data=json.dumps(payload), headers=headers)
    print(f"STATUS CODE: {r.status_code}")
    print(f"PREDICTIONS: {r.text}")
    print(f"TARGET: {y_score.iloc[1:2]}")
