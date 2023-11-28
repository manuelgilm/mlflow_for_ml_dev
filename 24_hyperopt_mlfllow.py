from mlflow_utils import create_dataset

from hyperopt import fmin 
from hyperopt import tpe
from hyperopt import Trials
from hyperopt import hp


def objective_function(params):

    y = (params["x"] + 3) ** 2 + 2 

    return y

search_space = {
    "x": hp.uniform("x", -10, 10)
}

trials = Trials()

best = fmin(
    fn=objective_function,
    space=search_space,
    algo=tpe.suggest,
    max_evals=100,
    trials=trials
)

print(best)