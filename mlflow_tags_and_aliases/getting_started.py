import mlflow  

# create custom model 
class MyModel(mlflow.pyfunc.PythonModel):
    def predict(self, context, model_input):
        return model_input.apply(lambda column: column * 2)

if __name__=="__main__":

    with mlflow.start_run() as run:
        # Log the model
        # mlflow.pyfunc.log_model("model", python_model=MyModel(), registered_model_name="MyModel")

        client = mlflow.MlflowClient()
        
        # client.set_model_version_tag(name="MyModel", version=1, key="tag_key2", value="New tag_value2")
        # client.set_model_version_tag(name="MyModel", version=1, key="model_status", value="validation")
        # client.set_model_version_tag(name="MyModel", version=2, key="model_status", value="Ready for production")

        # client.set_registered_model_alias(name="MyModel", alias="Archive",version= "1")
        # model_version = client.get_model_version_by_alias(name="MyModel", alias="Champion")

        client.delete_registered_model_alias(name="MyModel", alias="Archive")
