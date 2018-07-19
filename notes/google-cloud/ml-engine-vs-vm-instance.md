https://stackoverflow.com/questions/44310762/google-cloud-compute-engine-vs-machine-learning

### main differences

    - The most noticeable feature of Google CloudML is the deployment itself. You don't have to take care of things like setting up your cluster (that is, scaling), launching it, installing the packages and deploy your model for training. *So no VM instance, you lust upload data and run the job.*
    - However, there is a substancial difference of CloudMl vs Compute Engine when it comes to prediction. In Compute Engine, you would have to take care of all the quirks of TensorFlow Serving which are not that trivial (compared to training your model).

    - So like Datalab it has kind of very limited functionality.