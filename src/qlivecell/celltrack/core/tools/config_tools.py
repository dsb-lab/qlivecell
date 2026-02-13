import inspect
"""
    copied from https://stackoverflow.com/questions/12627118/get-a-function-arguments-default-value
"""
def get_default_args(func):
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }

