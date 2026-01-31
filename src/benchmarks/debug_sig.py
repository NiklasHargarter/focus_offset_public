import inspect
from src.models.architectures import ConvNeXtV2FocusRegressor

print(f"Class: {ConvNeXtV2FocusRegressor}")
sig = inspect.signature(ConvNeXtV2FocusRegressor.__init__)
print(f"Signature: {sig}")
for name, param in sig.parameters.items():
    print(f"  Param: {name}, Default: {param.default}, Annotation: {param.annotation}")
