from inspect import _empty as inspect_empty
from typing import Dict, Any, Optional
from pprint import pformat
import inspect


__all__ = ['Registry', 'inspect_empty']


class Registry:
    def __init__(self):
        self.table: Dict[str, Any] = {}

    def reg(
            self,
            name: Optional[str] = None,
            params: Optional[Dict[str, Any]] = None,
            param_desc: Optional[Dict[str, Any]] = None,
    ):
        params, param_desc = params or {}, param_desc or {}
        
        # Validate param_desc is subset of params
        param_keys = set(params.keys()) | set(param_desc.keys())
        for key in param_keys:
            if key in params and key not in param_desc:
                param_desc[key] = "No description"
            if key in param_desc and key not in params:
                raise KeyError(f"Parameter description '{key}' has no matching parameter")

        def wrapper(cls):
            # Register with lowercase key
            key = (name if name else cls.__name__ if hasattr(cls, '__name__') else str(cls)).lower()
            if key in self.table:
                raise KeyError(f"'{key}' already registered as {self.table[key]}")
            
            # Check all registered parameter names are valid
            if isinstance(cls, type):
                valid_params = inspect.signature(cls.__init__).parameters
                subkey = 'class'
            elif callable(cls):
                valid_params = inspect.signature(cls).parameters
                subkey = 'function'
            else:
                raise TypeError(f"Cannot register {cls} as a class or function")
            for p in params.keys() | param_desc.keys():
                assert p in valid_params, f"Try to register invalid parameter {p} in {cls.__name__}"

            self.table[key] = {
                subkey: cls,
                'registered_params': params,
                'registered_param_desc': param_desc,
                'valid_params': [p for p in valid_params.keys()],
            }
            return cls
        return wrapper

    def get(self, name: str, *args, **kwargs):
        if name.lower() not in self.table:
            raise KeyError(f"'{name}' not registered")
        cls = self.table[name.lower()]

        if 'class' in cls:
            cls = self.table[name.lower()]['class']
            valid_params = inspect.signature(cls.__init__).parameters
        elif 'function' in cls:
            cls = self.table[name.lower()]['function']
            valid_params = inspect.signature(cls).parameters

        # Validate and inject parameters
        for param_name, value in kwargs.items():
            if param_name not in valid_params:
                raise KeyError(f"Cannot inject '{param_name}={value}' into {cls}")
            if param_name in valid_params and param_name not in self.table[name.lower()]['registered_params']:
                self.table[name.lower()]['registered_param_desc'][param_name] = "Force injection"
            self.table[name.lower()]['registered_params'][param_name] = value

        # Check required parameters
        for param_name, value in self.table[name.lower()]['registered_params'].items():
            assert value is not inspect_empty, f"Required parameter '{param_name}' not provided for {name}"
        return cls(*args, **self.table[name.lower()]['registered_params'])
    
    def has_param(self, key: str, param: str):
        return param in self.table[key]['valid_params']

    def __str__(self) -> str:
        """Return string representation of registry contents."""
        return pformat(self.table)
