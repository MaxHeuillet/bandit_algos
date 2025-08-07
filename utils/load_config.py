
import os
import re
from types import SimpleNamespace

def resolve_vars(config):
    """Recursively resolve ${var} and ${oc.env:VAR} entries."""
    pattern = re.compile(r"\$\{([^\}]+)\}")

    def resolve_value(value, config):
        if isinstance(value, str):
            while True:
                matches = pattern.findall(value)
                if not matches:
                    break
                for match in matches:
                    if match.startswith("oc.env:"):
                        # Handle defaults: ${oc.env:VAR, default}
                        env_expr = match[len("oc.env:"):]
                        if ',' in env_expr:
                            varname, default = map(str.strip, env_expr.split(",", 1))
                            replacement = os.getenv(varname, default)
                        else:
                            replacement = os.getenv(env_expr, "")
                    else:
                        # Reference to another config key
                        parts = match.split(".")
                        subconfig = config
                        for part in parts:
                            if isinstance(subconfig, dict):
                                subconfig = subconfig.get(part, "")
                            else:
                                subconfig = getattr(subconfig, part, "")
                        replacement = subconfig
                    value = value.replace(f"${{{match}}}", str(replacement))
        return value

    def recursive_resolve(obj, config):
        if isinstance(obj, dict):
            return {k: recursive_resolve(resolve_value(v, config), config) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [recursive_resolve(resolve_value(item, config), config) for item in obj]
        else:
            return resolve_value(obj, config)

    resolved = config
    for _ in range(3):
        resolved = recursive_resolve(resolved, resolved)

    return resolved

def dict_to_namespace(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    return d