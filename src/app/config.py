import yaml


def load_config(config_path):
    """Loads and validates the YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Simple validation
    if 'bot' not in config or 'exchange' not in config or 'strategy' not in config['bot']:
        raise KeyError("Invalid config file")
        
    return config
