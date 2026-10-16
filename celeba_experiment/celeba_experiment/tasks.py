"""One task-selection contract shared by the legacy and rate-selection runners."""
DEFAULT_TASKS = ['Smiling', 'Eyeglasses', 'Blond_Hair', 'Young']


def select_tasks(attribute_names, config):
    cfg = config.get('tasks', {})
    names = list(attribute_names) if not cfg.get('reduced', True) else list(cfg.get('names', DEFAULT_TASKS))
    if not names or len(names) != len(set(names)):
        raise ValueError('Task names must be nonempty and unique')
    missing = set(names) - set(attribute_names)
    if missing:
        raise ValueError(f'Unknown attributes: {sorted(missing)}')
    return names, [attribute_names.index(name) for name in names]
