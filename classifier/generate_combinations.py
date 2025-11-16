import itertools


def generate_command_combinations(options_map):
    """
    Generates all possible combinations of command-line arguments based on a map of options.

    Args:
        options_map (dict): A dictionary where keys are flag names (e.g., '--flag')
                            and values are lists or ranges of possible values.

    Returns:
        list: A list of strings, where each string is a full command line argument set.
    """
    # 1. Prepare the value lists for each option
    # Use map(options_map.get, keys) and itertools.product to get combinations of values
    keys = list(options_map.keys())
    value_combinations = itertools.product(*map(options_map.get, keys))

    # 2. Format each combination into a command-line argument string
    commands = []
    for values in value_combinations:
        # Combine flags and their values into a single list of strings
        args_list = []
        for flag, value in zip(keys, values):
            # For boolean-like flags that don't take a value, just add the flag if True
            if value is True and flag.startswith('--'):
                args_list.append(flag)
            elif value is not None and value is not False:
                args_list.append(flag)
                args_list.append(str(value))

        # Join the arguments into a single command string
        command = " ".join(args_list)
        commands.append(command)

    return commands


# Example usage:

# Define options and their possible values/ranges
# Note: For integer ranges, use range() or list(range())
options = {
    "--model": ["resnet50", "vgg16"],
    "--learning_rate": [0.01, 0.001],
    "--batch_size": list(range(16, 65, 16)),  # Range from 16 to 64, step 16 (16, 32, 48, 64)
    "--gpu": [0, 1, None]  # Use None for optional flags that may not be present
}

# Generate all combinations
all_commands = generate_command_combinations(options)

# Print the results
print(f"Generated {len(all_commands)} command combinations:")
for i, cmd in enumerate(all_commands):
    print(f"{i + 1}: python script_name.py {cmd}")

