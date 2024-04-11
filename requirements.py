import yaml

# Path to the environment.yml file
env_file = 'environment.yml'

# Load the environment.yml file
with open(env_file, 'r') as file:
    env_spec = yaml.safe_load(file)

# Extract dependencies from the 'dependencies' section
dependencies = env_spec.get('dependencies', [])

# Extract dependencies from the 'pip' section
pip_dependencies = env_spec.get('dependencies', {}).get('pip', [])

# Combine the dependencies from both sections into a single list
all_dependencies = dependencies + pip_dependencies

# Write the dependencies to a requirements.txt file
with open('requirements.txt', 'w') as file:
    for dependency in all_dependencies:
        file.write(dependency + '\n')
