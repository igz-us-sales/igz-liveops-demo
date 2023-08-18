import os
import yaml

from src import create_and_set_project

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
    
project = create_and_set_project(git_source="git://github.com/igz-us-sales/igz-liveops-demo#master")

run_id = project.run(name="main", arguments=config, dirty=True, watch=True)