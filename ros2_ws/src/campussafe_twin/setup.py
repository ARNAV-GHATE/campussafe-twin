from glob import glob

from setuptools import find_packages, setup

package_name = "campussafe_twin"
nodes = ["scenario_manager", "attack_injector", "state_builder", "decision_node", "safety_validator",
         "robot_controller", "metrics_logger"]

setup(
    name=package_name,
    version="0.2.0",
    packages=find_packages(exclude=["test"]),
    package_data={"campussafe_twin.learning": ["models/*.pt", "models/*.json"]},
    include_package_data=True,
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/worlds", glob("worlds/*.sdf")),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Arnav Ajaykumar Ghate",
    maintainer_email="arnav.ghate@gmail.com",
    description="CampusSafe Twin: digital twin + SLM / world model + NCM emergency-response robot "
                "with cyberattack resilience",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [f"{n} = campussafe_twin.nodes.{n}:main" for n in nodes],
    },
)
