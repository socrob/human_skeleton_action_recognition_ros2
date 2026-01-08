from setuptools import setup, find_packages

package_name = "hsar"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=[
        "setuptools"
    ],
    zip_safe=True,
    maintainer="Rodrigo Serra",
    maintainer_email="rodrigo.serra@tecnico.ulisboa.pt",
    description="ROS 2 LifecycleNode wrapper for real-time Human Skeleton Action Recognition",
    license="GPL-3",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "detect_posture_node = hsar.detect_posture_node:main",
            "mediapipe_pose_node = hsar.mediapipe_pose_node:main",
        ],
    },
)
