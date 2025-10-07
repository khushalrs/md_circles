from setuptools import setup
package_name = 'md_circles'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name + '/config', ['config/qos_profiles.yaml']),
        ('share/' + package_name + '/launch', [
            'launch/circles.launch.py',
            'launch/playback.launch.py'
        ]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='you',
    maintainer_email='you@example.com',
    description='Multi-drone circle detection, 3D sizing, and dedupe.',
    license='BSD-3-Clause',
    entry_points={
        'console_scripts': [
            'circle_detector_node = md_circles.circle_detector_node:main',
            'dedupe_and_logger_node = md_circles.dedupe_and_logger_node:main',
            'circle_marker_node = md_circles.circle_marker_node:main',
        ],
    },
)
