## Reinforcement Learning for Control of 2DoF Inverted Pendulum
*Not that this project is not maintained, but should still be functional*

This is a reinforcement Learning based control solution for the [Quanser 2DoF Inverted Pendulum](https://www.quanser.com/products/2-dof-inverted-pendulumgantry/). 

The implementation mainly focus on [Stable Baselines](https://github.com/hill-a/stable-baselines) implementation of Proximal Policy Optimization (PPO) and [Keras-RL](https://github.com/keras-rl/keras-rl) implementation of Double Deep Q Network. It also includes a generic wrapper for [OpenAi GYM](https://github.com/openai/gym), [Stable Baselines](https://github.com/hill-a/stable-baselines) and [Keras-RL](https://github.com/keras-rl/keras-rl). 

### About
This project was part of the course *Design project in Systems, Control and Mechatronics* at Chalmers University of Technology, Sweden, during autumn of 2018.

##### Group Members
Jonathan Jansson  
Bernardo T. Barata   
Christian Garcia  
Fredrik J. Tornéus   

### Dependencies 
(Apart from OpenAi GYM + stable-baselines/keras-rl and respective dependencies)
* openai
* gym
* stable-baselines
* numpy
* PyQt5
* pyqtgraph
