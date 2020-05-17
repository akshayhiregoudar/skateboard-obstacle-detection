# Skateboard Force Estimation

## Instructions
The installation of RPi.GPIO and matplotlib libraries may be necessary.
```bash
pip install RPi.GPIO
pip install matplotlib
```
To run the code, clone or download the master branch, and run the **[kalman_filter.py](kalman_filter.py)** file.

## Introduction 

The purpose of the Skateboard Force Estimation project is to implement a variety of sensors along with Kalman Filtering in order to determine the force enacted on a skateboard as it rides over a bump. The current version of this project allows for an estimation of the distance to a bump using ultrasonic sensors, as well as velocity estimations using Andriod GPS and accelerometer data. The project proposal which we submitted before can be found [here](Proposal.pdf).

The contributors to this project are listed below along with their corresponding TAMU UIN number: 

- Madhu Areti - 128000411
- Zachary Harper - 824007015
- Briana Holton - 528008039
- Akshay Hiregoudar - 128003713
- Devendra Badapurkar - 430000844


## Program Guide 

**[WheelBumpDynmaics.py](WheelBumpDynmaics.py)** This python program receives an input of 
`WheelBumpDynamics(m, R, v, h)` which are the mass of the skateboard, radius of the skateboard wheel, observed velocity of the skateboard, and the height of the observed bump. The program then takes the input information and using dynamic equations to calculate the minimum velocity required to clear the bump. This minimum velocity is then compared to the measured velocity to tell the user whether or not the measured velocity is enough to clear the perceived bump. 
## Project Status
The Skateboard Force Estimation project ultimately had a limited scope due to the impact of COVID-19. Due to the nature of the Robotic Perception (MEEN 689) class being a one-semester class, this project will end with the conclusion of the class. However, future work in this area could include estimating heights of particular bumps and implementing the project in real time.

