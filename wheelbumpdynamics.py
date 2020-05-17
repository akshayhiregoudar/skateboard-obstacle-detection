import numpy as np
import sympy as sp

def wheelbumpdynamics(m, R, v, h):

    # Function Inputs
    # m - mass on the wheel (mass of the wheel + 0.25(mass of the person riding the board) )
    # R - radius of wheel
    # v - measured velocity of skateboard
    # h - perceived height of bump

    # This function assumes a cylindrical wheel that is moving at some velocity
    # towards a bump, the function uses dynamic equations to determine what the
    # minimum velocity is that is required to clear the bump. If the measured
    # velocity is not higher than the required velocity the function will
    # output, "The velocity is not enough to get over the bump", whereas if the
    # velocity is enough to clear the bump the function will output, "The
    # velocity is enough to get over the bump." The required velocity will also
    # be output for both scenarios.


    height = h
    I = (m*R)/2 #cylindrical MOI
    g = 9.81 #m/s**2
    vm = sp.Symbol('vm') #Define vm as a symbol
    w = vm *R;
    w2 = (m*vm*(R-h)+I*w)/(I+m*R**2)
    eq1 = sp.Eq(1/2*(I+m*R**2)*w2**2-m*g*h,0) #set dynamic equaiton equal to zero
    vm = sp.solve(eq1,vm) #using symbolic solver solve for vm using dynamic equation
    vm = vm[1]
    #since the dynamics equation is quadratic there will be two values for vm a positive and negative coming
    #from the square root thus take the second entry.

    #Print the minimum velocity required to get over a certain height of bump
    print('The minimum velocity required to clear a bump {} meters high is {:.3f} meters per second.'.format(height, vm))

    #if-else statement gives the user feedback if the velocity is high enough or not.
    if v > vm:
        print('The velocity is enough to get over the bump.')
    elif v < vm:
        print('The veloctiy is not enough to get over the bump.')
    else:
        print('The system is not solvable.')
