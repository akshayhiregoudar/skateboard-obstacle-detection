# Accelerometer Data acquisition frequency is higher than GPS

import numpy as np
import matplotlib.pyplot as plt
from kalman import kalman
from numpy.linalg import inv
from numpy import matrix as mat
from data_process import CSVReadGPS, CSVReadAcc, CSVReadUltra, GPSVel, GMTCDTconv
from wheelbumpdynamics import wheelbumpdynamics

# Kalman Filter
def KFprediction(x_prev_est, u_t, del_t, p_prev, q_matrix):
  # b is identify (2x2)
  b_matrix = np.array([del_t,0,0,del_t]).reshape(2,2)
  # convert all the matrices to regular dimensions
  p_prev_mat = mat(p_prev.reshape(2,2))
  q_matrix_mat = mat(q_matrix.reshape(2,2))
  #print ("U: ", u_t)
  # filter
  x_prior = x_prev_est +  np.matmul(b_matrix, u_t)
  p_prior = p_prev_mat + np.matmul(np.matmul(b_matrix, q_matrix_mat), b_matrix.transpose())

  return x_prior, p_prior.reshape(4,1)

def KFupdate(x_prior, r_matrix, p_prior, y_t, u_t):

  # convert all the matrices to regular dimensions
  p_prior_mat = p_prior.reshape(2,2)
  r_matrix_mat = r_matrix.reshape(2,2)

  # update
  if np.linalg.det(r_matrix_mat) == 0:
    r_mat_diag = r_matrix_mat.diagonal();
    # assign a very small value to make the matrix invertible
    for i in range(len(r_mat_diag)):
      if r_mat_diag(i) == 0:
        r_matrix_mat[i][i] = 1e-12

  p_post = inv(p_prior_mat + inv(r_matrix_mat))
  '''
  print("P: ", p_post)
  print("X: ", x_prior)
  print("Y: ", y_t)
  print("R-1: ", inv(r_matrix_mat) )
  print("x_post_tem: ", x_post_tem)
  print("x_post_temp0: ",x_post_temp0 )
  print(x_post_tem[0][0]* x_post_temp0[0] + x_post_tem[0][1]* x_post_temp0[0])
  '''
  x_post_tem = np.matmul(p_post,inv(r_matrix_mat))
  x_post_temp0 = y_t - x_prior
  # quick fix
  if x_post_temp0.shape == (1,2):
      x_post_temp0 = np.transpose(x_post_temp0)
  x_post_temp = np.matmul(x_post_tem, x_post_temp0)

  x_post = x_prior + x_post_temp

  return np.array(x_post, dtype = np.float32), p_post.reshape(4,1)

# execute the filter based on the input boolean values

def KF(x_prev_est, u_t, del_t, y_t, q_matrix, r_matrix, p_prev, bool_pred, bool_upd):
  if bool_pred and bool_upd : # execute both predication and update steps
    # prediction
    x_prior, p_prior = KFprediction(x_prev_est,u_t, del_t, p_prev, q_matrix)
    # update
    x_post, p_post = KFupdate(x_prev_est, r_matrix, p_prior, y_t, u_t,  del_t)
    return x_post, p_post
  if bool_pred :
    # prediction
    x_prior, p_prior = KFprediction(x_prev_est, u_t, del_t, p_prev, q_matrix)
    return x_prior, p_prior

  if bool_upd :
    # update: x_prev_est <=> x_prior, p_prev <=> p_prior
    x_post, p_post = KFupdate(x_prev_est, r_matrix, p_prev, y_t, u_t)
    return x_post, p_post

def plot_results(x_post,p_post,time,xd_post,pd_post,time_d):

    plt.figure(1)
    plt.plot(time,p_post[:,1],label = 'Covariance, x')
    plt.plot(time,p_post[:,2],label = 'Covariance, y')
    plt.xlabel('Time (sec)')
    plt.ylabel('Covariance')
    plt.title('Velocity')
    plt.grid()
    plt.legend()

    plt.figure(2)
    plt.plot(time_d,pd_post[:],label = 'Covariance')
    plt.xlabel('Time (sec)')
    plt.ylabel('Covariance')
    plt.title('Distance')
    plt.grid()
    plt.show()


def main():
  input_gps_csv_file = "data/gps.csv" # define the file name
  input_acc_csv_file = "data/data.csv"
  input_dist_csv_file = "data/distance.csv"

  # read the file and extract the required data
  lat, lon, time_gps = CSVReadGPS(input_gps_csv_file)
  accX, accY, time_acc = CSVReadAcc(input_acc_csv_file)
  dist, time_ultra = CSVReadUltra(input_dist_csv_file)

  # loop in the lists to estimate the velocities
  num_gps_val = len(lat)# Number of GPS data points captured
  num_acc_val = len(accX)# Number of Accelerometer data points captured
  num_dist_val = len(dist) # number of ultrasound readings collected

  #### Calculate the covariances of data

  # GPS based velocity estimate
  velNorth = []
  velWest = []
  velNorth.append(0) # intial Velocity
  velWest.append(0) # intial Velocity
  for i in range(1,num_gps_val):
    y_t =  GPSVel(time_gps[i], lat[i], lon[i], time_gps[i-1], lat[i-1], lon[i-1])
    velNorth.append(y_t[0])
    velWest.append(y_t[1])

  # Accelerometer based Velocity Estimate
  velX = []
  velY = []
  velX.append(0) # intial Velocity
  velY.append(0) # intial Velocity
  for i in range(1,num_acc_val):
    acc_time_prev = time_acc[i-1]
    acc_time_cur = time_acc[i]

    acc_time_cur_temp = acc_time_cur.split(":")
    acc_time_cur_temp = [float(item) for item in acc_time_cur_temp]
    acc_time_prev_temp = acc_time_prev.split(":")
    acc_time_prev_temp = [float(item) for item in acc_time_prev_temp]
    del_acc_t = (acc_time_cur_temp[0] - acc_time_prev_temp[0])*3600 + (acc_time_cur_temp[1] - acc_time_prev_temp[1])*60 + (acc_time_cur_temp[2] - acc_time_prev_temp[2])

    # Measurements from the accelerometer
    a_t = np.array([2,1],dtype =  np.float32)
    a_t[0] = accX[i]# X-direction
    a_t[1] = accY[i]# Y-direction

    velX.append(a_t[0]*del_acc_t + velX[i-1]) # V = U + a*t
    velY.append(a_t[1]*del_acc_t + velY[i-1])

  cov_gps_velN = np.cov(velNorth)
  cov_gps_velW = np.cov(velWest)
  cov_acc_velX = np.cov(velX)
  cov_acc_velY = np.cov(velY)
  cov_dist = np.cov(dist)

  print("Covariances Calculated")

  # Accelerometer Data acquisition frequency is higher than GPS. GPS: update, Accelerometer: Prediction
  # Intialize the filter parameters:
  # Velocity Estimate
  x_init = np.array([velX[0],velY[0]])
  p_init = np.array([1, 0, 1, 0]) # initialize with high covariance value
  q_matrix = np.array([cov_acc_velX, 0, 0, cov_acc_velY], dtype = np.float32)
  r_matrix = np.array([cov_gps_velN, 0, 0, cov_gps_velW],dtype = np.float32)
  acc_data_count = 1 # counter for data index in accelermeter
  ultra_data_count = 1 # counter for data index in ultrasound prediction step
  # intialize temporary variables
  acc_data_count_temp = 0
  ultra_data_count_temp = 0

  xd_init = dist[0]
  pd_init = 1e10
  r_ultra = cov_dist

  x_prior = np.empty([2,1])
  p_prior = np.empty([2,2])
  xd_prior = 0
  pd_prior = 0

  x_post = np.empty([2,1])
  p_post = np.empty([2,2])
  xd_post = 0
  pd_post = 0

  u_t = np.empty([2,1])
  y_t = np.empty([2,1])

  # store the state vector and covariance matrix
  #x_post_vec = np.empty([num_gps_val,2], dtype = np.float32)
  x1_post_vec = []
  p1_post_vec = []
  x2_post_vec = []
  p2_post_vec = []

  x_post_arr = np.zeros((24,4))
  p_post_arr = np.zeros((24,4))
  time_arr = np.zeros((24,1))

  xd_post_arr = np.zeros((3,1))
  pd_post_arr = np.zeros((3,1))
  d_time_arr = np.zeros((3,1))

  print("Q_matrix: ", q_matrix.reshape(2,2))
  print("R_matrix: ", r_matrix.reshape(2,2))

  # Filter Implementation
  for i in range(1, num_gps_val) :

    y_t = GPSVel(time_gps[i], lat[i], lon[i], time_gps[i-1], lat[i-1], lon[i-1])

    gps_time_cur_temp = time_gps[i].split(":")
    gps_time_cur_temp = [float(item) for item in gps_time_cur_temp]
    gps_time_cur_temp = GMTCDTconv(gps_time_cur_temp)
    gps_time_prev_temp = time_gps[i-1].split(":")
    gps_time_prev_temp = [float(item) for item in gps_time_prev_temp]
    gps_time_prev_temp = GMTCDTconv(gps_time_prev_temp)
    gps_del_t = (gps_time_cur_temp[0] - gps_time_prev_temp[0])*3600 + (gps_time_cur_temp[1] - gps_time_prev_temp[1])*60 + (gps_time_cur_temp[2] - gps_time_prev_temp[2])

    # accumulating previous estimates
    if i == 1:
      x_prev_est = x_init
      p_prev = p_init
    else:
      x_prev_est = x_post
      p_prev = p_post
    acc_data_count = acc_data_count_temp + acc_data_count
    acc_data_count_temp = 0
    # Prediction Steps
    for j in range(acc_data_count, num_acc_val) :
      if acc_data_count_temp > 0 : # Account for multiple prediction steps
        x_prev_est = x_prior
        p_prev = p_prior

      acc_data_count_temp = acc_data_count_temp + 1 # update the counter
      acc_time_cur = time_acc[j]
      acc_time_cur_temp = acc_time_cur.split(":")
      acc_time_cur_temp = [float(item) for item in acc_time_cur_temp]
      time_diff = (acc_time_cur_temp[0] - gps_time_cur_temp[0])*3600 + (acc_time_cur_temp[1] - gps_time_cur_temp[1])*60 + (acc_time_cur_temp[2] - gps_time_cur_temp[2])

      if time_diff < 0: # proceed to the prediction step for all the prediction steps before the GPS readings
        acc_time_prev = time_acc[j-1]
        #time difference
        acc_time_prev_temp = acc_time_prev.split(":")
        acc_time_prev_temp = [float(item) for item in acc_time_prev_temp]
        del_acc_t = (acc_time_cur_temp[0] - acc_time_prev_temp[0])*3600 + (acc_time_cur_temp[1] - acc_time_prev_temp[1])*60 + (acc_time_cur_temp[2] - acc_time_prev_temp[2])

        # Measurements from the accelerometer
        u_t = np.array([2,1],dtype =  np.float32)
        u_t[0] = accX[j]# X-direction
        u_t[1] = accY[j]# Y-direction

        # Kalman Filter Prediction
        x_prior, p_prior = KF(x_prev_est, u_t, del_acc_t, 0, q_matrix, 0, p_prev, True, False)
        acc_data_count = acc_data_count + 1

      else:
        break # end of prediction loop

    # run the update only if there is a prediction steps
    if acc_data_count_temp > 0 :
      # call the KF update
      x_post, p_post = KF(x_prior, u_t, del_acc_t, y_t, 0, r_matrix, p_prior, False, True)

      print("P: ", x_post)

      ######################################################################################################################################################################pyt
      #print(p_post)
      x1_post_vec.append(x_post[0][0])
      p1_post_vec.append(p_post[0])
      x2_post_vec.append(x_post[0][1])
      p2_post_vec.append(p_post[3])

      if i < 24:
          cur_time = (gps_time_cur_temp[0])*3600 + (gps_time_cur_temp[1])*60 + (gps_time_cur_temp[2])
          time_arr[i] = 84600 - cur_time

          x_post_new = x_post.flatten()

          if x_post_new.shape[0] == 2:
              x_post_arr[i,:2] = x_post_new
              p_post_arr[i,:] = p_post.flatten()
          else:
              x_post_arr[i,:] = x_post.flatten()
              p_post_arr[i,:] = p_post.flatten()

      ####
      # Distance Estimation
      if i == 1:
        xd_prev_est = xd_init
        pd_prev = pd_init
        kfd = kalman(pd_init,r_ultra,xd_prev_est, pd_prev) # filter initialization
      else:
        xd_prev_est = xd_post
        pd_prev = pd_post
        kfd = kalman(p_post[3],r_ultra, xd_prev_est, pd_prev) # filter initialization

      # prediction using calculated velocity

      xd_prior, pd_prior = kfd.update_meas(0,gps_del_t,x_post[0][1],False)

      # Update step: iterate over the different measurement data and use the value which is recorded closest to the current GPS time step
      ultra_data_count_temp = 0
      gps_ultra_diff = 0
      for j in range(ultra_data_count + 1, num_dist_val):
        # Time of the measurement
        dis_time_cur = time_ultra[j]
        dis_time_cur_temp = dis_time_cur.split(":")
        dis_time_cur_temp = [float(item) for item in dis_time_cur_temp]
        #print(dis_time_cur_temp)
        #print(gps_time_cur_temp)
        # keep iterating until the difference is positive
        gps_ultra_diff = (dis_time_cur_temp[0] - gps_time_cur_temp[0])*3600 + (dis_time_cur_temp[1] - gps_time_cur_temp[1])*60 + (dis_time_cur_temp[2] - gps_time_cur_temp[2])
        if gps_ultra_diff < 0:
          ultra_data_count_temp = ultra_data_count_temp + 1
        else:
          break; # break the for loop if measurement when ahead of prediction

      # end of for loop
      ultra_data_count_tem = ultra_data_count
      ultra_data_count = ultra_data_count + ultra_data_count_temp - 1

      if ultra_data_count_tem == ultra_data_count :
        continue
      if ultra_data_count_temp > 0: # only perform update if a measurement is detected in the given range
        #print(ultra_data_count)
        xd_post, pd_post = kfd.update_meas(dist[ultra_data_count],gps_del_t, x_post[0][1],True)

        if i < 3:
            time_curr = (gps_time_cur_temp[0])*3600 + (gps_time_cur_temp[1])*60 + (gps_time_cur_temp[2])
            d_time_arr[i] = 84600 - time_curr
            xd_post_arr[i] = xd_post
            pd_post_arr[i] = pd_post

      mass = 1.81 # kg (~ 4 pounds)
      radius = 0.062/2 # meters
      height = 0.033 # meters (bump height)
      wheelbumpdynamics(mass,radius,x_post_arr[0,0],height)
      print('================================================')
    # End of Distance Estimation
    else:
      continue

  # end of for loop
  #print(p1_post_vec)
  #print("R_ultra: ", r_ultra)
  #print(pd_post)
  #print(p2_post_vec)
  plot_results(x_post_arr,p_post_arr,time_arr,xd_post_arr,pd_post_arr,d_time_arr)

if __name__ == "__main__":
    main()
