# Planar motion is assumed

import csv
import math 
import numpy as np

# velocity extraction from GPS data
def GPSVel(time_gps2, lat2, lon2, time_gps1, lat1, lon1):    
      gps_time_cur = time_gps2
      gps_time_prev = time_gps1
  
      #time difference
      gps_time_cur_temp = gps_time_cur.split(":")
      gps_time_cur_temp = [float(item) for item in gps_time_cur_temp]
      gps_time_prev_temp = gps_time_prev.split(":")
      gps_time_prev_temp = [float(item) for item in gps_time_prev_temp]
      del_gps_t = (gps_time_cur_temp[0] - gps_time_prev_temp[0])*3600 + (gps_time_cur_temp[1] - gps_time_prev_temp[1])*60 + (gps_time_cur_temp[2] - gps_time_prev_temp[2])
  
      # measurement extraction
      delta_north = GPSdistanceKm(lat1, 0, lat2, 0)
      vel_north = delta_north/ del_gps_t
      delta_west = GPSdistanceKm(0, lon1, 0, lon2)
      vel_west = delta_west/ del_gps_t
  
      y_t = np.array([2,1],dtype = np.float32)
      y_t[0] = vel_north
      y_t[1] = vel_west
      
      return y_t
      
# distance between two points using DD	
def GPSdistanceKm(lat1, lon1, lat2, lon2):
  deg_rad = 0.017453292519943295    # PI / 180
  dist_temp = 0.5 - math.cos((lat2 - lat1)*deg_rad)/2 + math.cos(lat1*deg_rad) * math.cos(lat2*deg_rad) * (1 - math.cos((lon2 - lon1)*deg_rad))/2
  dist = 12742 * math.asin(math.sqrt(dist_temp)) # 2 * R; R = 6371 km
  return dist

      
#Read GPS CSV document
def CSVReadGPS(csv_file_name):
  with open(csv_file_name) as csv_file:
    csv_reader= csv.reader(csv_file, delimiter = ',', lineterminator = '\n')
    next(csv_file) # skip the header
    # initializing the titles and rows list 
    rows = [] 
    lat = []
    lon = []
    time = []
    
    for row in csv_reader:
       rows.append(row)
       # append the data to the workspace only if data is received from more than 3 satellites
       lat.append(float(row[1])) # converted string to number
       lon.append(float(row[2]))  # converted string to number
       # calculating the time difference between two readings
       array_temp1 = row[0].split("T") # cut the string when "T" detected
       array_temp2 = array_temp1[len(array_temp1)-1].split("Z") # parse the time array stored as a string
       time.append(array_temp2[0]) # store the value of time stamp()
  return lat, lon, time

def GMTCDTconv(gps_time_cur_temp):
  # adjusting time zones for sensor fusion
  gps_time_cur_temp[0] = gps_time_cur_temp[0] + 19
  gps_time_cur_temp[1] = gps_time_cur_temp[1] 
  gps_time_cur_temp[2] = gps_time_cur_temp[2] + 15.788803
  
  if (gps_time_cur_temp[2]/60) > 1:
    gps_time_cur_temp[2] = gps_time_cur_temp[2] % 60
    gps_time_cur_temp[1] = gps_time_cur_temp[1] + 1
    
  return gps_time_cur_temp
  
#Read Accelerometer CSV document
# only accelaration in X direction is recorded
def CSVReadAcc(csv_file_name):
  with open(csv_file_name) as csv_file:
    csv_reader= csv.reader(csv_file, delimiter = ',', lineterminator = '\n')
    # initializing the titles and rows list 
    rows = [] 
    accX = []
    accY = []
    time = []
    
    for row in csv_reader:
       # Data parsing
       # input: date(ax,)(ay,)(az,)(gx,)(gy,)(gz,) 
       t = row[0].split("(")
       ax = t[1]
       ay = row[1].split("(")[1]
       az = row[2].split("(")[1]
       gx = row[3].split("(")[1]
       gy = row[4].split("(")[1]
       gz = row[5].split("(")[1]
       
       
       rows.append(row)
       # append the data to the workspace only if data is received from more than 3 satellites
       accX.append(float(ax) - float(gx)) # account for gravity vector. converted string to number
       accY.append(float(ay) - float(gy))  # account for gravity vector. converted string to number
       # calculating the time difference between two readings
       array_temp1 = t[0].split("T") # cut the string when "T" detected
       time.append(array_temp1[1]) # store the value of time stamp()    
  return accX, accY, time

 
# Read Ultrasound CSV document
# only accelaration in X direction is recorded
def CSVReadUltra(csv_file_name):
  with open(csv_file_name) as csv_file:
    csv_reader= csv.reader(csv_file, lineterminator = '\n')
    # initializing the titles and rows list 
    rows = [] 
    dist = []
    time = []
    
    for row in csv_reader:
      row = row[0].split(" ")
      rows.append(row)
      # append the data to the workspace only if data is received from more than 3 satellites
      dist.append(float(row[1])) # converted string to number
      # calculating the time difference between two readings
      array_temp = row[0].split("T") # cut the string when "T" detected
      time.append(array_temp[1]) # store the value of time stamp()
  return dist, time