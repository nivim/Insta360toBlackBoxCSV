#Originaly created by g_gantic, mess was added by NiViM

# For the simple use case just run process to import a SMO4K video and export a CSV

import numpy as np
import struct
import pandas as pd
from scipy import signal



insvDataLen = {
    '0x300':56,    # accelerometer
    '0x400':16,    # exposure (ref 6)
    '0x600':8,     # timestamps (ref 6)
    '0x700':53,    # GPS
}
# Extract insta360 accelerometer data.
# Data is just added at the end of the file, it needs to be 
# read from the end to get each insta360 record size.
def _readAdjusted(fin):

	# last 32 chars = 8db42d694ccc418790edff439fe026bf (magic number)
	# then trailer length: 
	fin.seek(-(32+42+4),2)
	buf = fin.read(38+4)
	trailer_len = struct.unpack('<38xL', buf)[0] # 42 bytes


	offset = -78
	while offset > -trailer_len:
		# Iterates over each insta360 trailer records from the end.
		# Record ID and size are written after the actual data.
		fin.seek(offset,2)
		buf = fin.read(2+4)
		id, size = struct.unpack('<HL', buf)
		hid = hex(id)
		fin.seek(offset-size,2) # Go to the begining of the actual data.
		#if hid == '0x900': # Acelerometer data.
			#parseExpRecord(fin, size)
		if hid == '0x300': # Acelerometer data.
			print("Found insta360 accelerometer data: " + hid)
			d_times,d_gyros = parseAccRecord(fin, size)
			#return np.array(d_times), np.array(d_gyros) 
		if hid == '0x400': # exposure data
			print("Found insta360 exposure data: " + hid)
			expo_time = parseExpRecord(fin, size)
		#if hid == '0x101': # insv maker notes
			#parseNotes(fin, size)

			# 101, 200, 300, 400, 900 a00
		offset = offset - size - 4 - 2
	# print(expo_time)
	for i in range(0, len(d_times), 1):
		# print(d_times[i]-expo_time)
		if (d_times[i]-expo_time)>0:
			gyro_sync_item = i+400
			break
	print('Sync point:', gyro_sync_item)
	print(d_times[0])
	# TEST
	# gyro_sync_item = 690
	print('Current sync time:', d_times[gyro_sync_item])
	
	d_times = np.delete(d_times, range(0, gyro_sync_item, 1))
	d_gyros = np.delete(d_gyros, range(0, gyro_sync_item, 1), axis=0)
	start_time = d_times[0]
	for j in range(0, len(d_times), 1):
		d_times[j] = float((d_times[j] - start_time))
	print('In Read after leveling:', len(d_times), len(d_gyros))
	# print(d_gyros)
	return d_times,d_gyros


def parseExpRecord(fin, size):  # first at 5.688
	expo = []
	dlen = insvDataLen['0x400']
	# csv = open("exposure_out.csv", 'w')
	# csv.write('"time","exposure"\n')
	for i in range(0, size, dlen):
		buf = fin.read(dlen)
		timecode, exp = struct.unpack("<Qd", buf)
		tm = float(timecode/1000)
		expo.append((tm))
		#print("timecode={} exp={}".format(tm, exp))
		# csv.write("{},{}\n".format(tm , exp))
	# csv.close()
	return expo[1]

# New
def parseAccRecord(fin, size):  # first at 5.672
	dlen = insvDataLen['0x300']
	# csv = open("gyro_out.csv", 'w')
	# csv.write('"time","gyroADC[0]","gyroADC[1]","gyroADC[2]"\n')
	# time_at_arm = None
	d_times = []
	d_gyros = []
	for i in range(0, size, dlen):
		buf = fin.read(dlen)
		# normally it's  roll > pitch > yaw, insta360 is pitch > yaw > roll
		timecode, _, _, _, gyroPitch, gyroYaw, gyroRoll = struct.unpack("<Q6d", buf)
		# if time_at_arm is None:
		# 	time_at_arm = timecode
		tm = float(timecode/1000)
		#print("{} {} {} {}".format( tm, -gyroRoll, gyroPitch, -gyroYaw))
		d_times.append(tm)
		# Roll and yow are mirrored, pitch is fine.
		d_gyros.append((gyroRoll, gyroPitch, gyroYaw))
		# csv.write("{},{},{},{}\n".format(tm , -gyroRoll, gyroPitch,  -gyroYaw))
	#print("Found 0x300 Insta360 Accelerometer data: {} records".format(len(d_gyros)))
	#d_times, d_gyros = trim_to_video_length(fin, d_times, d_gyros)
	#print("data adjusted: {}".format(len(d_gyros)))
	# csv.close()
	return np.array(d_times), np.array(d_gyros)
# Extract insta360 accelerometer data.
# Data is just added at the end of the file, it needs to be 
# read from the end to get each insta360 record size.

def _read(fin):
	# last 32 chars = 8db42d694ccc418790edff439fe026bf (magic number)
	# then trailer length: 
	fin.seek(-(32+42+4),2)
	buf = fin.read(38+4)
	trailer_len = struct.unpack('<38xL', buf)[0]

	offset = -78
	while offset > -trailer_len:
		# Iterates over each insta360 trailer records from the end.
		# Record ID and size are written after the actual data.
		fin.seek(offset,2)
		buf = fin.read(2+4)
		id, size = struct.unpack('<HL', buf)
		hid = hex(id)
		if hid == '0x300': # Acelerometer data.
			fin.seek(offset-size,2) # Go to the begining of the actual data.
			d_times, d_gyros = parseAccRecord(fin, size)
			# d_gyros.insert(0, (0,0,0))
			# d_gyros.append((0,0,0))
			# d_times.insert(0, d_times[0]-.0001)
			# d_times.append(d_times[-1]+.0001)
			return np.array(d_times),np.array(d_gyros) 

		offset = offset - size - 4 - 2

	return false

def parseAccRecord_old(fin, size):
	dlen = insvDataLen['0x300']
	time_at_arm = None
	d_times = []
	d_gyros = []
	cnt = 0
	for i in range(0, size, dlen):
		cnt = cnt + 1
		buf = fin.read(dlen)
		if cnt > 500:
			# normally it's  roll > pitch > yaw, insta360 is pitch > yaw > roll
			timecode, _, _, _, gyroPitch, gyroYaw, gyroRoll = struct.unpack("<Q6d", buf)
			if time_at_arm is None:
				time_at_arm = timecode
			tm = float((timecode-time_at_arm)/1000)
			d_times.append(tm)
			d_gyros.append((gyroRoll, gyroPitch, gyroYaw))
	print("Found 0x300 Insta360 Accelerometer data: {} records".format(len(d_gyros)))
	return d_times, d_gyros

def accRecordToPandas(gyro_data_array):
    # Covert to pandas format with BlackBox labels
	# print('NEW')
	# print(len(gyro_data_array[1]), len(gyro_data_array[0]))
	gyro_df = pd.DataFrame(data=gyro_data_array[1], index=gyro_data_array[0])
	gyro_df.columns = ['gyroADC[0]','gyroADC[1]', 'gyroADC[2]']
	# gyro_df.columns = ['gyroADC[0]','gyroADC[1]', 'gyroADC[2]', 'accSmooth[0]', 'accSmooth[1]', 'accSmooth[2]']
	gyro_df.index.name = 'time'
	return gyro_df

def _SMO4kCorrection(gyro_df):
    # Change data to fit BlackBox CSV
    gyro_df['gyroADC[0]']=gyro_df['gyroADC[0]']*-1
    gyro_df['gyroADC[2]']=gyro_df['gyroADC[2]']*-1
    return gyro_df

def _SMOfile(path):
	fin = open(str(path), "rb")

	times, gyro = (_readAdjusted(fin))
	print(len(times), len(gyro))
	
	
	
	gyro_data = accRecordToPandas(_readAdjusted(fin))
	fin.close()
	return gyro_data

def loadSMO4KFile(path):
    # Use this to load SMO4K file
    return _SMO4kCorrection(_SMOfile(path))

def loadBboxFile(path, cameraAngle):
	#use this to load BlackBox file - needs bbox.py from BlackBoxToGPMF 
	import bbox
	fin = open(str(path), "r")
	gyro_data = accRecordToPandas(bbox.read(fin, cameraAngle))
	fin.close()
	return gyro_data
     
def _Columnfilter(df_column, orderOfFilter, criticalFrequency ):
	b, a = signal.butter(orderOfFilter, criticalFrequency)
    # b, a = signal.butter(3, 0.05)
    # b, a = signal.butter(orderOfFilter, criticalFrequency)
	zi = signal.lfilter_zi(b, a)
	d_column = df_column.to_numpy()
	z, _ = signal.lfilter(b, a, d_column, zi=zi*d_column[0])
	z2, _ = signal.lfilter(b, a, z, zi=zi*z[0])
	return signal.filtfilt(b, a, d_column)

def filterDataframe(df, orderOfFilters=10, criticalFrequencys=0.2):
	return df[:].apply(_Columnfilter, orderOfFilter=orderOfFilters,criticalFrequency=criticalFrequencys)

def cross_corr(y1, y2):
    """Copy paste from Stackoverflow - https://stackoverflow.com/questions/39336727/get-lag-with-cross-correlation
	As far as I played with it doesn't work"""

    """Calculates the cross correlation and lags without normalization.

    The definition of the discrete cross-correlation is in:
    https://www.mathworks.com/help/matlab/ref/xcorr.html

    Args:
        y1, y2: Should have the same length.

    Returns:
        max_corr: Maximum correlation without normalization.
        lag: The lag in terms of the index.
    """
    if len(y1) != len(y2):
        raise ValueError('The lengths of the inputs should be the same.')

    y1_auto_corr = np.dot(y1, y1) / len(y1)
    y2_auto_corr = np.dot(y2, y2) / len(y1)
    corr = np.correlate(y1, y2, mode='same')
    # The unbiased sample size is N - lag.
    unbiased_sample_size = np.correlate(
        np.ones(len(y1)), np.ones(len(y1)), mode='same')
    corr = corr / unbiased_sample_size / np.sqrt(y1_auto_corr * y2_auto_corr)
    shift = len(y1) // 2

    max_corr = np.max(corr)
    argmax_corr = np.argmax(corr)
    return max_corr, argmax_corr - shift

def resample(df, sampleRateInMillisec=2):
    # Downsample the gyro Dataframe, 2 will output 500Hz
	SampleMS = str(sampleRateInMillisec)+'L'
	downDF = df.copy()
	downDF['gyroTime'] = downDF.index
	downDF.index = pd.to_datetime(downDF.index,unit='s')
	downDF = downDF.resample(SampleMS).pad()
	# downDF = downDF.resample(SampleMS).mean()
	downDF['newTime']=downDF.index
	downDF.newTime = (downDF.newTime - pd.Timestamp("1970-01-01")) / pd.Timedelta("1s")
	# print(downDF.newTime)
	# downDF = downDF.set_index('gyroTime')
	# downDF = downDF.drop(columns=['newTime'])
	# downDF.index.names = ['time']

	downDF = downDF.set_index('newTime')
	downDF = downDF.drop(columns=['gyroTime'])
	downDF.index.names = ['time']

	return downDF


def plotSingle(df):
	import matplotlib.pyplot as plt
	# Plot single Dataframe
	df.plot()
	plt.show()

def plotTwo(df1, df2):
	import matplotlib.pyplot as plt
	# Plot ywo Dataframes
	ax = df1.plot()
	df2.plot(ax=ax)
	plt.show()

def shiftData(df, shift_in_secs, gyro_freq_hz):
	#In most cases the shift will be that the video file started before the BlackBox, in that case the shift in secs needs to be negative! 
	points = shift_in_secs*gyro_freq_hz
	return df.shift(int(points)).fillna(0)

def export(df, fullPath):
	# Exports the file...
	df.to_csv(str(fullPath))

def exportToBBoxFormat(df, fullPath):
	# Exports the file...
	df.index = df.index*1e6
	df[['gyroADC[0]', 'gyroADC[1]', 'gyroADC[2]']] = df[['gyroADC[0]', 'gyroADC[1]', 'gyroADC[2]']] * (180/np.pi)
	export(df, fullPath)

def process(video_path, export_path, filter_df=False, orderOfFilters=10, criticalFrequencys=0.2):
	# Loads and exports as CSV
	df = loadSMO4KFile(video_path)
	if filter_df:
		filterDataframe(df, orderOfFilters, criticalFrequencys)
	exportToBBoxFormat(df, export_path)
	

