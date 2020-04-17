import pickle
import glob

dumps = glob.glob("data_*.pickle")
data = dict()

for dump in dumps:
	# don't open the previous data 
	if dump != 'data_dump.pickle':
		with open(dump, 'rb') as f:
			parameter = dump[5:-7]
			data[parameter] = pickle.load(f)

# dump all to a single file
with open('data_dump.pickle', 'wb') as f:
    pickle.dump(data, f)