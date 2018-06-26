import numpy as np
import idam
idam.setHost("idam")

target_shot = 16161
shot_range = np.arange(241) + 16113
time = 0.1
plotting = True

traces = [#"ada_dalpha integrated",
	  #"amc_plasma current",
	  #"ane_density",
	  "efm_magnetic_axis_z"#,
	  #"efm_q_95",
	  #"efm_bphi_rmag",
	  #"anb_sw_sum_power"#,
	  #"anb_ss_sum_power"
	 ]
tols = [0.3,0.1,0.1,0.1]#,0.1,0.2]#,0.2]


target_trace_data = []
for trace in traces:
	target_trace_data.append(idam.Data(trace,target_shot))

comparable_shots = []
comparable_shots_data = {}
for shot in shot_range:
	test = True
	for i in np.arange(len(traces)):
		trace = traces[i]
		try:
			
			data = idam.Data(trace,shot)
			tind = (np.abs(data.time - time)).argmin()
			tind_target = (np.abs(target_trace_data[i].time - time)).argmin()	
			test = test and (np.abs(data.data[tind] - target_trace_data[i].data[tind_target])/np.max([np.max(np.abs(data.data[tind])),np.max(np.abs(target_trace_data[i].data[tind_target]))])) < tols[i]
		except:
			test = False
			pass

	if test and shot != target_shot:
		comparable_shots.append(shot)
  		if plotting:
			comparable_shots_data[str(shot)] = []
  			for trace in traces:
				comparable_shots_data[str(shot)].append(idam.Data(trace,shot))

 

print "Comparable shots to shot number "+str(target_shot)+" are "+str(comparable_shots)

if plotting:
	import matplotlib.pyplot as plt
	for i in np.arange(len(traces)):
		plt.plot(target_trace_data[i].time,target_trace_data[i].data,label=str(target_shot))
		for shot in comparable_shots:
			plt.plot(comparable_shots_data[str(shot)][i].time,comparable_shots_data[str(shot)][i].data,label=str(shot))
		plt.title(traces[i])
		plt.xlim(0,1.0)
		plt.legend()
		plt.show()

		

