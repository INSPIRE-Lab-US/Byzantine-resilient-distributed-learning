import numpy as np
import pickle
import matplotlib.pyplot as plt

dgd_b0 = []
dgd_b2 = []

byrdie_b2_faultless = []
byrdie_b2 = []

bridge_b2_faultless = []
bridge_b2 = []

median_b2_faultless = []
median_b2 = []

krum_b2_faultless = []
krum_b2 = []

krum_b4_faultless = []
krum_b4 = []

krum_b3_faultless = []
krum_b3 = []



for monte in range(10):
    with open(f'./result/DGD/result_DGD_b0_faultless_{monte}.pickle', 'rb') as handle:
        dgd_b0.append(pickle.load(handle))
    with open(f'./result/DGD/result_DGD_b2_{monte}.pickle', 'rb') as handle:
        dgd_b2.append(pickle.load(handle))
    
    with open(f'./result/ByRDiE/result_ByRDiE_b2_faultless_{monte}.pickle', 'rb') as handle:
        byrdie_b2_faultless.append(pickle.load(handle))
    with open(f'./result/ByRDiE/result_ByRDiE_b2_{monte}.pickle', 'rb') as handle:
        byrdie_b2.append(pickle.load(handle))
    
    with open(f'./result/BRIDGE/result_BRIDGE_b2_faultless_{monte}.pickle','rb') as handle:
        bridge_b2_faultless.append(pickle.load(handle))
    with open(f'./result/BRIDGE/result_BRIDGE_b2_{monte}.pickle','rb') as handle:
        bridge_b2.append(pickle.load(handle))
    
    with open(f'./result/Median/result_Median_b2_faultless_{monte}.pickle','rb') as handle:
        median_b2_faultless.append(pickle.load(handle))
    with open(f'./result/Median/result_Median_b2_{monte}.pickle','rb') as handle:
        median_b2.append(pickle.load(handle))

    with open(f'./result/Krum/result_Krum_b2_faultless_{monte}.pickle','rb') as handle:
        krum_b2_faultless.append(pickle.load(handle))
    with open(f'./result/Krum/result_Krum_b2_{monte}.pickle','rb') as handle:
        krum_b2.append(pickle.load(handle))

smooth_dgd_b0 = np.mean(dgd_b0, axis=0)
smooth_dgd_b2 = np.mean(dgd_b2, axis=0)

smooth_byrdie_b2_faultless = np.mean(byrdie_b2_faultless, axis=0)
smooth_byrdie_b2 = np.mean(byrdie_b2, axis=0)
smooth_byrdie_b2_FL = np.mean(smooth_byrdie_b2_faultless, axis=1)
smooth_byrdie_b2 = np.mean(smooth_byrdie_b2, axis=1)

smooth_bridge_b2_faultless = np.mean(bridge_b2_faultless, axis=0)
smooth_bridge_b2 = np.mean(bridge_b2, axis=0)

smooth_median_b2_faultless = np.mean(median_b2_faultless, axis=0)
smooth_median_b2 = np.mean(median_b2, axis=0)

smooth_krum_b2_faultless = np.mean(krum_b2_faultless, axis=0)
smooth_krum_b2 = np.mean(krum_b2, axis=0)

scalar_comms = [7840*n for n in range(100)]

byrdie_axis = []
for t in range(100):
    for p in range(39):
        byrdie_axis.append(t * 7840 + (p+1) * 200)
    for p in range(10, 11):
        byrdie_axis.append((t+1) * 7840 + p)

plot_faultless = plt.figure(figsize=(15,6))
plt.subplot(1,2,1)
plt.plot(scalar_comms, smooth_dgd_b0*100, markevery=5, marker='v')
plt.plot(byrdie_axis[:3960], smooth_byrdie_b2_FL*100, markevery=200, marker='.')
plt.plot(scalar_comms, smooth_bridge_b2_faultless*100, markevery=5, marker='p', color='g')
plt.plot(scalar_comms, smooth_median_b2_faultless*100, markevery=5, marker='s', color='r')
plt.plot(scalar_comms, smooth_krum_b2_faultless*100, markevery=5, marker='s', color='m')


plt.ylim((5,90))
plt.ylabel('Average classification accuracy (%)')
plt.xlabel('Number of scalar Broadcasts per node')
plt.title('Faultless setting')
plt.legend(['DGD','ByRDiE','BRIDGE','Median','Krum'], loc='right')


plt.subplot(1,2,2)
plt.plot(scalar_comms, smooth_dgd_b2*100, markevery=5, marker='v')
plt.plot(byrdie_axis[:3960], smooth_byrdie_b2*100, markevery=200, marker='.')
plt.plot(scalar_comms, smooth_bridge_b2*100, markevery=5, marker='p', color='g')
plt.plot(scalar_comms, smooth_median_b2*100, markevery=5, marker='s', color='r')
plt.plot(scalar_comms, smooth_krum_b2*100, markevery=5, marker='s', color='m')

plt.ylim((5,90))
plt.ylabel('Average classification accuracy (%)')
plt.xlabel('Number of scalar Broadcasts per node')
plt.title('Faulty setting')
plt.legend(['DGD','ByRDiE','BRIDGE','Median','Krum'], loc='right')
plt.savefig('./result/plot_dec.png', bboxinches='tight')