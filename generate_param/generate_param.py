import numpy as np
import sys

upper_tau = int(sys.argv[1])

path = "/cicutagroup/yz655/cuda_run/"

upper_scale = 10   # 2^10
lower_scale = 5

# upper_q = 40   # m-1
# lower_q = 16
# q_number = upper_q-lower_q+1

upper_lambda = 32   # m-1
lower_lamda = 8
lambda_number = int((upper_lambda-lower_lamda)/4+1)

# upper_tau = 100  # frame number. tau count should be less than 1/3 of the total video frame
lower_tau = 1
tau_gap = 1

with open(path+"lambda.txt", "w") as f:
    # calculate lambda from q
    # q_list = np.linspace(lower_q, upper_q, num=q_number)
    # lambda_list = 2**upper_scale*np.reciprocal(q_list)
    # lambda_list = np.flip(lambda_list)
    
    # assign lambda directly
    lambda_list = np.linspace(lower_lamda, upper_lambda, num=lambda_number)

    for l in lambda_list:
        f.write("%.2f" % l+'\n')

# The largest lambda-vector should be smaller than the smallest scale.
with open(path+"scale.txt", "w") as f:
    scale = upper_scale
    while 2**scale>=lambda_list[-1] and scale>=lower_scale:
    # while scale>=lower_scale:
        f.write(str(2**scale)+'\n')
        scale-=1
    
with open(path+"tau.txt", "w") as f:
    while upper_tau>=lower_tau:
        f.write(str(lower_tau)+'\n')
        lower_tau+=tau_gap

