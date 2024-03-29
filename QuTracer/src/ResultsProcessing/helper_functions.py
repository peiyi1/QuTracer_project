import numpy as np
def norm_dict(d):
    # Assuming the dictionary values are numeric
    # Calculate the sum of all values
    total = sum(d.values())
    # Create a new dictionary to store the normalized values
    normalized = {}
    # Loop through the original dictionary
    for key, value in d.items():
        # Divide each value by the total and store it in the new dictionary
        normalized[key] = value / total
    # Return the normalized dictionary
    return normalized

def two_bit_weight(dist, index):
    weight_list = [0] * 4 #initialize a list of zeros with length 4
    for key in dist.keys():
        bit_index=0
        if (len(key) - 2 - index)>=0:
            bit_index = int(key[len(key) - 2 - index : len(key) - index], 2)
        else:
            bits= key[:len(key) - index] + key[len(key) - 2 - index:] 
            bit_index = int(bits, 2)
        #add the value to the corresponding element in the list
        weight_list[bit_index] += dist[key]
    return weight_list

def update_dist(unmiti_dist, miti_dist, index, index_for_miti_dist=0):
    Ppost = {}
    w_weights = two_bit_weight(miti_dist, index_for_miti_dist)
    u_weights = two_bit_weight(unmiti_dist, index)
    
    # Ensure that none of the weights are exactly 0
    w_weights = [max(weight, 0.0000000000001) for weight in w_weights]
    u_weights = [max(weight, 0.0000000000001) for weight in u_weights]
    
    for key in unmiti_dist.keys():
        bits_index=0
        if(len(key) - 2 - index)>=0:
            bits_index = int(key[len(key) - 2 - index : len(key) - index], 2)
        else:
            bits= key[:len(key) - index] + key[len(key) - 2 - index:] 
            bits_index = int(bits, 2)
        # Calculate the updated probability based on the weights
        if bits_index < len(w_weights) and bits_index < len(u_weights):
            Ppost[key] = unmiti_dist[key] / u_weights[bits_index] * w_weights[bits_index]
        else:
            print("Incorrect bits_index value")

    return Ppost
def combine_dist(orign_dist, dist_list):
    output_dist = {}
    for key in orign_dist:
        value = orign_dist[key]
        for dist in dist_list:
            value += dist[key]
        output_dist[key] = value
    return output_dist
def total_counts(dictionary):
    total = 0
    for value in dictionary.values():
        total += value
    return total
def norm_dict(dictionary):
    total = total_counts(dictionary)
    norm_dist = {}
    for i in dictionary.keys():
        norm_dist[i] = dictionary[i]/total
    return norm_dist

def H_distance_dict(p, q):
    # distance between p an d
    # p and q are np array probability distributions
    sum = 0.0
    for key in p.keys():
        sum += (np.sqrt(p[key]) - np.sqrt(q[key]))**2
    result = (1.0 / np.sqrt(2.0)) * np.sqrt(sum)
    return result

def bayesian_reconstruct(unmiti_dist, miti_dist_list, threshold = 0.0001):
    temp_dist = unmiti_dist.copy()
    h_dist = 1
    while h_dist > threshold:
        temp_dist_start = temp_dist.copy()
        ppost = [0] * len(miti_dist_list)
        for i in range(0, len(miti_dist_list)):
            ppost[i] = update_dist(temp_dist, miti_dist_list[i][0], miti_dist_list[i][1], miti_dist_list[i][2])
        #print(ppost)
        #print(len(ppost))
        temp_dist = combine_dist(temp_dist, ppost)
        temp_dist = norm_dict(temp_dist)
        h_dist = H_distance_dict(temp_dist, temp_dist_start)
        #h_dist=0.0001
        #print("H-dist:", h_dist)
    return temp_dist
