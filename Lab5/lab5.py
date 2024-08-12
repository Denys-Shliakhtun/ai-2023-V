def hebbian_network(letters, expected_result, neurons_number, max_iter):
    for index in range(neurons_number):
       letters[index] = [1] + letters[index]

    weights = [[0] * len(letters[0]) for _ in range(neurons_number)]
    
    for _ in range(max_iter):
       for i_letter in range(neurons_number):
           for i_neuron in range(neurons_number):
               for i_weight in range(len(weights[i_neuron])):
                   # w_ji (new) = w_ji (old) + x_j * y_i
                   weights[i_neuron][i_weight] += letters[i_letter][i_weight] * expected_result[i_letter][i_neuron]
    
    actual_result = letter_recognition(letters, weights, neurons_number)
    
    if actual_result == expected_result:
       return weights
    
    raise Exception('There is unsolvable problem of weight adaptation. Weights: ' + str(weights))


def letter_recognition(letters, weights, neurons_number):
    result = []
    for i_letter in range(len(letters)):
       letter_result = []
       for i_neuron in range(neurons_number):
           s = 0
           for i_weight in range(len(weights[i_neuron])):
               s += weights[i_neuron][i_weight] * letters[i_letter][i_weight]
           if s > 0:
               letter_result += [1]
           else:
               letter_result += [-1]
       result += [letter_result]
    return result

D = [ 1,  1, -1,
      1, -1,  1,
      1,  1, -1]
Y = [ 1, -1,  1,
     -1,  1, -1,
     -1,  1, -1]
L = [ 1, -1, -1,
      1, -1, -1,
      1,  1,  1]
A = [-1,  1, -1,
      1,  1,  1,
      1, -1,  1]

expected_result = [[ 1, -1, -1, -1],
                   [-1,  1, -1, -1],
                   [-1, -1,  1, -1],
                   [-1, -1, -1,  1]]

train_data = [D, Y, L, A]
number_of_neurons = len(train_data)

weights = hebbian_network(train_data, expected_result, number_of_neurons, 1000)
print("Weights:")
[print(weights[i]) for i in range(len(weights))]

D_mistake = [ 1,  1, -1,
             -1, -1,  1,
             -1,  1, -1] 
Y_mistake = [-1, -1,  1,
             -1,  1, -1,
             -1,  1, -1] 

test_data = [D, Y, L, A, D_mistake, Y_mistake]
test_data = [[1] + test_data[i] for i in range(len(test_data))]

actual_result = letter_recognition(test_data, weights, number_of_neurons)

print("\nResult (D, Y, L, A, D with mistake, Y with mistake):", end="")
letters = ["D", "Y", "L", "A"]
for res in actual_result:
    for i in range(4):
        if res[i] == 1:
            print(" "+letters[i], end="")
    print(",", end="")
print("\n",actual_result)