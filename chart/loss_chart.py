import matplotlib.pyplot as plt

# Function to read the data file and extract the second values
def read_data(file_path):
    second_values = []
    with open(file_path, 'r') as file:
        for line in file:
            values = line.split()
            if len(values) > 1:
                second_values.append(float(values[1]))  # Append the second value
    return second_values

# File path to the data file (you can change it to the path of your file)
client = 3
file_path = f'./original_loss.text'

# Read the second values from the file
second_values = read_data(file_path)

# Create a plot of the second values
plt.plot(second_values)
plt.title(f'Plot of Loss Values in CNN')
plt.xlabel('Index')
plt.ylabel('Loss')
plt.grid(True)
plt.show()
