import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.weights_input_hidden = np.zeros((hidden_size, input_size))
        self.weights_hidden_output = np.zeros((output_size, hidden_size))

    def load_weights(self, filename):
        weights = np.loadtxt(filename)
        self.weights_input_hidden = weights[:self.hidden_size * self.input_size].reshape(self.hidden_size, self.input_size)
        self.weights_hidden_output = weights[self.hidden_size * self.input_size:].reshape(self.output_size, self.hidden_size)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, inputs):
        hidden_layer_input = np.dot(self.weights_input_hidden, inputs)
        hidden_layer_output = self.sigmoid(hidden_layer_input)
        output_layer_input = np.dot(self.weights_hidden_output, hidden_layer_output)
        output_layer_output = self.sigmoid(output_layer_input)
        return output_layer_output

# 創建神經網絡
input_size = 27
hidden_size = 10
output_size = 3
nn = NeuralNetwork(input_size, hidden_size, output_size)

# 從文件加載權重
nn.load_weights('matrix_weight.txt')

# 隨機生成輸入
inputs = np.random.rand(input_size)

# 前向傳播
outputs = nn.forward(inputs)

# 輸出結果
print("Inputs:")
print(inputs)
print("Outputs:")
print(outputs)