# Fashion-MNIST Neural Network from Scratch

Feedforward neural network built using only NumPy
for classifying Fashion-MNIST into 10 classes.
No deep learning frameworks were used for the
core implementation.

## Project Structure

notebooks/
└── experiments.ipynb  ← All code is here

## Requirements

pip install numpy pandas matplotlib wandb 
tensorflow seaborn scikit-learn

## How to Run

1. Open notebooks/experiments.ipynb in Google Colab
2. Run all cells from top to bottom
3. Login to WandB when prompted

## How to Train

parameters = initialize_parameters(
    input_size    = 784,
    hidden_layers = [128, 128, 128, 128, 128],
    output_size   = 10,
    init_type     = "xavier"
)

# Train with adam, lr=0.001, relu, 10 epochs

## How to Evaluate

test_output, _ = forward_propagation(
    X_test_flat, parameters, activation="relu"
)
y_pred    = np.argmax(test_output.T, axis=1)
test_acc  = np.mean(y_pred == y_test)
print(f"Test Accuracy: {test_acc*100:.2f}%")

## How to Run WandB Sweep

sweep_id = wandb.sweep(
    sweep_config, 
    project="fashion-mnist-atri"
)
wandb.agent(sweep_id, function=train_wandb, count=50)

## Optimizers Supported

| Optimizer  | Description                    |
|------------|--------------------------------|
| SGD        | Basic gradient descent         |
| Momentum   | SGD with momentum term         |
| Nesterov   | Lookahead gradient descent     |
| RMSprop    | Adaptive learning rate         |
| Adam       | Momentum + RMSprop combined    |
| Nadam      | Adam + Nesterov combined       |

## Best Results

| Config                        | Val Accuracy |
|-------------------------------|-------------|
| adam + relu + xavier (best)   | 88.9%       |
| nadam + relu + xavier         | 88.7%       |
| sgd + tanh + xavier           | 83.2%       |
| momentum + random init        | 9.5% (fail) |

## Key Findings

- Adam and Nadam gave best results consistently
- Xavier initialization is critical for deep networks
- ReLU outperformed sigmoid and tanh
- weight_decay=0.0005 was the sweet spot
- Cross entropy converges faster than MSE loss

## WandB Report

View all 57 experiments, hyperparameter sweep results,
parallel coordinates analysis and loss comparison here:

[Experiment Report](https://api.wandb.ai/links/vaishalinir-ymc2022-chennai-institute-of-technology/uvl153uy)
