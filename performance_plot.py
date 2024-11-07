import matplotlib.pyplot as plt

# skipping 50x25x10x4 for now b/c not sure about its training time
architectures = [ 
    "50x50x4",
    "50x20x4",
    "50x50x10x4",
    "50x50x10x4 (no ReLU)",
    "20x64x32x4"
]

training_times = [
    209.98,
    251.66,
    250.18,
    245.14,
    250.36
]

avg_losses = [
    0.2968025193359936,
    0.3921664776731049,
    0.214894094788935,
    0.4624589592027477,
    0.23489890832826504
]

# training time
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.barh(architectures, training_times, color='lightblue')
plt.xlabel('Training Time (seconds)')
plt.title('Training Time for Different Model Architectures')

# average loss
plt.subplot(1, 2, 2)
plt.barh(architectures, avg_losses, color='sandybrown')
plt.xlabel('Average Loss')
plt.title('Average Loss for Different Model Architectures')

plt.tight_layout()
plt.show()
