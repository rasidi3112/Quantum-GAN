
"""
Quantum GAN – Variational quantum generator (PennyLane) + classical discriminator (PyTorch)
Visualisasi dan inference tambahan menggunakan Qiskit Aer simulator.
Seluruh bagian kode diberi komentar untuk menjelaskan logika quantum & ML.
"""

import math     # type: ignore          ````
import numpy as np # type: ignore       
import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.optim as optim # type: ignore
import pennylane as qml # type: ignore
import matplotlib.pyplot as plt # type: ignore

# -------------------------------------------
# Opsi: gunakan Qiskit Aer untuk inferensi/noise
# -------------------------------------------
try:
    from qiskit import QuantumCircuit # type: ignore
    from qiskit.quantum_info import Statevector, SparsePauliOp # type: ignore
    from qiskit_aer import AerSimulator # type: ignore
    from qiskit.providers.aer.noise import NoiseModel, depolarizing_error # type: ignore
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False

# -------------------------------------------
# Konfigurasi umum & reproduksibilitas
# -------------------------------------------
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_default_dtype(torch.float32)  # memastikan konsistensi tipe data dengan PennyLane
device_torch = torch.device("cpu")      # semua komputasi di CPU (simulator kuantum)

# -------------------------------------------
# Dataset sintetis: lingkaran noisy di 2D
# -------------------------------------------
def sample_real_data(n_samples: int) -> torch.Tensor:
    """
    Menghasilkan data target 2D berbentuk lingkaran dengan sedikit noise gaussian.
    Digunakan sebagai "real data" untuk QGAN.
    """
    angles = 2 * math.pi * torch.rand(n_samples)
    radius = 1.0 + 0.05 * torch.randn(n_samples)
    x = radius * torch.cos(angles)
    y = radius * torch.sin(angles)
    return torch.stack((x, y), dim=1)

# -------------------------------------------
# Sampel laten (input acak ke generator kuantum)
# -------------------------------------------
def sample_latent(batch_size: int, latent_dim: int) -> torch.Tensor:
    """
    Mengambil vektor laten uniform di [0, 2π] untuk memutar qubit via gerbang rotasi.
    """
    return 2 * math.pi * torch.rand(batch_size, latent_dim)

# -------------------------------------------
# Parameter generator kuantum
# -------------------------------------------
n_qubits = 2          # dua qubit -> dua nilai ekspektasi -> ruang data 2 dimensi
latent_dim = n_qubits
n_layers = 3          # kedalaman VQC (jumlah lapisan variational)
scale_factor = 1.2    # faktor penskalaan output ke ruang data target

# PennyLane device (simulator ideal)
quantum_dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(quantum_dev, interface="torch")
def generator_qnode(latent_angles: torch.Tensor, weights: torch.Tensor):
    """
    QNode generator:
    1. Embed vektor laten sebagai rotasi Y/Z pada masing-masing qubit.
    2. Terapkan lapisan variational (RX-RY-RZ + entanglement CNOT).
    3. Kembalikan ekspektasi Pauli-Z (rentang [-1, 1]) yang akan diproyeksikan ke ruang data.
    """
    # Feature map: rotasi berdasarkan komponen laten pada setiap qubit
    for wire in range(n_qubits):
        qml.RY(latent_angles[wire], wires=wire)
        qml.RZ(latent_angles[wire], wires=wire)

    # Variational layers: parameter yang dilatih
    for layer in weights:
        for wire in range(n_qubits):
            qml.RX(layer[wire, 0], wires=wire)
            qml.RY(layer[wire, 1], wires=wire)
            qml.RZ(layer[wire, 2], wires=wire)
        # Entanglement topologi cincin agar informasi tersebar antar qubit
        if n_qubits > 1:
            for wire in range(n_qubits - 1):
                qml.CNOT(wires=[wire, wire + 1])
            qml.CNOT(wires=[n_qubits - 1, 0])

    # Ekspektasi Pauli-Z dipakai sebagai koordinat 2D
    return [qml.expval(qml.PauliZ(wire)) for wire in range(n_qubits)]

class QuantumGenerator(nn.Module):
    """
    Pembungkus PyTorch untuk VQC.
    Gradien dihitung otomatis melalui antar-muka PennyLane x Torch.
    """
    def __init__(self, n_layers: int, scale: float):
        super().__init__()
        weight_shape = (n_layers, n_qubits, 3)
        # Parameter inisialisasi kecil agar stabil di awal training
        self.theta = nn.Parameter(0.01 * torch.randn(weight_shape))
        self.scale = scale

    def forward(self, z_batch: torch.Tensor) -> torch.Tensor:
        """
        Evaluasi qnode per sampel z (loop Python; untuk produksi dapat dioptimalkan dengan batching qml).
        """
        outputs = []
        for latent_angles in z_batch:
            expvals = generator_qnode(latent_angles, self.theta)
            outputs.append(expvals)
            
        output_tensor = torch.tensor(outputs, dtype=torch.float32)

        return self.scale * output_tensor  # proyeksi ke ruang data target

class ClassicalDiscriminator(nn.Module):
    """
    Discriminator klasik sederhana (MLP) yang memisahkan real/fake pada ruang 2D.
    """
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1)  # logit (tanpa sigmoid karena pakai BCEWithLogitsLoss)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

# -------------------------------------------
# Inisialisasi model & optimizer
# -------------------------------------------
generator = QuantumGenerator(n_layers=n_layers, scale=scale_factor).to(device_torch)
discriminator = ClassicalDiscriminator().to(device_torch)

optimizer_G = optim.Adam(generator.parameters(), lr=1e-2)
optimizer_D = optim.Adam(discriminator.parameters(), lr=1e-2)
criterion = nn.BCEWithLogitsLoss()

# -------------------------------------------
# Dataset & buffer loss
# -------------------------------------------
dataset_size = 2048
real_dataset = sample_real_data(dataset_size)

num_epochs = 200
batch_size = 128

g_losses, d_losses = [], []

# -------------------------------------------
# Loop pelatihan adversarial
# -------------------------------------------
for epoch in range(1, num_epochs + 1):
    # --- Sampling mini-batch real data ---
    idx = torch.randint(0, dataset_size, (batch_size,))
    real_batch = real_dataset[idx].to(device_torch)

    # Label real/fake (1: real, 0: fake)
    real_labels = torch.ones((batch_size, 1), device=device_torch)
    fake_labels = torch.zeros((batch_size, 1), device=device_torch)

    # ====== Train Discriminator ======
    optimizer_D.zero_grad()

    # Loss untuk data real
    real_logits = discriminator(real_batch)
    loss_real = criterion(real_logits, real_labels)

    # Loss untuk data fake (detach agar grad generator tidak mengalir saat update D)
    z = sample_latent(batch_size, latent_dim)
    fake_batch = generator(z)
    fake_logits = discriminator(fake_batch.detach())
    loss_fake = criterion(fake_logits, fake_labels)

    loss_D = loss_real + loss_fake
    loss_D.backward()
    optimizer_D.step()

    # ====== Train Generator ======
    optimizer_G.zero_grad()

    z = sample_latent(batch_size, latent_dim)
    generated_samples = generator(z)
    # Generator ingin "menipu", sehingga target label = real (1)
    gen_logits = discriminator(generated_samples)
    loss_G = criterion(gen_logits, real_labels)
    loss_G.backward()
    optimizer_G.step()

    g_losses.append(loss_G.item())
    d_losses.append(loss_D.item())

    if epoch % 20 == 0 or epoch == 1:
        print(f"Epoch {epoch:03d} | D Loss: {loss_D.item():.4f} | G Loss: {loss_G.item():.4f}")

# -------------------------------------------
# Visualisasi hasil generator vs data asli
# -------------------------------------------
generator.eval()
with torch.no_grad():
    real_vis = sample_real_data(512).cpu()
    latent_vis = sample_latent(512, latent_dim)
    fake_vis = generator(latent_vis).cpu()

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].set_title("Real Data (Lingkaran)")
axes[0].scatter(real_vis[:, 0], real_vis[:, 1], s=15, alpha=0.7, color="#2ca02c")
axes[0].set_aspect("equal")

axes[1].set_title("Generated Data (QGAN)")
axes[1].scatter(fake_vis[:, 0], fake_vis[:, 1], s=15, alpha=0.7, color="#ff7f0e")
axes[1].set_aspect("equal")

axes[2].set_title("Kurva Loss")
axes[2].plot(d_losses, label="Discriminator", color="#1f77b4")
axes[2].plot(g_losses, label="Generator", color="#d62728")
axes[2].set_xlabel("Step")
axes[2].set_ylabel("Loss")
axes[2].legend()

plt.tight_layout()
plt.savefig("qgan_training_summary.png", dpi=200)
plt.show()

print("Plot disimpan ke 'qgan_training_summary.png'.")

# -------------------------------------------
# (Opsional) Inferensi & simulasi noise dengan Qiskit Aer
# -------------------------------------------
if QISKIT_AVAILABLE:
    print("\nQiskit Aer tersedia – contoh inferensi & noise simulation.")

    def _pauli_string(n_qubits: int, target: int) -> str:
        """
        Bangun string Pauli (Z) untuk qubit target.
        Qiskit menggunakan konvensi little-endian (qubit-0 = karakter paling kanan).
        """
        chars = ["I"] * n_qubits
        chars[n_qubits - 1 - target] = "Z"
        return "".join(chars)

    def qiskit_expectation(weights_np: np.ndarray,
                           latent_np: np.ndarray,
                           shots: int | None = None,
                           noise_strength: float = 0.0) -> np.ndarray:
        """
        Eksekusi circuit generator menggunakan Qiskit Aer.
        - shots=None -> ekspektasi eksak via Statevector
        - shots>0    -> estimasi statistik dengan opsional noise depolarizing
        """
        qc = QuantumCircuit(n_qubits)
        # Feature map
        for wire in range(n_qubits):
            qc.ry(float(latent_np[wire]), wire)
            qc.rz(float(latent_np[wire]), wire)
        # Variational layers
        for layer in weights_np:
            for wire in range(n_qubits):
                qc.rx(float(layer[wire, 0]), wire)
                qc.ry(float(layer[wire, 1]), wire)
                qc.rz(float(layer[wire, 2]), wire)
            if n_qubits > 1:
                for wire in range(n_qubits - 1):
                    qc.cx(wire, wire + 1)
                qc.cx(n_qubits - 1, 0)

        if shots is None:
            # Ekspektasi ideal (tanpa noise) menggunakan statevector
            state = Statevector.from_instruction(qc)
            expvals = []
            for target in range(n_qubits):
                op = SparsePauliOp.from_list([(_pauli_string(n_qubits, target), 1.0)])
                expvals.append(np.real(state.expectation_value(op)))
            return np.array(expvals, dtype=np.float32)

        # Untuk simulasi shot-based + noise, tambah pengukuran komputasional
        qc_measure = qc.copy()
        qc_measure.measure_all()

        noise_model = None
        if noise_strength > 0.0:
            noise_model = NoiseModel()
            single = depolarizing_error(noise_strength, 1)
            two = depolarizing_error(noise_strength, 2)
            noise_model.add_all_qubit_quantum_error(single, ["u1", "u2", "u3", "rx", "ry", "rz"])
            noise_model.add_all_qubit_quantum_error(two, ["cx"])

        simulator = AerSimulator(noise_model=noise_model)
        result = simulator.run(qc_measure, shots=shots).result()
        counts = result.get_counts()

        # Hitung ekspektasi Z dari statistik hasil pengukuran
        expvals = []
        for target in range(n_qubits):
            exp = 0.0
            for bitstring, count in counts.items():
                bitstring = bitstring[::-1]  # balik supaya index 0 = qubit 0
                bit = int(bitstring[target])
                exp += (1 if bit == 0 else -1) * count
            expvals.append(exp / shots)
        return np.array(expvals, dtype=np.float32)

    # Ambil sampel parameter & latent
    weights_np = generator.theta.detach().numpy()
    latent_np = sample_latent(1, latent_dim)[0].numpy()

    ideal_exp = qiskit_expectation(weights_np, latent_np, shots=None)
    noisy_exp = qiskit_expectation(weights_np, latent_np, shots=4096, noise_strength=0.01)

    print("Ekspektasi (Qiskit Statevector) :", ideal_exp)
    print("Ekspektasi (Qiskit + noise 1%)  :", noisy_exp)
else:
    print("\nQiskit Aer tidak ditemukan. Lewati bagian inferensi Qiskit.")
