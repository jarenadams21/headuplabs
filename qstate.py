# quantum_grant_search_with_particles.py

# Imports and Dependencies
import sys
import numpy as np
import argparse
import matplotlib.pyplot as plt
import string
import nltk
from scipy.linalg import expm
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
from qiskit.quantum_info import Statevector, Operator
from matplotlib.widgets import Button
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from qiskit.circuit.library import UnitaryGate  # Import UnitaryGate for custom gates

# Ensure NLTK resources are downloaded
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# Data Handling Classes
class Grant:
    def __init__(self, title, description, amount, location):
        self.title = title
        self.description = description
        self.amount = amount
        self.location = location

# Sample Grant Data
grants = [
    Grant("Climate Action Grant", "Funding for projects reducing carbon emissions.", 50000, "Boston"),
    Grant("Environmental Research Grant", "Support for environmental impact studies.", 75000, "New York"),
    Grant("Sustainability Initiative Grant", "Grants for sustainable development projects.", 100000, "Boston"),
    Grant("Green Technology Grant", "Funding for innovative green technologies.", 25000, "San Francisco"),
    Grant("Community Clean Energy Grant", "Support for community-based clean energy.", 60000, "Boston"),
    Grant("Eco-Friendly Transportation Grant", "Support for sustainable transportation initiatives.", 40000, "San Diego"),
    Grant("Recycling Innovation Grant", "Grants for innovative recycling technologies.", 35000, "Portland"),
    Grant("Sustainable Agriculture Grant", "Funding for sustainable farming practices.", 60000, "Austin"),
    Grant("Carbon Neutrality Grant", "Support for achieving carbon neutrality in organizations.", 70000, "Boston"),
    Grant("Healthcare Innovation Grant", "Funding for innovative healthcare solutions.", 80000, "Chicago"),
    Grant("Education Advancement Grant", "Support for educational programs and research.", 50000, "Boston"),
    Grant("Artistic Excellence Grant", "Grants for artists and cultural projects.", 20000, "New York"),
    Grant("Tech Startup Grant", "Funding for early-stage tech startups.", 100000, "Silicon Valley"),
    Grant("Renewable Energy Grant", "Support for renewable energy projects and research.", 75000, "Denver"),
    Grant("Wildlife Conservation Grant", "Funding for wildlife protection initiatives.", 60000, "Seattle"),
    Grant("Urban Development Grant", "Support for urban renewal and development projects.", 90000, "Detroit"),
    Grant("Cultural Heritage Grant", "Grants for preserving cultural heritage sites.", 55000, "Rome, Italy"),
    Grant("Ocean Cleanup Grant", "Funding for ocean and marine environment cleanup.", 85000, "Sydney, Australia"),
    Grant("AI Research Grant", "Support for artificial intelligence research.", 120000, "Boston"),
    Grant("Food Security Grant", "Grants to improve global food security.", 70000, "Nairobi, Kenya"),
    Grant("Space Exploration Grant", "Funding for space exploration technologies.", 150000, "Houston"),
    Grant("Mental Health Awareness Grant", "Support for mental health programs.", 40000, "London"),
    Grant("Disaster Relief Grant", "Grants for disaster preparedness and relief efforts.", 50000, "Tokyo"),
    Grant("Water Purification Grant", "Funding for clean water initiatives.", 65000, "Lagos, Nigeria"),
    Grant("Educational Exchange Grant", "Support for international educational exchanges.", 30000, "Berlin, Germany"),
    Grant("Cybersecurity Grant", "Grants for improving cybersecurity measures.", 80000, "Washington D.C."),
    Grant("Agricultural Technology Grant", "Funding for agri-tech innovations.", 90000, "Des Moines"),
    Grant("Veteran Support Grant", "Support for programs aiding military veterans.", 45000, "San Antonio"),
    Grant("Renewable Energy Egypt Grant", "Support for renewable energy projects in Egypt.", 60000, "Cairo, Egypt"),
    Grant("Middle East Education Grant", "Funding for educational initiatives in the Middle East.", 50000, "Dubai, UAE"),
    Grant("African Development Grant", "Grants for development projects across Africa.", 75000, "Accra, Ghana"),
    Grant("Sahara Desert Research Grant", "Funding for environmental research in the Sahara Desert.", 55000, "Cairo, Egypt"),
    Grant("Mediterranean Cultural Grant", "Support for cultural projects in the Mediterranean region.", 40000, "Athens, Greece"),
    Grant("Historical Preservation Grant", "Grants for preserving historical landmarks.", 50000, "Cairo, Egypt"),
    Grant("Educational Technology Grant", "Funding for EdTech solutions.", 80000, "Boston"),
    Grant("Global Health Initiative Grant", "Support for global health improvement projects.", 100000, "Geneva, Switzerland"),
    Grant("Digital Innovation Grant", "Support for digital transformation projects.", 60000, "New York"),
    Grant("Renewable Transportation Grant", "Funding for electric vehicle infrastructure.", 85000, "Los Angeles"),
    Grant("Biotechnology Research Grant", "Support for biotech research initiatives.", 90000, "San Francisco"),
    Grant("Clean Water Access Grant", "Grants to improve access to clean water.", 50000, "Mumbai, India"),
    Grant("Anti-Poverty Initiative Grant", "Funding for poverty reduction programs.", 70000, "Lagos, Nigeria"),
    Grant("Renewable Heating Grant", "Support for renewable heating solutions.", 65000, "Reykjavik, Iceland"),
    Grant("Language Preservation Grant", "Grants for preserving endangered languages.", 40000, "Quebec, Canada"),
    Grant("Marine Biology Research Grant", "Funding for marine ecosystem studies.", 75000, "Cape Town, South Africa"),
    Grant("Climate Education Grant", "Support for climate change education programs.", 55000, "Oslo, Norway"),
    Grant("Artificial Intelligence Ethics Grant", "Grants for AI ethics research.", 80000, "Cambridge, UK"),
    Grant("Robotics Innovation Grant", "Funding for robotics and automation projects.", 95000, "Tokyo, Japan"),
    Grant("Urban Agriculture Grant", "Support for city farming initiatives.", 60000, "Detroit"),
    Grant("Renewable Materials Grant", "Grants for developing sustainable materials.", 70000, "Berlin, Germany"),
    Grant("Childhood Education Grant", "Funding for early childhood education programs.", 50000, "Stockholm, Sweden"),
    Grant("Digital Literacy Grant", "Support for digital literacy initiatives.", 45000, "Seoul, South Korea"),
    Grant("Ecotourism Development Grant", "Grants for sustainable tourism projects.", 65000, "Bangkok, Thailand"),
    Grant("Pandemic Response Grant", "Funding for pandemic preparedness and response.", 100000, "Geneva, Switzerland"),
    Grant("Renewable Energy Storage Grant", "Support for energy storage solutions.", 85000, "Sydney, Australia"),
    Grant("Clean Air Initiative Grant", "Grants to reduce air pollution.", 75000, "Beijing, China"),
    Grant("Blockchain Research Grant", "Funding for blockchain technology research.", 80000, "Zurich, Switzerland"),
    Grant("Nanotechnology Grant", "Support for nanotech research projects.", 90000, "Singapore"),
    Grant("Space Habitat Grant", "Funding for space habitat development.", 120000, "Houston"),
    Grant("Quantum Computing Grant", "Support for quantum computing research.", 150000, "Boston"),
]

# Particle Classes
class Particle:
    def __init__(self, name):
        self.name = name

class Quark(Particle):
    def __init__(self, name, flavor, color_charge):
        super().__init__(name)
        self.flavor = flavor  # 'up', 'down', 'charm', 'strange', 'top', 'bottom'
        self.color_charge = color_charge  # 'red', 'green', 'blue'

class Lepton(Particle):
    def __init__(self, name, flavor, spin):
        super().__init__(name)
        self.flavor = flavor  # 'electron', 'muon', 'tau', 'electron neutrino', 'muon neutrino', 'tau neutrino'
        self.spin = spin  # '+1/2', '-1/2'

class Boson(Particle):
    def __init__(self, name, force_type):
        super().__init__(name)
        self.force_type = force_type  # 'strong', 'weak', 'electromagnetic', 'gravitational', 'mass'

# Quantum Searcher Base Class
class QuantumSearcher:
    def __init__(self, data_items):
        self.data_items = data_items
        self.num_items = len(data_items)
        self.num_qubits = int(np.ceil(np.log2(self.num_items)))
        self.backend = AerSimulator(method='statevector')  # Use statevector simulator for state tracking

        # Initialize particles
        self.particles = self.initialize_particles()

        # Core Time Backbone
        self.global_time = 0
        self.time_step = 0.1  # Define a suitable time step for synchronization

    def initialize_particles(self):
        particles = []

        # Define the 50 most well-known particles
        quark_flavors = ['up', 'down', 'charm', 'strange', 'top', 'bottom']
        lepton_flavors = ['electron', 'electron neutrino', 'muon', 'muon neutrino', 'tau', 'tau neutrino']
        bosons = [
            {'name': 'Photon', 'force_type': 'electromagnetic'},
            {'name': 'Gluon', 'force_type': 'strong'},
            {'name': 'W+', 'force_type': 'weak'},
            {'name': 'W-', 'force_type': 'weak'},
            {'name': 'Z', 'force_type': 'weak'},
            {'name': 'Higgs', 'force_type': 'mass'},
            {'name': 'Graviton', 'force_type': 'gravitational'},  # Hypothetical
        ]

        colors = ['red', 'green', 'blue']
        spins = ['+1/2', '-1/2']

        # Add quarks (18 particles: 6 flavors * 3 colors)
        for flavor in quark_flavors:
            for color in colors:
                particles.append(Quark(f'{flavor.capitalize()} Quark ({color})', flavor, color))

        # Add leptons (12 particles: 6 flavors * 2 spins)
        for flavor in lepton_flavors:
            for spin in spins:
                particles.append(Lepton(f'{flavor.capitalize()} ({spin})', flavor, spin))

        # Add bosons (7 particles)
        for boson in bosons:
            particles.append(Boson(boson['name'], boson['force_type']))

        # Add antiparticles for quarks and leptons
        antiparticles = []
        for particle in particles:
            if isinstance(particle, Quark):
                antiparticles.append(Quark(f'Anti-{particle.name}', particle.flavor, particle.color_charge))
            elif isinstance(particle, Lepton):
                antiparticles.append(Lepton(f'Anti-{particle.name}', particle.flavor, particle.spin))

        particles.extend(antiparticles)

        # Ensure we have at least as many particles as quantum states
        required_particles = 2 ** self.num_qubits
        while len(particles) < required_particles:
            particles.extend(particles[:required_particles - len(particles)])

        return particles

    def quark_phase_shift(self, color_charge):
        '''
        Calculate phase shift for quarks based on color charge.
        '''
        color_phases = {'red': 0, 'green': 2*np.pi/3, 'blue': 4*np.pi/3}
        return color_phases.get(color_charge, 0)

    def lepton_phase_shift(self, spin):
        '''
        Calculate phase shift for leptons based on spin.
        '''
        spin_phases = {'+1/2': np.pi/2, '-1/2': -np.pi/2}
        return spin_phases.get(spin, 0)

    def boson_phase_shift(self, force_type):
        '''
        Calculate phase shift for bosons based on force type.
        '''
        force_phases = {'electromagnetic': np.pi/4, 'weak': np.pi/3, 'strong': np.pi/6, 'gravitational': np.pi/8, 'mass': np.pi/5}
        return force_phases.get(force_type, 0)

    def create_particle_oracle(self, indices):
        '''
        Create an oracle that applies phase shifts based on particle properties.
        '''
        if not indices:
            return UnitaryGate(np.identity(2 ** self.num_qubits), label='Identity')

        # Initialize phase shifts
        phase_shifts = np.ones(2 ** self.num_qubits, dtype=complex)

        for index in range(2 ** self.num_qubits):
            if index in indices:
                # Retrieve associated particle
                particle = self.particles[index % len(self.particles)]

                # Apply phase shift based on particle type and properties
                if isinstance(particle, Quark):
                    # Quark phase shift (color charge)
                    phase = self.quark_phase_shift(particle.color_charge)
                elif isinstance(particle, Lepton):
                    # Lepton phase shift (spin)
                    phase = self.lepton_phase_shift(particle.spin)
                elif isinstance(particle, Boson):
                    # Boson-mediated force
                    phase = self.boson_phase_shift(particle.force_type)
                else:
                    phase = 0  # Default phase

                phase_shifts[index] = -np.exp(1j * phase)  # Mark the state
            else:
                phase_shifts[index] = 1

        oracle_matrix = np.diag(phase_shifts)
        # Convert the oracle_matrix into a gate with a name
        oracle_gate = UnitaryGate(oracle_matrix, label='Particle-Oracle')
        return oracle_gate

    def calculate_boson_effect(self):
        '''
        Calculate the cumulative effect of boson-mediated forces on the phase.
        '''
        boson_effect = 0
        for particle in self.particles:
            if isinstance(particle, Boson):
                boson_effect += self.boson_phase_shift(particle.force_type)
        return boson_effect

    def create_particle_diffuser(self):
        '''
        Create a diffuser that includes particle interactions.
        '''
        diffuser = QuantumCircuit(self.num_qubits)
        diffuser.h(range(self.num_qubits))
        diffuser.x(range(self.num_qubits))

        # Apply multi-controlled Z gate with phase adjustments based on boson interactions
        angle = np.pi  # Standard Grover diffuser angle
        boson_effect = self.calculate_boson_effect()

        if self.num_qubits == 1:
            diffuser.p(angle + boson_effect, 0)
        else:
            diffuser.h(self.num_qubits - 1)
            diffuser.mcx(list(range(self.num_qubits - 1)), self.num_qubits - 1)
            diffuser.h(self.num_qubits - 1)
            # Apply additional phase shift
            diffuser.p(boson_effect, self.num_qubits - 1)

        diffuser.x(range(self.num_qubits))
        diffuser.h(range(self.num_qubits))
        diffuser_gate = diffuser.to_gate(label='Particle-Diffuser')
        return diffuser_gate

    def construct_hamiltonian(self):
        '''
        Construct the Hamiltonian operator representing the system's energy, including interactions.
        '''
        dim = 2 ** self.num_qubits
        H = np.zeros((dim, dim), dtype=complex)

        # Loop over all states and assign energy levels based on particles
        for i in range(dim):
            for j in range(dim):
                if i == j:
                    # Diagonal elements: state energy
                    state_energy = self.calculate_state_energy(i)
                    H[i, i] = state_energy
                else:
                    # Off-diagonal elements: interaction-induced transitions
                    interaction_strength = self.calculate_state_interaction_strength(i, j)
                    H[i, j] = interaction_strength

        # Ensure Hamiltonian is Hermitian
        H = (H + H.conj().T) / 2

        # Convert to operator
        hamiltonian_operator = Operator(H)
        return hamiltonian_operator

    def calculate_state_energy(self, index):
        '''
        Calculate the energy of a quantum state based on associated particles.
        '''
        particle = self.particles[index % len(self.particles)]
        if isinstance(particle, Quark):
            # Energy based on flavor (in MeV)
            flavor_energies = {'up': 2.2, 'down': 4.7, 'charm': 1275, 'strange': 96, 'top': 173100, 'bottom': 4180}
            energy = flavor_energies.get(particle.flavor, 0)
        elif isinstance(particle, Lepton):
            # Energy based on flavor (in MeV)
            flavor_energies = {
                'electron': 0.511,
                'muon': 105.7,
                'tau': 1776.86,
                'electron neutrino': 0.000001,  # Approximate upper limits
                'muon neutrino': 0.000001,
                'tau neutrino': 0.000001
            }
            energy = flavor_energies.get(particle.flavor, 0)
        elif isinstance(particle, Boson):
            # Energy based on particle (in MeV)
            boson_energies = {
                'Photon': 0,
                'Gluon': 0,
                'W+': 80379,
                'W-': 80379,
                'Z': 91188,
                'Higgs': 125000,
                'Graviton': 0  # Hypothetical particle
            }
            energy = boson_energies.get(particle.name, 0)
        else:
            energy = 0

        return energy

    def calculate_state_interaction_strength(self, index_i, index_j):
        '''
        Calculate the strength of interaction-induced transitions between two states.
        '''
        particle_i = self.particles[index_i % len(self.particles)]
        particle_j = self.particles[index_j % len(self.particles)]

        # Use realistic interaction strengths based on particle physics
        interaction_strength = 0

        # Coupling constants (dimensionless)
        alpha_strong = 1  # Approximate
        alpha_em = 1/137
        alpha_weak = 1e-5

        if isinstance(particle_i, Quark) and isinstance(particle_j, Quark):
            # Quark-quark interactions (strong force)
            interaction_strength += alpha_strong
        elif isinstance(particle_i, Lepton) and isinstance(particle_j, Lepton):
            # Lepton-lepton interactions (electromagnetic and weak forces)
            interaction_strength += alpha_em + alpha_weak
        elif (isinstance(particle_i, Quark) and isinstance(particle_j, Lepton)) or (isinstance(particle_i, Lepton) and isinstance(particle_j, Quark)):
            # Quark-lepton interactions (weak force)
            interaction_strength += alpha_weak
        elif isinstance(particle_i, Boson) or isinstance(particle_j, Boson):
            # Interactions involving bosons (mediators)
            interaction_strength += alpha_em + alpha_weak + alpha_strong
        else:
            # Default minimal interaction
            interaction_strength += 0.0

        return interaction_strength

    def inner_sensing(self, probabilities, iteration):
        '''
        Perform inner sensing and log particle interactions.
        '''
        print(f"\nInner Sensing at Iteration {iteration + 1}:")
        entropy = -sum(prob * np.log2(prob) for prob in probabilities.values() if prob > 0)
        print(f"  Quantum State Entropy: {entropy:.4f} bits")

        # Log particle influences
        for state_str, prob in probabilities.items():
            if prob > 0.01:
                index = int(state_str, 2)
                particle = self.particles[index % len(self.particles)]
                print(f"  State {state_str}: Probability = {prob:.4f}, Particle = {particle.name}, Type = {type(particle).__name__}")
                if isinstance(particle, Quark):
                    print(f"    Flavor: {particle.flavor}, Color Charge: {particle.color_charge}")
                elif isinstance(particle, Lepton):
                    print(f"    Flavor: {particle.flavor}, Spin: {particle.spin}")
                elif isinstance(particle, Boson):
                    print(f"    Particle: {particle.name}, Mediated Force: {particle.force_type}")
        print("-" * 50)

    def generate_quasicrystal_points(self):
        '''
        Generate quasicrystal lattice points corresponding to the quantum states.
        '''
        num_points = 2 ** self.num_qubits
        golden_ratio = (1 + np.sqrt(5)) / 2
        indices = np.arange(num_points)
        theta = 2 * np.pi * indices / golden_ratio
        phi = np.arccos(1 - 2 * indices / num_points)
        x = np.cos(theta) * np.sin(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(phi)
        return x, y, z

    def plot_with_stepper(self, probabilities_list, state_to_item, indices_to_search):
        '''
        Creates an interactive plot with stepper functionality to move back and forth between iterations.
        '''
        # Prepare data for plotting
        item_titles = [self.data_items[i].title for i in range(len(self.data_items))]
        item_indices = list(range(len(self.data_items)))
        num_frames = len(probabilities_list)
        current_frame = [0]  # Mutable object to allow modification inside nested functions

        # Create a figure and axis for the plot
        fig, ax = plt.subplots(figsize=(12, 6))
        bar_width = 0.8  # Increase bar width
        bars = ax.bar(item_indices, [0]*len(self.data_items), bar_width, tick_label=item_titles)
        ax.set_ylim(0, 1)
        ax.set_ylabel('Probability')
        ax.set_title('Probability Evolution Over Iterations')
        plt.xticks(rotation=90)
        ax.set_xlabel(f'Iteration {current_frame[0]}')

        # Initialize the bars
        def init():
            y = []
            probabilities = probabilities_list[0]
            for item_index in range(len(self.data_items)):
                state_label = format(item_index, f'0{self.num_qubits}b')
                prob = probabilities.get(state_label, 0)
                y.append(prob)
            for bar, prob, idx in zip(bars, y, item_indices):
                bar.set_height(prob)
                bar.set_color('green' if idx in indices_to_search else 'blue')
                if prob > 0.01:
                    bar.set_alpha(1.0)
                else:
                    bar.set_alpha(0.2)
                # Add particle info as annotation
                particle = self.particles[idx % len(self.particles)]
                bar.set_label(f'{particle.name} ({type(particle).__name__})')
            return bars

        init()

        # Function to update the bars for each frame
        def update(frame):
            y = []
            probabilities = probabilities_list[frame]
            for item_index in range(len(self.data_items)):
                state_label = format(item_index, f'0{self.num_qubits}b')
                prob = probabilities.get(state_label, 0)
                y.append(prob)
            for bar, prob, idx in zip(bars, y, item_indices):
                bar.set_height(prob)
                bar.set_color('green' if idx in indices_to_search else 'blue')
                if prob > 0.01:
                    bar.set_alpha(1.0)
                else:
                    bar.set_alpha(0.2)
                # Update particle info
                particle = self.particles[idx % len(self.particles)]
                bar.set_label(f'{particle.name} ({type(particle).__name__})')
            ax.set_xlabel(f'Iteration {frame}')
            fig.canvas.draw_idle()

        # Button callback functions
        def next_iteration(event):
            if current_frame[0] < num_frames - 1:
                current_frame[0] += 1
                update(current_frame[0])

        def prev_iteration(event):
            if current_frame[0] > 0:
                current_frame[0] -= 1
                update(current_frame[0])

        # Play/Pause functionality
        is_playing = [False]

        def play_pause(event):
            is_playing[0] = not is_playing[0]
            if is_playing[0]:
                play_button.label.set_text('Pause')
                animate()
            else:
                play_button.label.set_text('Play')

        # Animation function
        def animate():
            if is_playing[0]:
                if current_frame[0] < num_frames - 1:
                    current_frame[0] += 1
                else:
                    current_frame[0] = 0  # Loop back to the start
                update(current_frame[0])
                fig.canvas.flush_events()
                fig.canvas.start_event_loop(0.5)
                animate()
            else:
                return

        # Add buttons
        ax_prev = plt.axes([0.7, 0.02, 0.1, 0.05])
        ax_next = plt.axes([0.81, 0.02, 0.1, 0.05])
        ax_play = plt.axes([0.59, 0.02, 0.1, 0.05])

        btn_prev = Button(ax_prev, 'Previous')
        btn_next = Button(ax_next, 'Next')
        play_button = Button(ax_play, 'Play')

        btn_prev.on_clicked(prev_iteration)
        btn_next.on_clicked(next_iteration)
        play_button.on_clicked(play_pause)

        plt.tight_layout()
        plt.show(block=False)

    def plot_quasicrystal_with_probabilities(self, probabilities_list, indices_to_search):
        '''
        Plot the quasicrystal lattice with points sized and colored according to the probabilities.
        '''
        num_iterations = len(probabilities_list)
        current_frame = [0]

        # Generate quasicrystal points
        x, y, z = self.generate_quasicrystal_points()
        num_points = len(x)

        # Prepare data mapping state indices to quasicrystal points
        state_indices = list(range(num_points))
        state_labels = [format(i, f'0{self.num_qubits}b') for i in state_indices]

        # Create a figure and axis for the plot
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Normalize probabilities for color mapping
        norm = Normalize(vmin=0, vmax=1)
        cmap = plt.get_cmap('viridis')

        # Initial scatter plot
        probabilities = probabilities_list[current_frame[0]]
        colors = []
        sizes = []
        for i, state_label in enumerate(state_labels):
            prob = probabilities.get(state_label, 0)
            colors.append(cmap(norm(prob)))
            sizes.append(prob * 2000)  # Scale size for visibility

        scatter = ax.scatter(x, y, z, c=colors, s=sizes, alpha=0.7)

        # Highlight target indices
        target_indices = [i for i in indices_to_search]
        target_x = x[target_indices]
        target_y = y[target_indices]
        target_z = z[target_indices]
        ax.scatter(target_x, target_y, target_z, c='red', s=100, label='Target Items')

        # Enhance visual appeal
        ax.set_title('Quasicrystal Lattice Representation of Quantum States')
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')

        # Add particle names as annotations
        for i, state_label in enumerate(state_labels):
            particle = self.particles[i % len(self.particles)]
            ax.text(x[i], y[i], z[i], f'{particle.name}', size=8)

        # Add legend
        ax.legend()

        # Adjust viewing angle
        ax.view_init(elev=30, azim=120)

        # Function to update the scatter plot
        def update(frame):
            probabilities = probabilities_list[frame]
            colors = []
            sizes = []
            for i, state_label in enumerate(state_labels):
                prob = probabilities.get(state_label, 0)
                colors.append(cmap(norm(prob)))
                sizes.append(prob * 2000)
            scatter._facecolor3d = colors
            scatter._sizes = sizes
            fig.canvas.draw_idle()

        # Button callback functions
        def next_iteration(event):
            if current_frame[0] < num_iterations - 1:
                current_frame[0] += 1
                update(current_frame[0])

        def prev_iteration(event):
            if current_frame[0] > 0:
                current_frame[0] -= 1
                update(current_frame[0])

        # Play/Pause functionality
        is_playing = [False]

        def play_pause(event):
            is_playing[0] = not is_playing[0]
            if is_playing[0]:
                play_button.label.set_text('Pause')
                animate()
            else:
                play_button.label.set_text('Play')

        # Animation function
        def animate():
            if is_playing[0]:
                if current_frame[0] < num_iterations - 1:
                    current_frame[0] += 1
                else:
                    current_frame[0] = 0  # Loop back to the start
                update(current_frame[0])
                fig.canvas.flush_events()
                fig.canvas.start_event_loop(0.5)
                animate()
            else:
                return

        # Add buttons
        ax_prev = plt.axes([0.7, 0.05, 0.1, 0.05])
        ax_next = plt.axes([0.81, 0.05, 0.1, 0.05])
        ax_play = plt.axes([0.59, 0.05, 0.1, 0.05])

        btn_prev = Button(ax_prev, 'Previous')
        btn_next = Button(ax_next, 'Next')
        play_button = Button(ax_play, 'Play')

        btn_prev.on_clicked(prev_iteration)
        btn_next.on_clicked(next_iteration)
        play_button.on_clicked(play_pause)

        plt.show()

    def plot_spike_graph(self, particle_probabilities):
        '''
        Plot a spike graph representing the volatility or stability of particle probabilities over iterations.
        '''
        iterations = len(particle_probabilities)
        num_particles = len(self.particles)

        # Prepare data for plotting
        particle_names = [particle.name for particle in self.particles]
        particle_types = [type(particle).__name__ for particle in self.particles]
        colors = {'Quark': 'blue', 'Lepton': 'green', 'Boson': 'red'}

        # Aggregate probabilities over iterations for each particle
        avg_probabilities = np.mean(particle_probabilities, axis=0)

        # Sort particles based on average probability
        sorted_indices = np.argsort(-avg_probabilities)
        sorted_particle_names = [particle_names[i] for i in sorted_indices]
        sorted_particle_types = [particle_types[i] for i in sorted_indices]
        sorted_avg_probabilities = avg_probabilities[sorted_indices]

        # Create a figure and axis for the plot
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot spike graph
        for idx, prob in enumerate(sorted_avg_probabilities):
            ax.vlines(idx, 0, prob, color=colors.get(sorted_particle_types[idx], 'black'), linewidth=2)
            ax.scatter(idx, prob, color=colors.get(sorted_particle_types[idx], 'black'))

        # Customize the graph
        ax.set_title("Spike Graph of Particle Influence Over Iterations")
        ax.set_xlabel("Particles")
        ax.set_ylabel("Average Probability")
        ax.set_xticks(range(len(sorted_particle_names)))
        ax.set_xticklabels(sorted_particle_names, rotation=90)
        ax.grid(True)
        plt.tight_layout()
        plt.show()

    def create_natural_teleportation_operator(self):
        '''
        Create an operator that naturally evolves the system by teleporting particles based on their interactions over time.
        '''
        dim = 2 ** self.num_qubits

        # Define interaction strengths based on particle properties
        interaction_matrix = np.zeros((dim, dim), dtype=complex)

        for i in range(dim):
            for j in range(dim):
                if i != j:
                    particle_i = self.particles[i % len(self.particles)]
                    particle_j = self.particles[j % len(self.particles)]
                    # Define an interaction based on similarities in properties
                    interaction = self.calculate_particle_interaction_strength(particle_i, particle_j)
                    interaction_matrix[i, j] = interaction

        # Ensure interaction_matrix is Hermitian
        interaction_matrix = (interaction_matrix + interaction_matrix.conj().T) / 2

        # Exponentiate the interaction matrix to simulate evolution over time
        time_evolution_operator = expm(-1j * interaction_matrix * self.time_step)

        # Convert to UnitaryGate
        teleportation_gate = UnitaryGate(time_evolution_operator, label='Natural-Teleportation')
        return teleportation_gate

    def calculate_particle_interaction_strength(self, particle_i, particle_j):
        '''
        Calculate the interaction strength between two particles.
        '''
        # Realistic interaction based on particle properties
        strength = 0

        # Coupling constants (dimensionless)
        alpha_strong = 1  # Approximate
        alpha_em = 1/137
        alpha_weak = 1e-5

        if type(particle_i) == type(particle_j):
            strength += 0.1  # Similar particles interact

        if isinstance(particle_i, Quark) and isinstance(particle_j, Quark):
            if particle_i.color_charge == particle_j.color_charge:
                strength += alpha_strong
            else:
                strength += alpha_strong * 0.5
        elif isinstance(particle_i, Lepton) and isinstance(particle_j, Lepton):
            if particle_i.spin == particle_j.spin:
                strength += alpha_em + alpha_weak
            else:
                strength += alpha_weak
        elif isinstance(particle_i, Boson) and isinstance(particle_j, Boson):
            if particle_i.force_type == particle_j.force_type:
                strength += alpha_em + alpha_weak + alpha_strong
            else:
                strength += alpha_weak
        elif (isinstance(particle_i, Quark) and isinstance(particle_j, Lepton)) or (isinstance(particle_i, Lepton) and isinstance(particle_j, Quark)):
            strength += alpha_weak

        return strength

# Quantum Grant Searcher
class QuantumGrantSearcher(QuantumSearcher):
    def __init__(self, grants):
        super().__init__(grants)

    def encode_query(self, query):
        '''
        Process the search query and identify matching grants.
        '''
        lemmatizer = WordNetLemmatizer()
        translator = str.maketrans('', '', string.punctuation)
        query_terms = nltk.word_tokenize(query.lower().translate(translator))
        lemmatized_query_terms = [lemmatizer.lemmatize(term) for term in query_terms]
        query_terms_set = set(lemmatized_query_terms)

        matching_indices = []
        partial_match_scores = []

        for i, grant in enumerate(self.data_items):
            grant_text = f"{grant.title} {grant.description} {grant.location}".lower()
            grant_text = grant_text.translate(translator)
            grant_terms = nltk.word_tokenize(grant_text)
            lemmatized_grant_terms = [lemmatizer.lemmatize(term) for term in grant_terms]

            grant_terms_set = set(lemmatized_grant_terms)
            common_terms = query_terms_set & grant_terms_set
            match_score = len(common_terms) / len(query_terms_set) if len(query_terms_set) > 0 else 0

            if match_score == 1.0:
                matching_indices.append(i)
            elif match_score > 0:
                partial_match_scores.append((i, match_score))

        return matching_indices, partial_match_scores

    def get_item_title(self, state_label):
        '''
        Retrieve the grant title based on the state label.
        '''
        item_index = int(state_label, 2)
        if item_index < len(self.data_items):
            return self.data_items[item_index].title
        return "Unknown Grant"

    def search(self, query):
        '''
        Perform the quantum search using the particle-enhanced Grover's algorithm.
        '''
        matching_indices, partial_match_scores = self.encode_query(query)

        if matching_indices:
            indices_to_search = matching_indices
            print("Exact matches found. Performing quantum search...")
        elif partial_match_scores:
            partial_match_scores.sort(key=lambda x: x[1], reverse=True)
            top_matches = [index for index, score in partial_match_scores[:3]]
            indices_to_search = top_matches
            print("No exact matches found. Performing quantum search on top potential candidates...")
        else:
            print("No matching grants found.")
            return

        # Initialize quantum circuit
        qr = QuantumRegister(self.num_qubits)
        cr = ClassicalRegister(self.num_qubits)
        qc = QuantumCircuit(qr, cr)

        # Apply Hadamard gates to create superposition
        qc.h(qr)

        # Prepare Oracle and Diffuser
        oracle_gate = self.create_particle_oracle(indices_to_search)
        diffuser_gate = self.create_particle_diffuser()

        num_iterations = int(np.floor(np.pi / 4 * np.sqrt(2 ** self.num_qubits / len(indices_to_search))))
        if num_iterations == 0:
            num_iterations = 1  # Ensure at least one iteration

        # Initialize the statevector
        state = Statevector.from_label('0' * self.num_qubits)
        state = state.evolve(qc)

        # Mapping from state labels to grant titles
        state_to_item = {}
        for i in range(len(self.data_items)):
            state_label = format(i, f'0{self.num_qubits}b')
            state_to_item[state_label] = self.data_items[i].title

        # Collect probabilities for plotting
        probabilities_list = []
        particle_probabilities = []  # For spike graph
        probabilities = state.probabilities_dict()
        probabilities_list.append(probabilities)
        # Collect initial particle probabilities
        particle_probs = []
        for idx in range(len(self.particles)):
            state_label = format(idx, f'0{self.num_qubits}b')
            prob = probabilities.get(state_label, 0)
            particle_probs.append(prob)
        particle_probabilities.append(particle_probs)

        # Construct Hamiltonian
        hamiltonian = self.construct_hamiltonian()

        # Time evolution parameters
        total_time = num_iterations * self.time_step

        # Apply Grover's algorithm iterations
        for iteration in range(num_iterations):
            # Update global time
            self.global_time += self.time_step

            # Apply Oracle
            qc.append(oracle_gate, qr)
            state = state.evolve(oracle_gate)

            # Quantum Teleportation through natural evolution
            teleport_gate = self.create_natural_teleportation_operator()
            qc.append(teleport_gate, qr)
            state = state.evolve(teleport_gate)

            # Apply Hamiltonian evolution
            time_evolution_operator = expm(-1j * hamiltonian.data * self.time_step)
            hamiltonian_gate = UnitaryGate(time_evolution_operator, label='Hamiltonian-Evolution')
            qc.append(hamiltonian_gate, qr)
            state = state.evolve(hamiltonian_gate)

            # Apply Diffuser
            qc.append(diffuser_gate, qr)
            state = state.evolve(diffuser_gate)

            # Compute probabilities
            probabilities = state.probabilities_dict()
            probabilities_list.append(probabilities)
            # Collect particle probabilities
            particle_probs = []
            for idx in range(len(self.particles)):
                state_label = format(idx, f'0{self.num_qubits}b')
                prob = probabilities.get(state_label, 0)
                particle_probs.append(prob)
            particle_probabilities.append(particle_probs)

            # Print the probability currents
            print(f"\nIteration {iteration + 1}:")
            for state_str in sorted(probabilities):
                prob = probabilities[state_str]
                item_title = state_to_item.get(state_str, None)
                if item_title:
                    print(f"  State {state_str} ({item_title}): Probability = {prob:.4f}")
            print("-" * 50)

            # Inner Sensing Routine
            self.inner_sensing(probabilities, iteration)

        # Measurement
        qc.measure(qr, cr)

        # Transpile the circuit after adding measurement
        transpiled_qc = transpile(qc, self.backend)

        # Execute the circuit
        job = self.backend.run(transpiled_qc, shots=1024)
        result = job.result()
        counts = result.get_counts()

        # Map counts to grant titles
        counts_items = {}
        for state_str, count in counts.items():
            item_index = int(state_str, 2)
            if item_index < len(self.data_items):
                item_title = self.data_items[item_index].title
                counts_items[item_title] = counts_items.get(item_title, 0) + count

        if counts_items:
            max_count = max(counts_items.values())
            most_common_titles = [title for title, count in counts_items.items() if count == max_count]
            best_item_title = most_common_titles[0]
            for item in self.data_items:
                if item.title == best_item_title:
                    best_item = item
                    break
        else:
            print("No valid grant found in measurement results.")
            return

        # Output the result
        print(f"\nThe most relevant grant based on your query '{query}' is:")
        print(f"Title: {best_item.title}")
        print(f"Description: {best_item.description}")
        print(f"Amount: ${best_item.amount}")
        print(f"Location: {best_item.location}")

        # If there were partial matches, list them
        if not matching_indices and partial_match_scores:
            print("\nOther potential candidate grants:")
            for index, score in partial_match_scores[:3]:
                item = self.data_items[index]
                print(f"- Title: {item.title}, Match Score: {score:.2f}")

        # Plot histogram of measurement results with grant titles
        plot_histogram(counts_items)
        plt.show(block=False)

        # Plot probability currents with stepper functionality
        self.plot_with_stepper(probabilities_list, state_to_item, indices_to_search)

        # Plot the quasicrystal visualization connected to the search
        self.plot_quasicrystal_with_probabilities(probabilities_list, indices_to_search)

        # Plot the spike graph of particle probabilities
        self.plot_spike_graph(particle_probabilities)

# Main Function
def main():
    parser = argparse.ArgumentParser(description='Quantum Grant Search Tool with Standard Model Particles')
    parser.add_argument('query', type=str, nargs='?', default='',
                        help='Search query to find relevant grants (e.g., "renewable energy projects in Boston")')
    args = parser.parse_args()

    if not args.query:
        print("Please provide a search query.")
        sys.exit(1)

    # Initialize the Quantum Grant Searcher
    searcher = QuantumGrantSearcher(grants)

    # Perform the quantum search
    searcher.search(args.query)

if __name__ == "__main__":
    main()
