# quantum_grant_search_quantum.py

# Imports and Dependencies
import sys
import numpy as np
import argparse
import matplotlib.pyplot as plt  # type: ignore
import string
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
from qiskit.quantum_info import Statevector, Operator
from matplotlib import animation, cm
from matplotlib.widgets import Button
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import splprep, splev

# Ensure NLTK resources are downloaded
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# System Equation:
# The quantum state vector evolves under the Grover operator G:
# |\psi_{k+1}\rangle = G |\psi_k\rangle
# where G = D * O
# - O is the Oracle operator
# - D is the Diffuser (Inversion about the mean) operator
# This equation governs the evolution of the quantum state in Grover's algorithm.
# The probability amplitudes evolve analogously to celestial bodies under gravitational influence,
# with amplitudes being 'pulled' towards the target states, similar to how masses influence each other in space.

# Grant Data Handling
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
    Grant("Little Onion Restaurant Grant", "Support for small businesses and restaurants in California and Nevada.", 5000, "Santa Ana"),
    Grant("Mike's Grant", "I am legit but also a scam, but I'll give you more! Give me business, now!", 10000, "Orange Grove"),
    Grant("Subnautic Travelers", "All sea-men and voyagers of the blue alike!", 100000, "Highwaters, LN"),
    Grant("A Time Ago", "Subsidizing Egyptian student housing and groceries", 3500, "Cairo, Egypt"),
    # Additional Grants
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
]


# Quantum Grant Searcher
class QuantumGrantSearcher:
    def __init__(self, grants):
        self.grants = grants
        self.num_grants = len(grants)
        self.num_qubits = int(np.ceil(np.log2(self.num_grants)))
        self.backend = AerSimulator(method='statevector')  # Use statevector simulator for state tracking

    # Modify the create_oracle method
    def create_quasi_oracle(self, indices):
        '''
        Create a quasicrystal-inspired oracle that applies custom phase shifts.
        '''
        oracle = QuantumCircuit(self.num_qubits)
        if not indices:
            # No indices to flip; return identity oracle
            return oracle.to_gate()

        # Create a phase shift function based on quasicrystal properties
        phase_shifts = np.ones(2 ** self.num_qubits, dtype=complex)  # Ensure complex data type
        for index in indices:
            # Apply a custom phase shift
            phase_shifts[index] = np.exp(2j * np.pi * self.quasicrystal_phase(index))

        # Construct the oracle matrix
        oracle_matrix = np.diag(phase_shifts)
        oracle_operator = Operator(oracle_matrix)
        oracle_gate = oracle_operator.to_instruction()
        oracle_gate.label = "Quasi-Oracle"
        return oracle_gate


    def quasicrystal_phase(self, index):
        '''
        Define a phase function inspired by quasicrystal patterns.
        '''
        # Example using the golden ratio
        golden_ratio = (1 + np.sqrt(5)) / 2
        phase = (index * golden_ratio) % 1  # Fractional part
        return phase

    # Modify the create_diffuser method
    def create_quasi_diffuser(self):
        '''
        Create a diffuser that accounts for quasicrystal-inspired superpositions.
        '''
        diffuser = QuantumCircuit(self.num_qubits)
        # Custom operations can be added here
        diffuser.h(range(self.num_qubits))
        diffuser.x(range(self.num_qubits))
        # Apply a controlled-phase gate with custom angles
        angle = 2 * np.pi / (2 ** self.num_qubits)
        diffuser.mcp(angle, list(range(self.num_qubits - 1)), self.num_qubits - 1)
        diffuser.x(range(self.num_qubits))
        diffuser.h(range(self.num_qubits))
        diffuser_gate = diffuser.to_gate()
        diffuser_gate.label = "Quasi-Diffuser"
        return diffuser_gate
    
        # Generate Quasicrystal Lattice Points
    def generate_quasicrystal_points(num_points):
        # Use the 3D generalization of the Penrose tiling method
        golden_ratio = (1 + np.sqrt(5)) / 2
        indices = np.arange(1, num_points + 1)
        theta = 2 * np.pi * indices / golden_ratio
        phi = np.arccos(1 - 2 * indices / num_points)
        x = np.cos(theta) * np.sin(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(phi)
        return x, y, z

    # Create the Quantum-Inspired Search Path
    def create_search_path(x, y, z):
        num_points = len(x)
        # Select a subset of points to create a path
        path_indices = np.linspace(0, num_points - 1, int(num_points / 10), dtype=int)
        path_x = x[path_indices]
        path_y = y[path_indices]
        path_z = z[path_indices]

        # Smooth the path using spline interpolation
        tck, u = splprep([path_x, path_y, path_z], s=2)
        u_new = np.linspace(u.min(), u.max(), 400)
        smooth_path = splev(u_new, tck)
        return smooth_path

    # Plotting Function
    def plot_quasicrystal_with_path():
        # Generate lattice points
        num_points = 1000
        x, y, z = generate_quasicrystal_points(num_points)

        # Generate search path
        path_x, path_y, path_z = create_search_path(x, y, z)

        # Create 3D plot
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Plot lattice points
        scatter = ax.scatter(x, y, z, c=z, cmap='viridis', alpha=0.7, s=20, label='Grant Data Points')

        # Plot search path
        ax.plot(path_x, path_y, path_z, color='red', linewidth=2, alpha=0.8, label='Quantum Search Path')

        # Enhance visual appeal
        ax.set_title('Quantum-Inspired Grant Search in a Quasicrystal Lattice')
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')

        # Add color bar
        cbar = fig.colorbar(scatter, shrink=0.5, aspect=10)
        cbar.set_label('Depth (Z-axis)')

        # Add legend
        ax.legend()

        # Adjust viewing angle
        ax.view_init(elev=30, azim=120)

        plt.show()

    def encode_query(self, query):
        '''
        encode_query:
        [
            1. Query Processing: Splits user query into lowercase terms and lemmatizes them.
            2. Matching Logic: For each grant, lemmatize the grant terms and compute the overlap with lemmatized query terms.
                i) Exact Match: Grants where all query terms are present after lemmatization.
                ii) Partial Match: Grants with some overlap, scored based on the proportion of matching terms.
            Returns:
                matching_indices: Indices of grants with exact matches.
                partial_match_scores: List of tuples containing grant indices and their respective match scores for partial matches.
        ]
        '''
        # Initialize lemmatizer
        lemmatizer = WordNetLemmatizer()
        
        # Remove punctuation from query terms and lemmatize them
        translator = str.maketrans('', '', string.punctuation)
        query_terms = nltk.word_tokenize(query.lower().translate(translator))
        lemmatized_query_terms = [lemmatizer.lemmatize(term) for term in query_terms]
        query_terms_set = set(lemmatized_query_terms)
        
        matching_indices = []
        partial_match_scores = []
        
        for i, grant in enumerate(self.grants):
            grant_text = f"{grant.title} {grant.description} {grant.location}".lower()
            # Remove punctuation from grant text and lemmatize the terms
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

    def create_oracle(self, indices):
        '''
        create_oracle: Constructs the Oracle gate used in Grover's search, flipping the phase of states corresponding to target indices (matching grants).
        '''
        oracle = QuantumCircuit(self.num_qubits)
        if not indices:
            # No indices to flip; return identity oracle
            return oracle.to_gate()
        
        # Create an oracle that flips the phase of the states corresponding to indices
        oracle_matrix = np.identity(2 ** self.num_qubits)
        for index in indices:
            oracle_matrix[index][index] = -1
        
        # Convert the oracle matrix to an operator
        oracle_operator = Operator(oracle_matrix)
        oracle_gate = oracle_operator.to_instruction()
        return oracle_gate

    def create_diffuser(self):
        '''
        create_diffuser: Implements Grover diffuser (inversion about mean), amplifying probability amplitudes of target states.
        '''
        diffuser = QuantumCircuit(self.num_qubits)
        diffuser.h(range(self.num_qubits))
        diffuser.x(range(self.num_qubits))
        if self.num_qubits == 1:
            diffuser.z(0)
        else:
            diffuser.h(self.num_qubits - 1)
            diffuser.mcx(list(range(self.num_qubits - 1)), self.num_qubits - 1)
            diffuser.h(self.num_qubits - 1)
        diffuser.x(range(self.num_qubits))
        diffuser.h(range(self.num_qubits))
        return diffuser.to_gate()

    def search(self, query):
        '''
        search: Performs the quantum search using Grover's algorithm.
        '''
        matching_indices, partial_match_scores = self.encode_query(query)

        if matching_indices:
            indices_to_search = matching_indices
            print("Exact matches found. Performing quantum search...")
        elif partial_match_scores:
            # Sort grants by match score in descending order
            partial_match_scores.sort(key=lambda x: x[1], reverse=True)
            top_matches = [index for index, score in partial_match_scores[:3]]  # Get top 3 partial matches
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
        oracle = self.create_quasi_oracle(indices_to_search)
        diffuser = self.create_quasi_diffuser()

        # Determine the number of iterations
        num_iterations = int(np.round(np.pi / 4 * np.sqrt(2 ** self.num_qubits / len(indices_to_search))))
        if num_iterations == 0:
            num_iterations = 3  # Ensure at least one iteration

        # Initialize the statevector
        initial_state = Statevector.from_label('0' * self.num_qubits)

        # Apply Hadamard gates to create superposition
        state = initial_state.evolve(qc)

        # Mapping from state labels to grant titles
        state_to_grant = {}
        for i in range(len(self.grants)):
            state_label = format(i, f'0{self.num_qubits}b')
            state_to_grant[state_label] = self.grants[i].title

        # Collect probabilities for plotting
        probabilities_list = []
        probabilities = state.probabilities_dict()
        probabilities_list.append(probabilities)

        # Apply Grover's algorithm iterations
        for iteration in range(num_iterations):
            # Apply Oracle
            qc.append(oracle, qr)
            state = state.evolve(oracle)
            # Apply Diffuser
            qc.append(diffuser, qr)
            state = state.evolve(diffuser)
            # Compute probabilities according to Born's rule
            probabilities = state.probabilities_dict()
            probabilities_list.append(probabilities)

            # Print the probability currents
            print(f"Iteration {iteration + 1}:")
            for state_str in sorted(probabilities):
                prob = probabilities[state_str]
                grant_title = state_to_grant.get(state_str, None)
                if grant_title:
                    print(f"  State {state_str} ({grant_title}): Probability = {prob:.4f}")
            print("-" * 50)

        # Measurement
        qc.measure(qr, cr)

        # Transpile the circuit for the backend
        transpiled_qc = transpile(qc, self.backend)

        # Execute the circuit
        job = self.backend.run(transpiled_qc, shots=1024)
        result = job.result()
        counts = result.get_counts()

        # Map counts to grant titles
        counts_grants = {}
        for state_str, count in counts.items():
            grant_index = int(state_str, 2)
            if grant_index < len(self.grants):
                grant_title = self.grants[grant_index].title
                counts_grants[grant_title] = counts_grants.get(grant_title, 0) + count
            else:
                continue  # Skip invalid indices

        if counts_grants:
            # Find the most frequent result
            max_count = max(counts_grants.values())
            most_common_titles = [title for title, count in counts_grants.items() if count == max_count]
            best_grant_title = most_common_titles[0]
            # Find the grant with this title
            for grant in self.grants:
                if grant.title == best_grant_title:
                    best_grant = grant
                    break
        else:
            print("No valid grant found in measurement results.")
            return

        # Output the result
        print(f"\nThe most relevant grant based on your query '{query}' is:")
        print(f"Title: {best_grant.title}")
        print(f"Description: {best_grant.description}")
        print(f"Amount: ${best_grant.amount}")
        print(f"Location: {best_grant.location}")

        # If there were partial matches, list them
        if not matching_indices and partial_match_scores:
            print("\nOther potential candidate grants:")
            for index, score in partial_match_scores[:3]:
                grant = self.grants[index]
                print(f"- Title: {grant.title}, Match Score: {score:.2f}")

        # Plot histogram of measurement results with grant titles
        plot_histogram(counts_grants)
        plt.show()

        # Plot probability currents with stepper functionality
        self.plot_with_stepper(probabilities_list, state_to_grant, indices_to_search)

    def plot_with_stepper(self, probabilities_list, state_to_grant, indices_to_search):
        '''
        plot_with_stepper: Creates an interactive plot with stepper functionality to move back and forth between iterations.
        '''
        # Prepare data for plotting
        iterations = range(len(probabilities_list))
        grant_titles = [self.grants[i].title for i in range(len(self.grants))]
        grant_indices = list(range(len(self.grants)))
        num_frames = len(probabilities_list)
        current_frame = [0]  # Use a mutable object to allow modification inside nested functions

        # Create a figure and axis for the plot
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(grant_indices, [0]*len(self.grants), tick_label=grant_titles)
        ax.set_ylim(0, 1)
        ax.set_ylabel('Probability')
        ax.set_title('Probability Evolution Over Iterations')
        plt.xticks(rotation=90)
        ax.set_xlabel(f'Iteration {current_frame[0]}')

        # Initialize the bars
        def init():
            y = []
            probabilities = probabilities_list[0]
            for grant_index in range(len(self.grants)):
                state_label = format(grant_index, f'0{self.num_qubits}b')
                prob = probabilities.get(state_label, 0)
                y.append(prob)
            for bar, prob in zip(bars, y):
                bar.set_height(prob)
                if prob > 0.01:
                    bar.set_color('green' if grant_index in indices_to_search else 'blue')
                else:
                    bar.set_color('grey')
            return bars

        init()

        # Function to update the bars for each frame
        def update(frame):
            y = []
            probabilities = probabilities_list[frame]
            for grant_index in range(len(self.grants)):
                state_label = format(grant_index, f'0{self.num_qubits}b')
                prob = probabilities.get(state_label, 0)
                y.append(prob)
            for bar, prob in zip(bars, y):
                bar.set_height(prob)
                if prob > 0.01:
                    bar.set_color('green' if grant_index in indices_to_search else 'blue')
                else:
                    bar.set_color('grey')
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
        plt.show()

# Main Function
def main():
    '''
    Functionality:
        i) Argument Parsing: Accepts a search query as a command-line argument.
        ii) Validation: Ensures query is given, otherwise exits.
        iii) Execution: Creates a QuantumGrantSearcher and invokes the search on the query from arguments.
    '''
    parser = argparse.ArgumentParser(description='Quantum Grant Search Tool using Grover\'s Algorithm')
    parser.add_argument('query', type=str, nargs='?', default='',
                        help='Search query to find relevant grants (e.g., "renewable energy Boston")')
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
