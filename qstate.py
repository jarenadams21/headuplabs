# quantum_grant_search_quantum.py

# Imports and Dependencies
import sys
import numpy as np
import argparse
import matplotlib.pyplot as plt
import string
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
from qiskit.quantum_info import Statevector, Operator
from matplotlib.widgets import Button
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# Ensure NLTK resources are downloaded
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

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

    def create_quasi_oracle(self, indices):
        '''
        Create a quasicrystal-inspired oracle that applies custom phase shifts.
        '''
        if not indices:
            # No indices to flip; return identity oracle
            return Operator(np.identity(2 ** self.num_qubits))

        # Create a phase shift function based on quasicrystal properties
        phase_shifts = np.ones(2 ** self.num_qubits, dtype=complex)
        for index in indices:
            # Apply a custom phase shift
            phase_shifts[index] = np.exp(2j * np.pi * self.quasicrystal_phase(index))

        # Construct the oracle matrix
        oracle_matrix = np.diag(phase_shifts)
        oracle_operator = Operator(oracle_matrix)
        return oracle_operator

    def quasicrystal_phase(self, index):
        '''
        Define a phase function inspired by quasicrystal patterns.
        '''
        # Using the golden ratio
        golden_ratio = (1 + np.sqrt(5)) / 2
        phase = (index * golden_ratio) % 1  # Fractional part
        return phase

    def create_quasi_diffuser(self):
        '''
        Create a diffuser that accounts for quasicrystal-inspired superpositions.
        '''
        diffuser = QuantumCircuit(self.num_qubits)
        diffuser.h(range(self.num_qubits))
        diffuser.x(range(self.num_qubits))
        # Apply a multi-controlled phase gate with custom angles
        angle = 2 * np.pi / (2 ** self.num_qubits)

        if self.num_qubits == 1:
            # For a single qubit, apply a phase shift directly
            diffuser.p(angle, 0)
        else:
            # For multiple qubits, apply a multi-controlled phase gate
            diffuser.mcp(angle, list(range(self.num_qubits - 1)), self.num_qubits - 1)

        diffuser.x(range(self.num_qubits))
        diffuser.h(range(self.num_qubits))
        diffuser_gate = diffuser.to_gate()
        diffuser_gate.label = "Quasi-Diffuser"
        return diffuser_gate

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

        for i, grant in enumerate(self.grants):
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

    def search(self, query):
        '''
        Perform the quantum search using the quasicrystal-inspired Grover's algorithm.
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
        oracle_operator = self.create_quasi_oracle(indices_to_search)
        diffuser_gate = self.create_quasi_diffuser()

        num_iterations = 4  # Use a constant number of iterations
        if num_iterations == 0:
            num_iterations = 1  # Ensure at least one iteration

        # Initialize the statevector
        state = Statevector.from_label('0' * self.num_qubits)
        state = state.evolve(qc)

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
            qc.append(oracle_operator.to_instruction(), qr)
            state = state.evolve(oracle_operator)
            # Apply Diffuser
            qc.append(diffuser_gate, qr)
            state = state.evolve(diffuser_gate)
            # Compute probabilities
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

            # **Inner Sensing Routine**
            self.inner_sensing(probabilities)

        # Measurement
        qc.measure(qr, cr)

        # Transpile and execute the circuit
        transpiled_qc = transpile(qc, self.backend)
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

        if counts_grants:
            max_count = max(counts_grants.values())
            most_common_titles = [title for title, count in counts_grants.items() if count == max_count]
            best_grant_title = most_common_titles[0]
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
        plt.show(block=False)

        # Plot probability currents with stepper functionality
        self.plot_with_stepper(probabilities_list, state_to_grant, indices_to_search)

        # Plot the quasicrystal visualization connected to the search
        self.plot_quasicrystal_with_probabilities(probabilities_list, indices_to_search)

    def plot_with_stepper(self, probabilities_list, state_to_grant, indices_to_search):
        '''
        Creates an interactive plot with stepper functionality to move back and forth between iterations.
        '''
        # Prepare data for plotting
        grant_titles = [self.grants[i].title for i in range(len(self.grants))]
        grant_indices = list(range(len(self.grants)))
        num_frames = len(probabilities_list)
        current_frame = [0]  # Mutable object to allow modification inside nested functions

        # Create a figure and axis for the plot
        fig, ax = plt.subplots(figsize=(12, 6))
        bar_width = 0.8  # Increase bar width
        bars = ax.bar(grant_indices, [0]*len(self.grants), bar_width, tick_label=grant_titles)
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
            for bar, prob, idx in zip(bars, y, grant_indices):
                bar.set_height(prob)
                bar.set_color('green' if idx in indices_to_search else 'blue')
                if prob > 0.01:
                    bar.set_alpha(1.0)
                else:
                    bar.set_alpha(0.2)
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
            for bar, prob, idx in zip(bars, y, grant_indices):
                bar.set_height(prob)
                bar.set_color('green' if idx in indices_to_search else 'blue')
                if prob > 0.01:
                    bar.set_alpha(1.0)
                else:
                    bar.set_alpha(0.2)
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
        ax.scatter(target_x, target_y, target_z, c='red', s=100, label='Target Grants')

        # Enhance visual appeal
        ax.set_title('Quasicrystal Lattice Representation of Quantum States')
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')

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

    # Define the Inner Sensing Method
    def inner_sensing(self, probabilities):
        '''
        Perform an inner sensing routine to analyze quantum information after each iteration.
        '''
        print("Inner Sensing: Performing remote sensing analysis on quantum information...")
        
        # Entropy Calculation: Measure the uncertainty in the quantum state
        entropy = -sum(prob * np.log2(prob) for prob in probabilities.values() if prob > 0)
        print(f"  Quantum State Entropy: {entropy:.4f} bits")
        
        #! Significant Probability Shifts: Identify states with high probabilities (>5%)
        significant_states = {state: prob for state, prob in probabilities.items() if prob > 0.05}
        if significant_states:
            print("  Significant Probability Shifts Detected:")
            for state, prob in significant_states.items():
                grant_title = self.get_grant_title(state)
                print(f"    - State {state} ({grant_title}): Probability = {prob:.4f}")
        else:
            print("  No significant probability shifts detected.")
        
        # Geographical Distribution Analysis: Analyze high-probability grants' locations
        high_prob_grants = [self.get_grant_title(state) for state in significant_states]
        locations = [grant.location for grant in self.grants if grant.title in high_prob_grants]
        location_counts = {}
        for loc in locations:
            location_counts[loc] = location_counts.get(loc, 0) + 1
        
        if location_counts:
            print("\n  Geographical Distribution of High-Probability Grants:")
            for location, count in location_counts.items():
                print(f"    - {location}: {count} grant(s)")
        else:
            print("  No high-probability grants to analyze for geographical distribution.")
        
        print("Inner Sensing Complete.\n")

    def get_grant_title(self, state_label):
        '''
        Retrieve the grant title based on the state label.
        '''
        grant_index = int(state_label, 2)
        if grant_index < len(self.grants):
            return self.grants[grant_index].title
        return "Unknown Grant"

# Main Function
def main():
    parser = argparse.ArgumentParser(description='Quantum Grant Search Tool using a Quasicrystal-Inspired Algorithm')
    parser.add_argument('query', type=str, nargs='?', default='',
                        help='Search query to find relevant grants (e.g., "renewable energy Boston")')
    parser.add_argument('--wiki', action='store_true',
                        help='If set, perform a Wikipedia search for the query instead of searching grants')
    args = parser.parse_args()

    if not args.query:
        print("Please provide a search query.")
        sys.exit(1)

    if args.wiki:
        perform_wikipedia_search(args.query)
    else:
        # Initialize the Quantum Grant Searcher
        searcher = QuantumGrantSearcher(grants)

        # Perform the quantum search
        searcher.search(args.query)

# Function to perform Wikipedia search
def perform_wikipedia_search(query):
    import wikipedia

    try:
        # Attempt to get the exact page
        page = wikipedia.page(query)
        print(f"\nWikipedia Page Found:\nTitle: {page.title}\nURL: {page.url}\n")
    except wikipedia.DisambiguationError as e:
        # If a disambiguation page is encountered, choose the first suggested option
        print(f"Disambiguation Error: The query '{query}' may refer to multiple pages.")
        suggested_title = e.options[0]
        page = wikipedia.page(suggested_title)
        print(f"\nSuggested Wikipedia Page:\nTitle: {page.title}\nURL: {page.url}\n")
    except wikipedia.PageError:
        # If no page matches the query, perform a search and suggest the best match
        print(f"No exact Wikipedia page found for '{query}'. Searching for related pages...")
        search_results = wikipedia.search(query)
        if search_results:
            best_match = search_results[0]
            try:
                page = wikipedia.page(best_match)
                print(f"\nBest Matching Wikipedia Page:\nTitle: {page.title}\nURL: {page.url}\n")
            except Exception as e:
                print(f"An error occurred while fetching the page: {e}")
        else:
            print("No related Wikipedia pages found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
